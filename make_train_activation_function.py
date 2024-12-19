import jax
from jax import random
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
import gymnax
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper
from wrappers import BraxGymnaxWrapper, LogWrapper, ClipAction, VecEnv, NormalizeVecObservation, NormalizeVecReward
import wandb
import flax
    
def init_params(key, num_nodes=32, num_hidden_layers=1): 
    w_hidden = random.normal(key, (1, num_nodes))
    b_hidden = jnp.zeros((num_nodes,))
    
    # Initialize weights and biases for the output layer
    w_output = random.normal(key, (num_nodes, 1))
    b_output = jnp.zeros((1,))
    
    out = {"w_hidden": w_hidden, 
            "b_hidden": b_hidden, 
            "w_output": w_output, 
            "b_output": b_output}
    
    if num_hidden_layers > 1:
        for i in range(1, num_hidden_layers):
            out[f"w_hidden{i}"] = random.normal(key, (num_nodes, num_nodes))
            out[f"b_hidden{i}"] = b_hidden
    
    return out
    
def NonLinearActivation(params, x, inner_activation, num_hidden_layers=1):
    # Unpack parameters
    w_hidden = params["w_hidden"] 
    b_hidden = params["b_hidden"]
    w_output = params["w_output"] 
    b_output = params["b_output"]
    
    # Hidden layer computation
    hidden_layer = jnp.dot(x, w_hidden) + b_hidden
    
    def tanh_activation(x):
        return jnp.tanh(x)
    
    def sigmoid_activation(x):
        return 1 / (1 + jnp.exp(-x))
    
    def relu_activation(x):
        return jnp.maximum(0, x)
    
    def leaky_relu_activation(x):
        return jnp.where(x > 0, x, 0.01 * x)
    
    # List of activation functions
    activation_functions = [tanh_activation, sigmoid_activation, relu_activation, leaky_relu_activation]
    
    # Select the appropriate activation function
    hidden_output = jax.lax.switch(inner_activation, activation_functions, hidden_layer)
    
    if num_hidden_layers > 1:
        for i in range(1, num_hidden_layers):
            hidden_layer = jnp.dot(hidden_output, params[f"w_hidden{i}"]) + params[f"b_hidden{i}"]
            hidden_output = jax.lax.switch(inner_activation, activation_functions, hidden_layer)
        
    # Output layer computation
    output = jnp.dot(hidden_output, w_output) + b_output
    
    return jnp.sum(output)


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation_params: dict
    num_layers: int = 1
    inner_activation: str = "tanh"
    num_hidden_layers: int = 1
    activation: str = "nonlinear"
    
    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            actor_mean = nn.Dense(
                64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(x)
            
            sh = jnp.shape(actor_mean)
            
            if self.activation == "nonlinear":
                actor_mean = jnp.reshape(jax.vmap(NonLinearActivation, in_axes=(None, 0, None, None))(self.activation_params, jnp.ravel(actor_mean), self.inner_activation, self.num_hidden_layers), sh)
            elif self.activation == "relu":
                actor_mean = nn.relu(actor_mean)
            else:
                actor_mean = nn.tanh(actor_mean)
                
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        for _ in range(self.num_layers):
            critic = nn.Dense(
                64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(x)
            
            sh = jnp.shape(critic)
            
            if self.activation == "nonlinear":
                critic = jnp.reshape(jax.vmap(NonLinearActivation, in_axes=(None, 0, None, None))(self.activation_params, jnp.ravel(critic), self.inner_activation, self.num_hidden_layers), sh)
            elif self.activation == "relu":
                critic = nn.relu(critic)
            else:
                critic = nn.tanh(critic)    
                
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    
    if config["ENV_NAME"] in ["hopper", "ant"]:                            
        env, env_params = BraxGymnaxWrapper(config["ENV_NAME"]), None
        env = LogWrapper(env)
        env = ClipAction(env)
        env = VecEnv(env)
        if config["NORMALIZE_ENV"]:
            env = NormalizeVecObservation(env)
            env = NormalizeVecReward(env, config["GAMMA"])
    else:
        env, env_params = gymnax.make(config["ENV_NAME"])
        env = FlattenObservationWrapper(env)
        env = LogWrapper(env)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(activation_params, rng):
        try:
            n = env.action_space(env_params).n
        except AttributeError:
            n = env.action_space(env_params).shape[0]
            
        network = ActorCritic(n, num_layers=config["NN_LAYERS"], activation_params=activation_params, inner_activation=config["INNER_ACTIVATION"], num_hidden_layers=config["NUM_HIDDEN_LAYERS"], activation=config["ACTIVATION"])
        
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
            
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx = tx
        )
        
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        
        if config["ENV_NAME"] in ["hopper", "ant"]:                                
            obsv, env_state = env.reset(reset_rng, env_params)
        else:
            obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                
                if config["ENV_NAME"] in ["hopper", "ant"]:                                
                
                    obsv, env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)
                else:
                    obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
                        rng_step, env_state, action, env_params
                    )
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        if config["ENV_NAME"] in ["hopper", "ant"]:                                
                            # CALCULATE ACTOR LOSS
                            alpha = config["DPO_ALPHA"]
                            beta = config["DPO_BETA"]
                            log_diff = log_prob - traj_batch.log_prob
                            ratio = jnp.exp(log_diff)
                            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                            is_pos = (gae >= 0.0).astype("float32")
                            r1 = ratio - 1.0
                            drift1 = nn.relu(r1 * gae - alpha * nn.tanh(r1 * gae / alpha))
                            drift2 = nn.relu(
                                log_diff * gae - beta * nn.tanh(log_diff * gae / beta)
                            )
                            drift = drift1 * is_pos + drift2 * (1 - is_pos)
                            loss_actor = -(ratio * gae - drift).mean()
                            entropy = pi.entropy().mean()
                        else:
                            # CALCULATE ACTOR LOSS
                            ratio = jnp.exp(log_prob - traj_batch.log_prob)
                            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                            loss_actor1 = ratio * gae
                            loss_actor2 = (
                                jnp.clip(
                                    ratio,
                                    1.0 - config["CLIP_EPS"],
                                    1.0 + config["CLIP_EPS"],
                                )
                                * gae
                            )
                            loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                            loss_actor = loss_actor.mean()
                            entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            loss = loss_info[0]
            
            metric = traj_batch.info
            metric["loss"] = loss.mean()
            metric["returns"] = metric["returned_episode_returns"].mean()
            
            rng = update_state[-1]
            
            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train