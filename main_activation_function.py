import jax.numpy as jnp
import jax
from evosax import OpenES, CMA_ES, PSO, DE
import wandb
import argparse
import time
import pickle
import yaml
import plotly.graph_objs as go

from utils.helpers import parse_config, initialize_logging, setup_strategy, save_model
from make_train_activation_function import init_params, make_train, NonLinearActivation

def rollout_function(train_fn):
    """Defines the rollout function."""
    def single_rollout(rng_input, activation_params):
        """Perform a single rollout and return the mean episode return."""
        out = train_fn(activation_params, rng_input)
        return out["metrics"]['returned_episode_returns'].mean()
    
    vmap_rollout = jax.vmap(single_rollout, in_axes=(0, None))
    return jax.jit(jax.vmap(vmap_rollout, in_axes=(None, 0)))

def run_meta_evolution(config, rng, activation_params_pholder, param_shapes, rollout):
    """Run the meta-evolution process."""
    strategy, state = setup_strategy(config, rng, activation_params_pholder)
    best = 0
    best_member = None

    for gen in range(config["NUM_GENERATIONS"]):
        t0 = time.time()
        rng, rng_ask, rng_eval = jax.random.split(rng, 3)
        x, state = jax.jit(strategy.ask)(rng_ask, state)
        
        batch_rng = jax.random.split(rng_eval, config["NUM_ROLLOUTS"])
        fitness = rollout(batch_rng, x).mean(axis=1)
        
        state = jax.jit(strategy.tell)(x, fitness, state)
        fitness = jax.block_until_ready(fitness)
        
        metric = {
            "generation": gen,
            "fitness": fitness.mean(),
            "best": state.best_fitness,
            "time": time.time() - t0
        }
        
        if state.best_fitness > best:
            best = state.best_fitness
            best_member = state.best_member

        print(f"Generation: {gen}, Fitness: {fitness.mean():.2f}, Best: {state.best_fitness:.2f}")
            
        if not config["LOG"]:
            continue
        
        best_member = list(jnp.array(best_member))
        best_member = [arr.item() for arr in best_member]

        activation_model = {}
        index = 0

        for key, shape in param_shapes.items():
            size = jnp.prod(jnp.array(shape))
            values_slice = best_member[index:index + size]
            activation_model[key] = jnp.array(values_slice).reshape(shape)
            index += size

        # Create a Plotly figure
        fig = go.Figure()

        activation_net_ins = jnp.linspace(-2.0, 2.0, 100)

        activation_net_outs = jax.vmap(NonLinearActivation, in_axes=(None, 0, None))(activation_model, activation_net_ins, config["INNER_ACTIVATION"])
            
        fig.add_trace(go.Scatter(x=activation_net_ins, y=activation_net_outs, mode='lines', name=f'Learned Function'))

        title = f"Activation with {config['INNER_ACTIVATION']}"

        # Update layout
        fig.update_layout(title=title,
                        xaxis_title='x',
                        yaxis_title='f(x)',
                        showlegend=True,
                    legend=dict(x=0, y=1))

        metric["plot"] = fig

        wandb.log(metric)
    
    return best_member, best_fitness

def save_model(best_member, param_shapes, config, best_fitness):
    """Save the model to a file."""
    best_member = [arr.item() for arr in jnp.array(best_member)]
    activation_model = {}
    index = 0

    for key, shape in param_shapes.items():
        size = jnp.prod(jnp.array(shape))
        values_slice = best_member[index:index + size]
        activation_model[key] = jnp.array(values_slice).reshape(shape)
        index += size

    model_path = f'models/{config["ENV_NAME"]}-{best_fitness:.2f}.pkl'
    
    with open(model_path, "wb") as f:
        pickle.dump(activation_model, f)

def main():
    parser = argparse.ArgumentParser(description="Run activation function evolution.")
    parser.add_argument("--config", type=str, help="Path to the configuration YAML file.", default="configs/default_config.yaml")
    args = parser.parse_args()

    config = parse_config(args.config)
    if config["LOG"]:
        initialize_logging(config)

    activation_map = {
        'tanh': 0,
        'sigmoid': 1,
        'relu': 2,
        'leaky_relu': 3
    }

    config["INNER_ACTIVATION"] = activation_map[config["INNER_ACTIVATION"]]
    
    rng = jax.random.PRNGKey(config["SEED"])
    activation_params_pholder = init_params(rng, num_nodes=config["NUM_NODES"], num_hidden_layers=config["NUM_HIDDEN_LAYERS"])
    param_shapes = {
        "b_hidden": (config["NUM_NODES"],),
        "b_output": (1,),
        "w_hidden": (1, config["NUM_NODES"]),
        "w_output": (config["NUM_NODES"], 1)
    }
    
    train_fn = make_train(config)
    rollout = rollout_function(train_fn)
    
    best_member, best_fitness = run_meta_evolution(config, rng, activation_params_pholder, param_shapes, rollout)
    save_model(best_member, param_shapes, config, best_fitness)

if __name__ == "__main__":
    main()
