import yaml
import wandb
import jax
from evosax import OpenES, CMA_ES, PSO, DE
import os
import pickle

def parse_config(config_path):
    """Parse YAML configuration file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def initialize_logging(config):
    """Initialize Weights & Biases logging."""
    wandb.init(
        project=config["PROJECT"],
        tags=["PPO", config["ENV_NAME"].upper(), f"jax_{jax.__version__}"],
        name=f'{config["ENV_NAME"]}_{config["EVO_ALG"]}_{config["POPSIZE"]}_{config["INNER_ACTIVATION"]}',
        config=config,
        mode=config["WANDB_MODE"],
    )

def setup_strategy(config, rng_init, activation_params_pholder):
    """Set up the evolutionary strategy."""
    strategies = {
        "OpenES": OpenES(popsize=config["POPSIZE"], pholder_params=activation_params_pholder, opt_name="adam", centered_rank=True, maximize=True),
        "CMA_ES": CMA_ES(popsize=config["POPSIZE"], pholder_params=activation_params_pholder, maximize=True),
        "PSO": PSO(popsize=config["POPSIZE"], pholder_params=activation_params_pholder, maximize=True),
        "DE": DE(popsize=config["POPSIZE"], pholder_params=activation_params_pholder, maximize=True)
    }
    strategy = strategies.get(config["EVO_ALG"], None)
    state = strategy.initialize(rng_init) if strategy else None
    return strategy, state

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

    os.makedirs("models", exist_ok=True)
    model_path = f'models/{config["ENV_NAME"]}-{best_fitness:.2f}.pkl'
    with open(model_path, "wb") as f:
        pickle.dump(activation_model, f)
    print(f"Model saved to {model_path}")
