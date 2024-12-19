import argparse
import pickle
import yaml
import jax
import jax.numpy as jnp
from make_train_activation_function import make_train

def evaluate_model(model_path, config):
    with open(model_path, "rb") as f:
        activation_model = pickle.load(f)

    train_fn = make_train(config)
    rng = jax.random.PRNGKey(config["SEED"])

    # Example evaluation rollout
    rollout_return = train_fn(activation_model, rng)
    print("Evaluation Return:", rollout_return["metrics"]["returned_episode_returns"].mean())

def main():
    parser = argparse.ArgumentParser(description="Run evolved model evaluation.")
    parser.add_argument("--model", type=str, required=True, help="Path to the saved model file.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to the configuration file.")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    evaluate_model(args.model, config)

if __name__ == "__main__":
    main()