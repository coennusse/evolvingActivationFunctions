# Evolving Activation Functions

This project involves evolving activation functions for neural networks using evolutionary strategies like OpenES, CMA-ES, PSO, and DE. The goal is to optimize the parameters of custom activation functions to improve training performance on different environments.

## Directory Structure
- `main_activation_function.py`: The main script to run the activation function evolution.
- `make_train_activation_function.py`: Contains functions to initialize parameters and train the models.
- `run_evolved_model.py`: Script to evaluate the evolved model.
- `wrappers.py`: Additional utility functions.
- `configs/`: Directory for configuration files.
- `models/`: Directory to save trained models.
- `utils/`: Utility functions and helpers.
- `requirements.txt`: List of required Python packages.
- `LICENSE`: License for the project.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/coennusse/jax_evolve_activation.git
    cd  jax_evolve_activation
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Make sure you have a version of jaxlib installed that is compatible with your device.

## Usage
### Running the Evolution
To run the evolution process, use:
```bash
python main_activation_function.py --config configs/default_config.yaml
```

To evaluate the evolved model, use:
```bash
python run_evolved_model.py --model models/best_model.pkl
```

## Citation
If you use this project in your research, please cite the following paper:

```bash
@inproceedings{Nussekooiactivations,
  title={Learning Nonlinear Activation Functions in RL Through Evolutionary Computation},
  author={Coen, Nusse and Kooi, Jacob E.},
  booktitle={Symposium on Computational Intelligence in Artificial Life and Cooperative Intelligent Systems Companion},
  year={2025}
}
```
