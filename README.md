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
@INPROCEEDINGS{11030986,
  author={Nusse, Coen and Kooi, Jacob E.},
  booktitle={2025 IEEE Symposium on Computational Intelligence in Artificial Life and Cooperative Intelligent Systems Companion (ALIFE-CIS Companion)}, 
  title={Learning Nonlinear Activation Functions in RL Through Evolutionary Computation}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Adaptation models;Heuristic algorithms;Computational modeling;Reinforcement learning;Evolutionary computation;Learning (artificial intelligence);Complexity theory;Intelligent systems;Optimization;Faces;reinforcement learning},
  doi={10.1109/ALIFE-CISCompanion65078.2025.11030986}}
```

