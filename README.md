# PPO Algorithm for Deep Reinforcement Learning

This repository contains the implementation of the Proximal Policy Optimization (PPO) algorithm for executing Deep Reinforcement Learning (DRL) tasks in various environments, including a custom media environment.

## Repository Overview

- **ppo.py**: Core PPO algorithm implementation.
- **media_env/**: Custom media environment for testing.
- **video.mkv**: To be included as part of the training.
- **main.py**: Main script to execute the PPO algorithm.
- **data/**: Logging folder.

## Installation

To set up the environment and install the necessary dependencies, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/Kaiser-14/drl-ppo-media.git
    cd drl-ppo-media
    ```

2. Install the required packages and media environment:

    ```bash
    pip install -r requirements.txt
    pip install -e .
    ```

## Usage

To run the PPO algorithm, execute the `main.py` script:

```bash
python main.py
