# Order-Optimal Global Convergence for Average Reward Actor-Critic with General Policy and Neural Critic

## Required environments:
- on Ubuntu 20.04
- Python 3.9.19
- mujoco_py 2.1.2
- tensorboard 2.18.0
- ...

## How to run our algorithm

```bash
cd nac_dd
python main.py --drop_num W --seed X --algorithm Y --env Z
```
- W is the drop number for which we choose from [1, 3, 5];

- X is the random seed for which we choose from [0, 500, 1000];

- Y is the algorithm for which we set as NACDD;

- Z is the environment which can be one of [Hopper-v3, HalfCheetah-v3, Walker2d-v3].

## How to run the baselines

```bash
cd pg_travel/mujoco
python main.py --seed X --algorithm Y --env Z
```

- X is the random seed for which we choose from [0, 500, 1000];

- Y is one of the baselines: [PG, NPG, TRPO, PPO];

- Z is the environment which can be one of [Hopper-v3, HalfCheetah-v3, Walker2d-v3].

[PG, NPG, TRPO, PPO] correspond to the following policy gradient (PG) algorithms:
* Vanilla Policy Gradient: R. Sutton, et al., "Policy Gradient Methods for Reinforcement Learning with Function Approximation", NIPS 2000.
* Truncated Natural Policy Gradient: S. Kakade, "A Natural Policy Gradient", NIPS 2002.
* Trust Region Policy Optimization: J. Schulman, et al., "Trust Region Policy Optimization", ICML 2015.
* Proximal Policy Optimization: J. Schulman, et al., "Proximal Policy Optimization Algorithms", arXiv, https://arxiv.org/pdf/1707.06347.pdf.

## How to reproduce the results

The raw training logs are available at 'nac-dd/results'. The scripts for making the polts are 'nac_dd/draw_figure.py' and 'nac_dd/draw_subfigures.py'.



## Reference
We referenced the codes from:
* [pg_travel](https://github.com/reinforcement-learning-kr/pg_travel)
