# Pong-Python (Self-Play)

## About
This repository contains a collection of Reinforcement Learning algorithms which are trained with 
the self-play method to play on my [Pong clone](https://github.com/bottamichele/Pong-Python) game.

The "dueling_dqn.py" file contains the code which uses the Dueling DQN algorithm and 
is mainly based on [my old repository work](https://github.com/bottamichele/Pong-Python-RL).

The "ppo.py" file contains the code which uses the PPO algorithm and trains the agent 
with its latest version of policy.

## Libraries
The following libraries were used for this project:
- [dqn-algorithms](https://github.com/bottamichele/dqn-algorithms) v1.0
- [ppo-algorithm](https://github.com/bottamichele/ppo-algorithm) v1.0.1
- [pong_gym](https://github.com/bottamichele/Pong-Gym) v1.0
- [pong_pz](https://github.com/bottamichele/Pong-PZ) v1.0
- gymnasium v1.1.1
- pettingzoo v1.24.3
- SuperSuit v3.10.0
- numpy v2.1.2
- torch v2.6.0
- tensorboard v2.19.0
