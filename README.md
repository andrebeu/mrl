# Meta-RL
[Wang et al., 2018 paper](https://www.nature.com/articles/s41593-018-0147-8.pdf?proof=t)
Meta-RL agent on bandits task.
Two variants of meta-learning on bandits are implemented:

- easy: bandits switch between episodes. i.e. for a given episode, the bandit with highest probability is fixed
- bandits switch within episode
  - medium: bandit switches once at given episode step
  - difficult: bandit switches within an episode with probability p. 