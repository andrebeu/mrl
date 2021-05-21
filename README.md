# Meta Reinforcement Learning on bandits task.
Simple demo implementation of meta-RL in pytorch

### Repo materials
- `utils.py` defines the agent and task environment
- `main.ipynb` has a simple training loop, and an evaluation of the learnt agent


### Task
- Two variants of meta-learning on bandits are implemented:
  - easy: bandits switch between episodes. i.e. for a given episode, the bandit with highest probability is fixed
  - medium: bandit switches once at given episode step
  - difficult: bandit switches within an episode with probability p. 

### Agent
- policy gradient method
  - target can be REINFORCE (MC) or ActorCritic (TD)

### Optimizations
- forward layers computed in parallel by folding time into batch dimension

### Resources
[Wang et al., 2018 paper](https://www.nature.com/articles/s41593-018-0147-8.pdf?proof=t)

