# The-Tragedy-of-the-commons
This repo is implementation of environment in paper 'Social diversity and social preferences in mixed-motive reinforcement learning'


```python
env = TOCEnv(num_agents=4) # n=4

actions = [action_1, action_2, action_3 ... action_n]
obs, reward, info = env.step(actions)
```


## TODO
- [ ] Reset map on there's no apple one the map
- [ ] Add environment parameter for `episode_max_length`
- [ ] Generate patch spawn area follow some distribution
- [ ] Re-spawn apples following quantity of surrounding apples
- [ ] Return agent's individual views on `step` called
- [ ] Implement Agent's direction and make it available when action called on `step`
- [ ] Implement action `Punish`
