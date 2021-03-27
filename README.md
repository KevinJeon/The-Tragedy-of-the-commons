# The-Tragedy-of-the-commons
This repo is implementation of environment in paper 'Social diversity and social preferences in mixed-motive reinforcement learning'


## Environment Specification
### Settings
Placeholder

### Observation
Placeholder

### Action


### Info
#### Agents
##### Directions
Which directions the agents are looking for  

WIP

## Example
```python
env = TOCEnv(num_agents=4) # n=4

actions = [action_1, action_2, action_3 ... action_n]
obs, reward, info = env.step(actions)
```


## TODO
### Environment
- [X] Reset map on there's no apple one the map
- [X] Add environment parameter for `episode_max_length`
- [ ] Generate patch spawn area follow some distribution
- [X] Re-spawn apples following quantity of surrounding apples
- [X] Return agent's individual views on `step` called
- [X] Implement Agent's direction and make it available when action called on `step`
- [X] Implement action `Punish`

### Algorithms
- [X] Make CPC agent
- [ ] Make CPC Module
- [ ] Debug and Train CPC
- [ ] 
