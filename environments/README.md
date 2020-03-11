# Environments 

---

## Implemented: 
* Gridworld 

## To do: 
* Objectworld 
* Dynamic Pricing 
* Highway Driving 

---

Each environment fully specifies an MDP and is defined by a **class** which, in general, takes as input: 
* states
* actions 
* discount rate, gamma 
* goal state
* initialisation state

`env.get_transitions` takes as input a state-action pair and returns 
* a vector of possible next states 
* the corresponding transition probabilities

`env.step` takes as input a state-action pair and returns 
* a realised next state 
* the corresponding reward value


