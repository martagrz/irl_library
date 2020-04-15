# Algorithms 

---

## Implemented: 
* LinearMDP
* MaxEnt 
* GP-IRL
* DeepMaxEnt IRL
* Linear IRL

## To do: 
* DMGP-IRL
* NP-IRL
* GAIL

---

## MDP solvers 
MDP solver used to generate synthetic examples
* LinearMDP - based on LMDP framework
* StandardMDP - based on standard Bellman equations

## IRL Algorithms 
* MaxEnt - Maximum Entropy 
* GP-IRL - Gaussian process 

---

Each IRL algorithm directory contains the following files: 
* model for extracting reward function (e.g. NN, GP)
* the algorithm itself
* testing script

In utils, there is: 
* get_statistics
* policy propagation

