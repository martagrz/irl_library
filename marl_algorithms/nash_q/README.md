# Implementation of Nash Q-Learning 

Nash Q-Learning (or Nash-Q) [(Hu and Wellman, 2003)](http://www.jmlr.org/papers/v4/hu03a.html) allows agents to learn simultaneously in a multi-agent setting by constructing a belief over the actions of the other players. It follows the single-agent method, where an agent chooses an action based on some policy and updates Q-values based on the rewards they have observed. It differs in two major ways. Firstly, in that the update rule incorporates a Nash equilibirum component, which uses the policies of the other players. However, an agent does not know the policies of the other players and has to approximate them based on the observed rewards and actions. This leads to the second difference, where an agent keeps a Q-table for their state-action pairs and also for all the other agents in the game. For a full explanation of the algorithm, see here. 

We implement this algorithm in a simple grid-world setting with two agents and a goal state each, where both agents cannot occupy the same cell in the grid. 


