# Discrete Lot Sizing

The project implements the Discrete Lot Sizing Problem (DLSP).

It is organized in the following modules:

- models:  it contains the math models
- agents : it contains different implementations of DRL, ADP, and heuristic methosd so solve the MDP model
- envs: The environment implamentations
- logs: logs for the models
- results: results folder to save the simulation results
- cfg_sol: it contains the solution setting for the multi-stage
- cfg_env: it contains different environment configurations

the configuration file is:

~~~ json
{
    "time_horizon":100,
    "n_items": 2,
    "n_machines": 1,
    "initial_setup": [0],
    "machine_production": [[3,3]],
    "max_inventory_level": [10, 10],
    "initial_inventory": [0, 0],
    "holding_costs": [1, 1],
    "lost_sales_costs": [10, 20],
    "demand_distribution": {
        "name": "probability_mass_function",
        "vals": [0,1,2],
        "probs": [0.341,0.58,0.079]
    },
    "demand_distribution_example_item_specific_uniform": {
        "name": "item_specific_uniform",
        "distributions": [
            {
                "name": "probability_mass_function",    
                "vals": [0,1,2],
                "probs": [0.341,0.58,0.079]
            },{
                "name": "probability_mass_function",    
                "vals": [0,1,2,3],
                "probs": [0.241,0.18,0.179,0.1 ]
            }
        ]  
    },
    "setup_costs": [[1, 1]],
    "setup_loss": [[1, 1]]
}
~~~


Flow of operations of time t:

1. demand realization ($d_t$)
2. change machine setup if needed
3. item production  $x_{ijt}$
4. demand satisfaction
5. holding costs computation
6. decision of next set up $\delta_{ijt}$

### Deterministic Model

The general mathematical model is:

$$
\min \sum_{t=0}^T \sum_{i=1}^n (h_iI_{it}+\sum_{j=1}^mf_i\delta_{ijt})
$$
s.t.
$$
I_{i, t+1} = I_{i, t} + \sum_{j=1}^m(p_ix_{ijt}-l_i\delta_{ijt}) -d_{it+1} i \in [n],t\in [0:T-1]
$$

$$
\sum_{i=1}^n x_{ijt} \leq 1\ \ j \in [m]\ t \in [0:T-1]
$$

$$
\sum_{j=1}^m x_{ijt} \leq 1\ \ i \in [n]\ \ t \in [0:T-1]
$$

$$
d_{ijt} \geq x_{ijt+1}-x_{ijt}\ i \in [n]\ j \in [m]\ t \in [0:T-1]
$$

$$
I_{it} \geq 0 \ \ i \in [n] t \in [0:T-1]
$$

$$
x_{ijt}, \delta_{ijt} \in \{0,1\}\ i \in [n]\ j \in [m]\ t \in [0:T-1]
$$

### Stochastic Model

Consider a tree and the following notation:

- $s \in \mathcal{S}$ is a generic node of the scenario tree;
- $\{0\}$ is the root node
- $a(s)$ is the immediate predecessor for node $s$

The model is

$$
\min \sum_{s} p^{[s]} \sum_{i=1}^n (h_iI_{i}^{[s]}+\rho_i z_{i}^{[s]}+\sum_{j=1}^mf_i\delta_{ij}^{[s]})
$$

s.t.

$$
I_{i}^{[s]} - z_{i}^{[s]} = I_{i}^{[a(s)]} + \sum_{j=1}^m(p_ix_{ij}^{[a(s)]}-l_i\delta_{ij}^{[a(s)]}) -d_{i}^{[s]}\ \ i \in [n], s \in \mathcal{S}-\{0\}
$$

$$
\sum_{i=1}^n x_{ij}^{[s]} \leq 1\ \ j \in [m]\ s \in \mathcal{S}
$$

$$
\sum_{j=1}^m x_{ij}^{[s]} \leq 1\ \ i \in [n]\ \ s \in \mathcal{S}-\{0\}
$$

$$
d_{ij}^{[a(s)]} \geq x_{ij}^{[s]}-x_{ij}^{[a(s)]}\ i \in [n]\ j \in [m]\ s \in \mathcal{S}-\{0\}
$$

$$
I_{i}^{[s]} \geq 0\ \ z_{i}^{[s]} \geq 0 \ \ i \in [n] s \in \mathcal{S}
$$

$$
x_{ij}^{[s]}, \delta_{ij}^{[s]} \in \{0,1\}\ i \in [n]\ j \in [m]\ s \in \mathcal{S}
$$

**Oss:** $d_i^{[0]}$ non viene considerata.

### Notes:

- in *simplePlant* we consider that the setup costs and setup time to go in the idlle state is 0.

  
## How to play test this code:

You can access ``test_files`` and choose or create a file to configure the experiment, including new models and changing the environment configurations.

To execute an experiment, simply run the following:

```
python -m test_files.2items_1machine
```

# References:

This project was built in partnership with Politecnico di Torino Researchers. The base environment here employed can also be accessed in [dicrete_lot_sizing](https://github.com/EdoF90/discrete_lot_sizing)

The paper can be downloaded in [Expert Systems with Applications](https://www.sciencedirect.com/science/article/pii/S0957417423035388)

To reference this work use these information:

```
@article{FELIZARDO2024123036,
title = {Reinforcement learning approaches for the stochastic discrete lot-sizing problem on parallel machines},
journal = {Expert Systems with Applications},
pages = {123036},
year = {2024},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2023.123036},
url = {https://www.sciencedirect.com/science/article/pii/S0957417423035388},
author = {Leonardo Kanashiro Felizardo and Edoardo Fadda and Emilio Del-Moral-Hernandez and Paolo Brandimarte},
keywords = {Dynamic programming, Stochastic programming, Multi-agent systems, Machine learning, Reinforcement Learning},
abstract = {This paper addresses the stochastic discrete lot-sizing problem on parallel machines, which is a computationally challenging problem also for relatively small instances. We propose two heuristics to deal with it by leveraging reinforcement learning. In particular, we propose a technique based on approximate value iteration around post-decision state variables and one based on multi-agent reinforcement learning. We compare these two approaches with other reinforcement learning methods and more classical solution techniques, showing their effectiveness in addressing realistic size instances.}
}
```

