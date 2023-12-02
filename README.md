# Discrete Lot Sizing

The project implements the Discrete Lot Sizing Problem (DLSP).

It is organized in the following modules:

- models:  it contains the math models
- agents
- envs

the configuration file is:

~~~ json
{
    "time_horizon": 3,
    "n_items": 1,
    "n_machines": 1,"initial_setup": [0],
    "machine_production": [[10]],
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

  
### ToDo

[] controlla su più istanze

[] check consistenza dei tempi. Per es. il 2 stadi non funziona perchè inizio in t=0 e la decisione di cambio si stato influenza t=1 istante in cui non produco (dovrei considerare $x_{ij}^{a}$ in Eq 9)

[] Interpretazione univoca dello 0 (stato per le macchine o produzione item 0?  Attulamente gestito da env.)

[] vettorializzare modelli



# Code

main files descriptions:

- *multistage_vs_exact_vi* compares the multistave vs the exact value iteration.
