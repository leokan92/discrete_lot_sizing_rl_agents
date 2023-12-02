# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout


def prod(val):  
    res = 1 
    for ele in val:  
        res *= ele  
    return res   


class ScenarioTree(nx.DiGraph):
    def __init__(self, name, depth, branching_factors, dim_observations, initial_value, stoch_model):
        # branching_factors: se specifico un solo branching factor lo assumo cotante per tutta la profondità dell'albero
        nx.DiGraph.__init__(self)
        self.starting_node = 0
        self.dim_observations = dim_observations
        self.stoch_model = stoch_model
        self.add_node(
            self.starting_node,
            obs=initial_value,
            prob=1,
            t=0,
            id=0,
            stage=0
        )
        self.name = name
        self.breadth_first_search = []
        self.filtration = []
        
        self.breadth_first_search = []
        self.depth = depth
        if len(branching_factors) > 1:
            self.branching_factors = branching_factors
        else:
            self.branching_factors = branching_factors * self.depth

        self.n_scenarios = prod(self.branching_factors)

        count = 1
        last_added_nodes = [self.starting_node]
        n_nodes_per_level = 1
        for i in range(self.depth):
            next_level = []
            self.filtration.append([])
            n_nodes_per_level *= self.branching_factors[i]
            
            demand_reduced, probs_reduced = stoch_model.reduce(
                self.branching_factors[i]
            )
            for parent_node in last_added_nodes:
                for j in range(self.branching_factors[i]):
                    id_new_node = count
                    self.add_node(
                        id_new_node,
                        obs=demand_reduced[:,j],
                        prob=self.nodes[parent_node]['prob'] * probs_reduced[j],
                        # obs=self._generate_new_scenario(parent_node),
                        # prob=1/(n_nodes_per_level),
                        t=i,
                        id=count,
                        stage=i
                    )
                    self.add_edge(parent_node, id_new_node)
                    next_level.append(id_new_node)
                    count += 1
            last_added_nodes = next_level
            self.n_nodes = count
        self.leaves = last_added_nodes

    def _generate_new_scenario(self, parent_node):
        history = self.get_history_node(parent_node)
        ris = self.stoch_model.generate_scenario(history=history)
        return ris

    def get_filtration_time(self, t):
        return self.filtration[t]

    def get_leaves(self):
        return self.leaves

    def get_history_node(self, n):  
        ris = np.array([self.nodes[n]['obs']]).T
        # print(n, "--", ris)
        if n == 0:
            return ris 
        while n != self.starting_node:
            n = list(self.predecessors(n))[0]
            ris = np.hstack((np.array([self.nodes[n]['obs']]).T, ris))
        return ris

    def set_scenario_chain(self, simulation_data):
        """
        Set the scenario simulation_data in all the nodes.
        Use it only in the perfect information case
        """
        for t in range(len(self.branching_factors)):
            self.nodes[t]['obs'] = simulation_data[:, t]

    def print_matrix_form_on_file(self, name_details=""):
        f = open(f"./results/tree_matrix_form_{name_details}.csv", "w")
        f.write("leaf, item")
        for ele in range(self.depth + 1):
            f.write(f",t{ele}")
        f.write("\n")
        for leaf in self.leaves:
            for obs in range(self.dim_observations):
                y = self.get_history_node(leaf)
                str_values = ",".join([f"{ele}" for ele in y[obs,:]])
                f.write(f"{leaf},{obs},{str_values}\n")
        f.close()

    def plot(self, filename=None):
        """It prints on the file path "./results/filename" the graph
        """
        pos = graphviz_layout(self, prog="dot")
        nx.draw(
            self, pos,
            with_labels=True, arrows=True
        )
        if filename:
            plt.savefig(f'./results/filename')
            plt.close()
        else:
            plt.show()
    
    def plot_all_scenarios_png(self):
        for leaf in self.leaves:    
            y = self.get_history_node(leaf)
            for obs in range(self.dim_observations):
                plt.plot(y[obs, :], label=f'obs {obs}')
            plt.legend()
            plt.ylabel(f'History scenario {leaf}')
            plt.savefig(f'./results/scenario_{leaf}.png')
            plt.close()
