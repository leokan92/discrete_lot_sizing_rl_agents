# -*- coding: utf-8 -*-
import time
import numpy as np
import gurobipy as grb
from envs import *


class PerfectInfoOptimization():
    def __init__(self, instance: SimplePlant):
        self.name = "perfectInfoOptimization"
        self.instance = instance
        # Sets
        self.items = range(instance.n_items)
        self.machines = range(instance.n_machines)
        self.time_steps = range(instance.T)
        # Model
        self.model = grb.Model(self.name)

        # 1 if machine m is able to produce item i at time t
        X = self.model.addVars(
            instance.n_items, instance.n_machines, instance.T,
            vtype=grb.GRB.BINARY,
            name='X'
        )

        # Setup 1 if a pay the setup related to item i in machine m
        D = self.model.addVars(
            instance.n_items, instance.n_machines, instance.T,
            vtype=grb.GRB.BINARY,
            name='D'
        )

        # Inventory
        I = self.model.addVars(
            instance.n_items, instance.T + 1,
            vtype=grb.GRB.CONTINUOUS,
            lb=0.0,
            name='I'
        )
        
        # Lost Sales
        Z = self.model.addVars(
            instance.n_items, instance.T + 1,
            vtype=grb.GRB.CONTINUOUS,
            lb=0.0,
            name='Z'
        )
    
        obj_func = grb.quicksum(
            (instance.lost_sales_costs[i] * Z[i, t] + instance.holding_costs[i] * I[i, t])
            for i in self.items
            for t in range(1, instance.T + 1)
        )
        obj_func += grb.quicksum(
            instance.setup_costs[m][i] * D[i, m, t]
            for i in self.items
            for m in self.machines
            for t in self.time_steps
        )

        self.model.setObjective(obj_func, grb.GRB.MINIMIZE)

        self.model.addConstrs(
            (I[i, t] <= instance.max_inventory_level[i] for i in self.items for t in self.time_steps),
            name='max_inventory'
        )

        # INITIAL STATE
        for m in self.machines:
            for i in self.items:
                # if m in state i, then:
                if i == instance.machine_initial_setup[m] - 1:
                    self.model.addConstr(
                        (D[i, m, 0] >= X[i, m, 0] - 1), 
                        name=f'initial_state_machine_{m}'
                    )
                    self.model.addConstr(
                        1 - X[i, m, 0] + 0.01 <= (1 + 0.01)*(1 - D[i, m, 0])
                    )
                else:
                    # if m not in state i, then, there is a change
                    self.model.addConstr(
                        (D[i, m, 0] >= X[i, m, 0]), # instance.machine_initial_setup[m] - 1
                        name=f'initial_state_machine_{m}'
                    )
                    self.model.addConstr(
                        0 - X[i, m, 0] + 0.01 <= (1 + 0.01)*(1 - D[i, m, 0])
                    )

        # EVOLUTION
        self.model.addConstrs(
            ( I[i, 0] == instance.initial_inventory[i] for i in self.items),
            name=f'initial_condition'
        )
        for t in range(1, self.instance.T + 1):
            # print(self.instance.scenario_demand[:, t])
            self.model.addConstrs(
                (I[i, t] - Z[i, t] == I[i, t-1] + grb.quicksum( instance.machine_production_matrix[m][i] * X[i, m, t-1] - instance.setup_loss[m][i] * D[i, m, t-1] for m in self.machines ) - self.instance.scenario_demand[i, t-1] for i in self.items),
                name=f'item_flow_{t}'
            )
  
        # Machine no multiple state    
        self.model.addConstrs(
            (grb.quicksum(X[i, m, t] for i in self.items) <= 1 for m in self.machines for t in self.time_steps ),
            name=f"no_more_setting_machine_node"
        )

        # avoid change to items with no production
        for i in self.items:
            for m in self.machines:
                if instance.machine_production_matrix[m][i] == 0:
                    self.model.addConstrs(
                        (D[i, m, t] == 0  for t in self.time_steps),
                        name=f'no_change_in_forbidden_state'
                    )

        # LINK X D
        for t in range(1, self.instance.T):
            self.model.addConstrs(
                D[i, m, t] >=  X[i, m, t] - X[i, m, t-1]
                for i in self.items for m in self.machines
            )
            self.model.addConstrs(
                X[i, m, t-1] - X[i, m, t] + 0.01 <= (1 + 0.01)*(1 - D[i, m, t])
                for i in self.items for m in self.machines
            )
        self.model.update()
        self.X = X
        self.D = D
        self.I = I
        self.Z = Z
        self.obj_func = obj_func

    def solve(
        self, time_limit=None,
        gap=None, verbose=False, debug_model=False
    ):

        if gap:
            self.model.setParam('MIPgap', gap)
        if time_limit:
            self.model.setParam(grb.GRB.Param.TimeLimit, time_limit)
        if verbose:
            self.model.setParam('OutputFlag', 1)
        else:
            self.model.setParam('OutputFlag', 0)
        
        self.model.setParam('MIPgap', 0.05)
        # self.model.setParam('OutputFlag', 1)
        self.model.setParam('LogFile', './logs/gurobi.log')
        if debug_model:
            self.model.write(f"./logs/{self.name}.lp")
        # self.model.write(f"./logs/perfectInfo.lp")

        start = time.time()
        self.model.optimize()
        end = time.time()
        comp_time = end - start

        if self.model.status == grb.GRB.Status.OPTIMAL:
            sol = np.zeros((self.instance.n_machines, self.instance.T))
            for t in self.time_steps:
                for m in self.machines:
                    sol[m,t] = sum([ (i+1) * self.X[i, m, t].X for i in self.items])

            # for t in range(0, self.instance.T+1):
            #     # print(f"time: {t}")
            #     str_to_print = f"{t}] "
            #     for i in self.items:
            #         str_to_print += f"inv_{i}: {abs(self.I[i, t].X):.0f} \t"
            #     str_to_print += f"[ls: {grb.quicksum(self.instance.lost_sales_costs[i] * self.Z[i, t].X for i in self.items).getValue():.2f}]"
            #     str_to_print += f"[hc: {grb.quicksum(self.instance.holding_costs[i] * self.I[i, t].X for i in self.items).getValue():.2f}]"
            #     print("\t ", str_to_print)

            # for t in range(self.instance.T):
            #     a = grb.quicksum(
            #         self.instance.setup_costs[m][i] * self.D[i, m, t]
            #         for i in self.items
            #         for m in self.machines
            #     ).getValue()
            #     print(f"{t}] setup: {a}")
            # for t in range(self.instance.T):
            #     for m in self.machines:
            #         for i in self.items:
            #             # if self.X[i, m, t].X > 0.5:
            #             #     print(f"X[{i}, {m}, {t}]")
            #             if self.D[i, m, t].X > 0.5:
            #                 if t >= 1:
            #                     print(f"D[{i}, {m}, {t}] {self.D[i, m, t].X} >= {self.X[i, m, t].X} - {self.X[i, m, t-1].X} ")
            #                 else:
            #                     print(f"D[{i}, {m}, {t}] >= {self.X[i, m, t].X} ")
            
            # print(">>> OF [PI model]: ", self.model.getObjective().getValue()) 
            # the of is different due to the first time step, but for the initial condition this is fixed
            return self.model.getObjective().getValue(), sol, comp_time
        else:
            print("MODEL INFEASIBLE OR UNBOUNDED")
            return -1, [], comp_time
