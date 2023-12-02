# -*- coding: utf-8 -*-
import json 
import numpy as np
import warnings


def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

def generate_json_file(time_horizon,
                       n_items,
                       n_machines,
                       initial_setup,
                       machine_production,
                       max_inventory_level,
                       initial_inventory_level,
                       holding_costs,
                       lost_sales_costs,
                       demand_distribution,
                       setup_costs,
                       setup_loss,
                       file_name):
    file_dict = {                           
                           "time_horizon": time_horizon,
                           "n_items": n_items,
                           "n_machines": n_machines,
                           "initial_setup": initial_setup,
                           "machine_production": machine_production,
                           "max_inventory_level": max_inventory_level,
                           "initial_inventory": initial_inventory_level,
                           "holding_costs": holding_costs,
                           "lost_sales_costs": lost_sales_costs,
                           "demand_distribution": demand_distribution,
                           "setup_costs": setup_costs,
                           "setup_loss": setup_loss
                }
    # Serializing json  
    json_object = json.dumps(file_dict, indent = 4) 
    print(json_object)
    with open(f"{file_name}.json", "w") as outfile:
        outfile.write(json_object)
    



def generate_vector_min_bias(n_elements = 10,max_value = 10,min_value= 0 ,max_prob = 0.6):
    if max_value == min_value:
        return np.ones(n_elements,dtype = int)*max_value
    else:
        choices = np.arange(min_value,max_value+1)
        probs = np.zeros(len(choices))
        probs[0] = max_prob
        if len(choices)<2:
            leftprob = 1-max_prob
        else:
            leftprob = (1-max_prob)/(len(choices)-1)
        probs[1:] = leftprob
        return np.random.choice(choices,n_elements,p = probs.tolist())


def generate_matrix(n_lines,n_columns,max_value,min_value,max_prob):
    final_matrix = []
    for i in range(n_lines):
        final_matrix.append(generate_vector_min_bias(n_columns,max_value,min_value,max_prob))
    if n_lines <2:
        return final_matrix[0].tolist()
    else:
        final_matrix = [l.tolist() for l in final_matrix]
        return final_matrix

def check_all_zeros(matrix):
    for i in range(len(matrix)):
        if sum(np.array(matrix)[:,i]) == 0:
            return warnings.warn("The generated matrix contains columns with zeros only")

# time_horizon = 5
# n_items = 100
# n_machines = 80

# #initial_setup = generate_matrix(1,n_machines,n_items,0,0.4)
# initial_setup = generate_matrix(1,n_machines,n_items,0,0.4)
# machine_production = generate_matrix(n_machines,n_items,4,0,0.4)
# check_all_zeros(machine_production)
# max_inventory_level = generate_matrix(1,n_items,10,10,0.5)
# initial_inventory_level = generate_matrix(1,n_items,0,0,0.05)
# holding_costs = generate_matrix(1,n_items,0.1,0.1,0.5)
# lost_sales_costs = generate_matrix(1,n_items,3,1,0.5)
# demand_distribution = {
#         "name": "binomial",
#         "n": 4,
#         "p": 0.4
#     }
# setup_costs = generate_matrix(n_machines,n_items,2,1,0.2)
# setup_loss = generate_matrix(n_machines,n_items,1,1,0.7)
# file_name = f'setting_{n_items}items_{n_machines}machines_t{time_horizon}'

# generate_json_file( time_horizon,
#                     n_items,
#                     n_machines,
#                     initial_setup,
#                     machine_production,
#                     max_inventory_level,
#                     initial_inventory_level,
#                     holding_costs,
#                     lost_sales_costs,
#                     demand_distribution,
#                     setup_costs,
#                     setup_loss,
#                     file_name)

time_horizon = 100
n_items = 25
n_machines = 10

#initial_setup = generate_matrix(1,n_machines,n_items,0,0.4)
initial_setup = generate_matrix(1,n_machines,n_items,0,0.4)
machine_production = generate_matrix(n_machines,n_items,40,0,0.4)
check_all_zeros(machine_production)
max_inventory_level = generate_matrix(1,n_items,100,100,0.5)
initial_inventory_level = generate_matrix(1,n_items,0,0,0.05)
holding_costs = generate_matrix(1,n_items,0.01,0.01,0.5)
lost_sales_costs = generate_matrix(1,n_items,0.2,0.2,0.5)
demand_distribution = {
        "name": "binomial",
        "n": 20,
        "p": 0.4
    }
setup_costs = generate_matrix(n_machines,n_items,2,1,0.2)
setup_loss = generate_matrix(n_machines,n_items,1,1,0.7)
file_name = f'setting_{n_items}items_{n_machines}machines'
generate_json_file( time_horizon,
                    n_items,
                    n_machines,
                    initial_setup,
                    machine_production,
                    max_inventory_level,
                    initial_inventory_level,
                    holding_costs,
                    lost_sales_costs,
                    demand_distribution,
                    setup_costs,
                    setup_loss,
                    file_name)


# for m in [2,5,10]:
#     for n in [10,20,30,40,50,60,70,80,90,100]:
#         time_horizon = 5
#         n_items = n
#         n_machines = m
        
#         #initial_setup = generate_matrix(1,n_machines,n_items,0,0.4)
#         initial_setup = generate_matrix(1,n_machines,n_items,0,0.4)
#         machine_production = generate_matrix(n_machines,n_items,4,0,0.2)
#         check_all_zeros(machine_production)
#         max_inventory_level = generate_matrix(1,n_items,10,10,0.5)
#         initial_inventory_level = generate_matrix(1,n_items,0,0,0.05)
#         holding_costs = generate_matrix(1,n_items,0.1,0.1,0.5)
#         lost_sales_costs = generate_matrix(1,n_items,3,1,0.5)
#         demand_distribution = {
#                 "name": "binomial",
#                 "n": 4,
#                 "p": 0.4
#             }
#         setup_costs = generate_matrix(n_machines,n_items,2,1,0.2)
#         setup_loss = generate_matrix(n_machines,n_items,1,1,0.7)
#         file_name = f'setting_{n_items}items_{n_machines}machines_t{time_horizon}'
        
#         generate_json_file( time_horizon,
#                             n_items,
#                             n_machines,
#                             initial_setup,
#                             machine_production,
#                             max_inventory_level,
#                             initial_inventory_level,
#                             holding_costs,
#                             lost_sales_costs,
#                             demand_distribution,
#                             setup_costs,
#                             setup_loss,
#                             file_name)