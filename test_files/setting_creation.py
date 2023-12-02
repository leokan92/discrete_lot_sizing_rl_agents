import json
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# TODO: check with generate_setting.py

time_horizon = 7
n_items = 50
n_machines = 10
MAX_INVENTORY = 50

dict_template = {
    "time_horizon": time_horizon,
    "n_items": n_items,
    "n_machines": n_machines,
    "initial_setup": [],
    "machine_production": [],
    "max_inventory_level": [],
    "initial_inventory": [],
    "holding_costs": [],
    "lost_sales_costs": [],
    "demand_distribution": {
        "name": "item_specific_uniform",
        "distributions": []
    },
    "setup_costs": [],
    "setup_loss": []
}



# GENERAL PARAMETERS
dict_template['time_horizon'] = time_horizon
dict_template['n_items'] = n_items
dict_template['n_machines'] = n_machines

MIN_PROD = 10
MAX_PROD = 20

machine_production_binary_mask = np.ones(
    shape=(n_machines, n_items)
)
for i in range(n_items):
    n_rnd_machines = np.random.randint(low=1, high= n_machines*0.5)
    rnd_machines = np.random.choice(
        range(n_machines),
        size=n_rnd_machines,
        replace=False
    )
    for m in rnd_machines:
        machine_production_binary_mask[m, i] = 0

dict_template["machine_production"] = np.random.randint(
    low=MIN_PROD,
    high=MAX_PROD,
    size=(n_machines, n_items)
) * machine_production_binary_mask

possible_machine_production = []
for i in range(n_machines):
    possible_machine_production.append([0])
    possible_machine_production[-1].extend(
        list(
            np.nonzero(dict_template["machine_production"][i])[0] + 1
        )
    )

# INITIAL SETUP
dict_template["initial_setup"] = [0] * n_machines
for m in range(n_machines):
    dict_template["initial_setup"][m] = 0 #np.random.choice(possible_machine_production[m])


# COSTS
dict_template["holding_costs"] = np.round(np.random.uniform(
    low=0,
    high=1,
    size=n_items
), 1).tolist()
dict_template["holding_costs"] = dict_template["holding_costs"]
dict_template["lost_sales_costs"] = np.random.randint(
    low=5,
    high=10,
    size=n_items
).tolist()
dict_template['setup_costs'] = np.random.randint(
    low=1,
    high=5,
    size=(n_machines, n_items)
) * machine_production_binary_mask
dict_template['setup_costs'] = dict_template['setup_costs'].tolist()

# DEMAND:
dict_template['demand_distribution']['name'] = 'item_specific_uniform'
for i in range(n_items):
    avg_prod = np.mean(
        [dict_template["machine_production"][m][i] for m in range(n_machines) if dict_template["machine_production"][m][i] > 0]
    )
    ratio_items_machines = n_items / n_machines
    dict_template['demand_distribution']['distributions'].append(
        {
            "name": "discrete_uniform",
            "low": 0,
            "high": np.random.randint(
                low=max(int(avg_prod / ratio_items_machines), 1),
                high=max(int(avg_prod / (0.5 * ratio_items_machines) ), 2)
            )
        }
    )

# INVENTORY:
dict_template["max_inventory_level"] = [MAX_INVENTORY] * n_items

dict_template["initial_inventory"] = [0] * n_items
for i in range(n_items):
    avg_demand = (dict_template['demand_distribution']['distributions'][i]['high']+dict_template['demand_distribution']['distributions'][i]['low']) / 2.0
    dict_template['initial_inventory'][i] = avg_demand * np.random.randint(0, 3)

# SETUP LOSS
dict_template['setup_loss'] = np.floor(dict_template["machine_production"] * np.random.uniform(
    low=0.1, high=0.2,
    size=(n_machines, n_items)
)).astype(int) * machine_production_binary_mask


# CONVERSIONS TO JSON
dict_template["machine_production"] = dict_template["machine_production"].tolist()
dict_template['setup_loss'] = dict_template['setup_loss'].tolist()
with open(f"./cfg_env/I_I{n_items}xM{n_machines}xT{time_horizon}.json", "w") as outfile:
    outfile.write(
        json.dumps(dict_template, indent = 4, cls=NpEncoder) 
    )

from envs import *
from agents import *
from scenarioManager.stochasticDemandModel import StochasticDemandModel
stoch_model = StochasticDemandModel(dict_template)

env = SimplePlant(dict_template, stoch_model)
env.plot_production_matrix()

fp = open(f"./cfg_sol/sol_setting.json", 'r')
setting_sol_method = json.load(fp)
fp.close()

agent = PerfectInfoAgent(env, setting_sol_method)

obs = env.reset_time()
done = False
while not done:
    action = agent.get_action(obs)
    obs, _, done, info = env.step(action, verbose=False)
    print(f"{env.current_step}] {action}")
