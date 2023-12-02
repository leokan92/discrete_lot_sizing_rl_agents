import os
import numpy as np
import pandas as pd


def table_plotting(model_names = ['PPO'],experiment_names = [''], execution_type = '', PI_proportion = True):
    if PI_proportion:
        df = pd.DataFrame(columns = ['Experiment','Model','All Costs (PI%)','Holding Costs (PI%)','Lost Sales (PI%)','Setup Costs (PI%)'])  
    else:
        df = pd.DataFrame(columns = ['Experiment','Model','All Costs','Holding Costs','Lost Sales','Setup Costs'])
    path = os.path.dirname(os.path.abspath('__file__'))  
    for experiment in experiment_names:
        for model in model_names:
            holding_costs_path = os.path.join(path,'results',model+'_'+experiment+'_'+'holding_costs'+'_'+execution_type+'.npy')
            lost_sales_path = os.path.join(path,'results',model+'_'+experiment+'_'+'lost_sales'+'_'+execution_type+'.npy')
            setup_costs_path = os.path.join(path,'results',model+'_'+experiment+'_'+'setup_costs'+'_'+execution_type+'.npy')
            
            all_holding_costs = np.load(holding_costs_path)
            all_lost_sales= np.load(lost_sales_path)
            all_setup_costs = np.load(setup_costs_path)
            all_costs = all_holding_costs+all_lost_sales+all_setup_costs
            if model == 'PI':
                pi_all_holding_costs = all_holding_costs
                pi_all_lost_sales_path = all_lost_sales
                pi_all_setup_costs_path = all_setup_costs
                pi_all_costs = all_costs
                
            if PI_proportion:
                try:
                    df_temp = pd.DataFrame(
                        {
                            'Experiment': experiment,
                            'Model':model,
                            'All Costs (PI%)':      str(int(round(np.mean(all_costs - pi_all_costs)*100/np.mean(pi_all_costs),0)))+'$\pm$'+str(int(round(np.std((all_costs - pi_all_costs)*100/np.mean(pi_all_costs)),0))),
                            'Holding Costs (PI%)':  str(int(round(np.mean(all_holding_costs - pi_all_holding_costs)*100/np.mean(pi_all_holding_costs),0)))+'$\pm$'+str(int(round(np.std((all_holding_costs - pi_all_holding_costs)*100/np.mean(pi_all_holding_costs)),0))),
                            'Lost Sales (PI%)':     str(int(round(np.mean(all_lost_sales - pi_all_lost_sales_path)*100/np.mean(pi_all_lost_sales_path),0)))+'$\pm$'+str(int(round(np.std((all_lost_sales - pi_all_lost_sales_path)*100/np.mean(pi_all_lost_sales_path)),0))),
                            'Setup Costs (PI%)':    str(int(round(np.mean(all_setup_costs - pi_all_setup_costs_path)*100/np.mean(pi_all_setup_costs_path),0)))+'$\pm$'+str(int(round(np.std((all_setup_costs - pi_all_setup_costs_path)*100/np.mean(pi_all_setup_costs_path)),0)))
                        },
                        index=[0]
                    )
                except: print('error')
            else:
                df_temp = pd.DataFrame(
                    {
                        'Experiment': experiment,
                        'Model':model,
                        'All Costs':      str(int(round(np.mean(np.sum(all_costs,0)),0)))+'$\pm$'+str(int(round(np.std(np.sum(all_costs,0)),0))),
                        'Holding Costs':  str(int(round(np.mean(np.sum(all_holding_costs,0)),0)))+'$\pm$'+str(int(round(np.std(np.sum(all_holding_costs,0)),0))),
                        'Lost Sales':     str(int(round(np.mean(np.sum(all_lost_sales,0)),0)))+'$\pm$'+str(int(round(np.std(np.sum(all_lost_sales,0)),0))),
                        'Setup Costs':    str(int(round(np.mean(np.sum(all_setup_costs,0)),0)))+'$\pm$'+str(int(round(np.std(np.sum(all_setup_costs,0)),0)))
                    },
                    index=[0]
                )
            df = df.append(df_temp)
    
    caption = 'Average cost and inventory level for each experimented model trained on 5 different seeds. One machine and two items scenario'
    print(
        df.to_latex(
            index=False,
            label='tab:coslidated_results',
            caption=caption,escape = False
        )
    )
