import os
import re
import pandas as pd 
from IPython.display import clear_output
from matplotlib import pyplot as plt
import numpy as np
import copy

def aver_list(l, n): 
    r = len(l)%n
    c = len(l)-r
    av = np.array(l[:len(l)-r])
    if c !=0:
        av = np.average(np.array(l[:len(l)-r]).reshape(-1, n), axis=1)
    if r != 0: 
        av = np.append(av, sum(l[len(l)-r:])/r)
    return av

def extract_number(filename):
    match = re.search(r'(\d+)$', filename)
    return int(match.group(1)) if match else -1
    
def get_the_last_checkpoint(folder_path):
    files = os.listdir(folder_path)

    # Get the file with the highest number
    if files:
        sorted_files = sorted(files, key=extract_number)
        last_file = sorted_files[-1]
        return folder_path + "/" + last_file
    else:
        return None
        
def get_new_df_interval(name, n, interval): 
    df = pd.read_csv(name) 
    selected_rows = df[df['nsites'] == n].iloc[interval]

    # Creating a new DataFrame
    new_df = pd.DataFrame(selected_rows)
    return new_df


def create_plots(data_list, save = False, show = True, path_to_the_main_dir = None, env_name = None, name = None):  
    
        font = {'family': 'serif',
#         'color':  'Black',
        'weight': 'normal',
        'size': 14,
        }
        numb = int(len(data_list)//3) + 1*(int(len(data_list)%3)!=0)
        if show: 
            clear_output(wait=True)

        fig, axes = plt.subplots(numb, 3)
        plt.rc('font', **font)
        fig.set_figwidth(20)    #  ширина и
        fig.set_figheight(8)
        axes = axes.flatten()
        for item_ax in zip(data_list.keys(), axes):
            item = item_ax[0]
            ax = item_ax[1]
            max_data = []
            min_data = []
            for label, data in zip(data_list[item][0],data_list[item][1]):
                if data_list[item][3] is not None: 
                    fmt = data_list[item][3]
                else:
                    fmt = "-"
                ax.plot(data, fmt, label = label)
                ax.set_title(item, fontdict=font)
                
            if len(data)!= 0 and data_list[item][4] is not None: 
                ax.plot(data_list[item][4], min(data)*np.ones(len(data_list[item][4])),'|r')
                
            if type(data_list[item][2]) is np.ndarray or type(data_list[item][2]) is list:
                ylim = copy.deepcopy(data_list[item][2])
                if len(data)!= 0 and None not in data:
                    max_data.append(max(data))
                    min_data.append(min(data))
                    ylim[1] = min(max(max_data), ylim[1])
                    ylim[0] = max(min(min_data), ylim[0])
                    if ylim[0] == ylim[1]:
                        ylim = None
            else: 
                    ylim = None
            ax.set_ylim(ylim)         
            ax.set_xlabel("Number of steps", fontdict=font)
            ax.legend()
        plt.tight_layout()
        
        if show: 
            plt.show()
        if save: 
            if not os.path.exists(path_to_the_main_dir + "/" + 'plots/'):
                os.makedirs(path_to_the_main_dir + "/" +'plots/')
    
            path_to_save = "{0}".format(env_name)
            fig.savefig(path_to_the_main_dir + "/" +"plots/" + path_to_save + name)
            plt.close(fig)
            
