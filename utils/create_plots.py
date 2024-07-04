from IPython.display import clear_output
from matplotlib import pyplot as plt
import numpy as np
import copy


def create_plots(i, num_episodes, data_list):  
    
        font = {'family': 'serif',
#         'color':  'Black',
        'weight': 'normal',
        'size': 14,
        }

        numb = int(len(data_list)//3) + 1*(int(len(data_list)%3)!=0)
        clear_output(wait=True)

        fig, axes = plt.subplots(numb, 3)
        plt.rc('font', **font)
        fig.set_figwidth(20)    #  ширина и
        fig.set_figheight(8)
#         fig.suptitle(f"Episode {1+i}/{num_episodes}", fontsize=16)
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
        plt.show()
