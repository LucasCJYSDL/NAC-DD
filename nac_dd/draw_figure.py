import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def nf_plot(id):
    # data files
    current_directory = os.path.dirname(os.path.abspath(__file__))
    file_dir = {'NAC-DD-5': [0, 1, 2], 'PG': [0, 1, 2], 'NPG': [0, 1, 2], 'TRPO': [0, 1, 2], 'PPO': [0, 1, 2]}
    
    common_dir = current_directory + '/results/' + id

    if id == 'hp':
        threshold = 7000
    else:
        threshold = 15000

    data_frame = pd.DataFrame()
    for alg, dir_name_list in file_dir.items():
        j = 0
        for dir_name in dir_name_list:
            csv_file_name = str(dir_name) + '.csv'
            csv_file_dir = os.path.join(common_dir, alg, csv_file_name)
            print("Loading from: ", alg, csv_file_dir)

            temp_df = pd.read_csv(csv_file_dir)
            if j == 0:
                temp_step = np.array(temp_df['Step'])
            j += 1

            temp_value = np.array(temp_df['Value'])
            print("Average rate across the episodes: ", np.mean(temp_value))
            temp_len = len(temp_step)
            
            mov_avg_agent_2 = MovAvg(window_size=30)
            for i in range(temp_len):

                if i >= len(temp_value) or temp_step[i] >= threshold:
                    break
                data_frame = data_frame.append({'algorithm': alg, 'Number of Outer Loops': int(temp_step[i]),
                                                'Return': mov_avg_agent_2.update(temp_value[i])}, ignore_index=True)
    
    return data_frame


def draw_overall_nf():
    sns.set_theme(style="darkgrid")
    sns.set(font_scale=2.5)
    fig, axs = plt.subplots(1, 3, figsize=(36, 8))

    sns.lineplot(x="Number of Outer Loops", y="Return", hue='algorithm', data=nf_plot('hp'), ax=axs[0])
    sns.lineplot(x="Number of Outer Loops", y="Return", hue='algorithm', data=nf_plot('hc'), ax=axs[1])
    sns.lineplot(x="Number of Outer Loops", y="Return", hue='algorithm', data=nf_plot('wk'), ax=axs[2])

    axs[0].set_title("Hopper-v3")
    axs[1].set_title("HalfCheetah-v3")
    axs[2].set_title("Walker2d-v3")

    # Get handles and labels for shared legend
    handles, labels = axs[0].get_legend_handles_labels()
    # print(handles, labels)
    # Remove individual legends from each subplot
    for ax in axs:
        ax.legend().remove()

    # Add a shared legend below all subplots
    fig.legend(handles, labels, ncol=len(labels), loc='lower center', bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)

    current_directory = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(current_directory + '/results/return.png')

class MovAvg(object):

    def __init__(self, window_size=20, is_max=False): 
        self.window_size = window_size
        self.data_queue = []
        self.is_max = is_max

    def set_window_size(self, num):
        self.window_size = num

    def clear_queue(self):
        self.data_queue = []

    def update(self, data):
        if len(self.data_queue) == self.window_size:
            del self.data_queue[0]
        self.data_queue.append(data)
        if not self.is_max:
            return sum(self.data_queue) / len(self.data_queue)
        return max(self.data_queue)
    


if __name__ == '__main__':
    draw_overall_nf()


