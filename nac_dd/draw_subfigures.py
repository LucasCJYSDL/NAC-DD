import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot():
    sns.set_theme(style="darkgrid")
    
    # data files
    common_dir = './results/wk'
    file_dir = {'NAC-DD-1': [0, 1, 2], 'NAC-DD-3': [0, 1, 2], 'NAC-DD-5': [0, 1, 2]}
    # file_dir = {'NAC-DD-5': [0, 1, 2], 'PG': [0, 1, 2], 'NPG': [0, 1, 2], 'TRPO': [0, 1, 2], 'PPO': [0, 1, 2]}

    if 'hp' in common_dir:
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
            print("Average rwd across the episodes: ", np.mean(temp_value))
            temp_len = len(temp_step)
            
            # mov_avg_agent_1 = MovAvg(is_max=True, window_size=50)
            mov_avg_agent_2 = MovAvg(window_size=30)
            for i in range(temp_len):
                if temp_step[i] > threshold or i >= len(temp_value):
                    break
                data_frame = data_frame.append({'algorithm': alg, 'Number of Outer Loops': int(temp_step[i]),
                                                'Return': mov_avg_agent_2.update(mov_avg_agent_2.update(temp_value[i]))}, ignore_index=True)

    print(data_frame)
    sns.set(font_scale=1.8)
    # pal = sns.xkcd_palette((['red', 'blue', 'green', 'yellow', 'orange', 'brown']))
    pal = sns.color_palette("tab10")
    g = sns.relplot(x="Number of Outer Loops", y="Return", hue='algorithm', kind="line",
                    data=data_frame, legend='brief')
    # g.ax.set_ylim(3500, 6500)
    leg = g._legend
    leg.set_bbox_to_anchor([0.65, 0.43])  # coordinates of lower left of bounding box [0.75, 0.4], [0.75, 0.6], [0.55, 0.65]
    # g = sns.relplot(x="training step", y="mean reward", hue='algorithm', kind="line", data=data_frame)
    g.fig.set_size_inches(20, 6)
    plt.savefig(common_dir + '-return.png')


class MovAvg(object):

    def __init__(self, window_size=20, is_max=False): # 20, 20, 50
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
    plot()


