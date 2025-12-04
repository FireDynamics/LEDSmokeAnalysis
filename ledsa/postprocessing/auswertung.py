import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import seaborn as sns
import numpy as np
import pandas as pd
from ledsa.postprocessing.simulation import SimData
import os
from matplotlib.ticker import FormatStrFormatter

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10
cm = 1 / 2.54

# Input Dir (Simulations) !alle channel müssen abgegriffen werden
path_simulation = os.getcwd()
sim = SimData(path_simulation)
output_path = os.path.join(path_simulation,'output')

if not os.path.isdir(output_path):
    os.makedirs (output_path)


#read possible times
df_info = pd.read_csv('image_infos.csv', header = 0)
times = df_info['Experiment_Time[s]']

# Parameters
led_array_id = 0  # LED array to analyze
window = 1  # Size of moving average window
color_channels = [0, 1, 2] # RGB channels

# Create figure Extinction
fig, ax = plt.subplots(figsize=(10, 6))

# Plot extinction coefficients for each channel
for channel in color_channels:
    # Get extinction coefficients at specified height
    extco = sim.get_extco_at_led_array(channel=channel, led_array_id=led_array_id, window=window, yaxis='layer')
    middleIndex = (len(extco.columns) - 1)//2
    # Plot with different colors for each channel
    colors = ['red', 'green', 'blue']
    ax.plot(extco.index, extco.iloc[:,middleIndex],
            color=colors[channel],
            label=f'Channel {channel}')
    # ax.fill_between(extco.index, 10,20, color=colors[channel], alpha = 0.5)
    ax.fill_between(extco.index, extco.iloc[:,:].T.max(), extco.iloc[:,:].T.min(), color=colors[channel], alpha = 0.1)

ax.set_xlabel('Time / s')
ax.set_ylabel('Extinction Coefficient / $\mathrm{m^{-1}}$')
ax.set_title(f'Extinction Coefficients at LED-ID {middleIndex}, LED Array {led_array_id}')
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_path,'extco_time_range.png'))
plt.close()

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot extinction coefficients for each channel
for channel in color_channels:
    # Get extinction coefficients at specified height
    extco = sim.get_ledparams_at_led_array(channel=channel, led_array_id=led_array_id, window=window, yaxis='height')
    middleIndex = (len(extco.columns) - 1)//2
    # Plot with different colors for each channel
    colors = ['red', 'green', 'blue']
    ax.plot(extco.index, extco.iloc[:,middleIndex],
            color=colors[channel],
            label=f'Channel {channel}')
    ax.fill_between(extco.index, extco.iloc[:,:].T.max(), extco.iloc[:,:].T.min(), color=colors[channel], alpha = 0.2)

ax.set_xlabel('Time / s')
ax.set_ylabel('LED Parameters')
ax.set_title(f'LED Parameters at LED Parameter {middleIndex}, LED Array {led_array_id}')
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_path,'led_time_range.png'))
plt.close()

# Parameters
for channel in color_channels:
    fig, ax = plt.subplots(figsize=(10, 6))
    extco = sim.get_extco_at_led_array(channel = channel, led_array_id =led_array_id, yaxis='height', window=window).T
    extco = extco.set_index(np.round(extco.index, decimals = 2))
    sns.heatmap(extco, cmap='jet',  vmax=extco.max().max(), vmin=0, cbar_kws={'label': 'Extinction Coefficient / $\mathrm{m^{-1}}$'})#,yticklabels = np.round(extco.index, decimals=2))
    plt.tight_layout()
    plt.title(f'channel {channel}')
    plt.savefig(os.path.join(output_path,f'sns_extco_time_{channel}.png'))
    plt.close()

for channel in color_channels:
    fig, ax = plt.subplots(figsize=(10, 6))
    extco = sim.get_ledparams_at_led_array(channel=channel, led_array_id=0, window=window, yaxis='height').T #, n_ref=10
    extco = extco.set_index(np.round(extco.index, decimals = 2))
    sns.heatmap(extco[::-1], cmap='jet',  vmax=1.0,vmin= 0.1, cbar_kws={'label': 'LED-Parameter'})#,yticklabels = np.round(extco.index, decimals=2))
    plt.tight_layout()
    plt.title(f'channel{channel}')
    plt.savefig(os.path.join(output_path,f'sns_led_time_{channel}.png'))
    plt.close()