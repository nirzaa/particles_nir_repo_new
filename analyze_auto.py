import os
import h5py
import numpy as np
import matplotlib.pyplot as plt


output_list = list()
target_list = list()
with open(os.path.join('./csv_files', 'stats.txt'), 'w') as f:
    f.write('Stats for our data\n')
    f.write('='*40)
    f.write('\n\n')
for run_num in range(1):
    my_path = f'./csv_files/1_class_2d/run_{run_num}'
    for epoch_num in [10, 20, 30, 40, 50, 60]:
        with h5py.File(os.path.join(my_path, f'epoch_{epoch_num}', 'data.h5'), 'r') as hf:
            output = np.array(hf.get('dataset_1'))
            target = np.array(hf.get('dataset_2'))
            rel_error = abs(output.sum()-target.sum()) / target.sum()
            with open(os.path.join('./csv_files', 'stats.txt'), 'a+') as f:
                f.write(f'The average results for {epoch_num} epoch\n')
                f.write('='*50)
                f.write(f'\nThe output average number of particles per event is: {output.mean():.2f}')
                f.write(f'\nThe target average number of particles per event is: {target.mean():.2f}')
                f.write(f'\nthe output N value is: {output.sum()}'
                f'\nthe target N value is: {target.sum()}')
                f.write(f'\nrelative error for total N: {rel_error*100:.2f}%\n\n')
        output_list.append(output)
        target_list.append(target)
my_output = np.stack(output_list, axis=0)
my_target = np.stack(target_list, axis=0)
mean_output = my_output.mean(axis=0)
mean_target = my_target.mean(axis=0)
bars = np.linspace(0, 13, mean_output.shape[1])
bars = [float(f'{i:.2f}') for i in bars]
text = 'Bin Energy range [GeV]: \n'
for i in range(mean_output.shape[1] - 1):
    text += f'{i}: {bars[i]:.1f} - {bars[i + 1]:.1f} \n'
rng = [i + 1 for i in range(mean_output.shape[1])]
plt.figure(figsize=(12, 6))
plt.bar(rng, mean_output.sum(axis=0), label='output', alpha=0.5)
plt.errorbar(rng, mean_output.sum(axis=0), yerr=(1 / np.sqrt(np.abs(mean_output.sum(axis=0)))), fmt="+", color="b")
plt.bar(rng, mean_target.sum(axis=0), label='true_val', alpha=0.3)
plt.xlabel('bins number for energies')
plt.ylabel('number of particles')
# plt.text(15.5, 0.015, text,
plt.text(15.5, 0.015, text,
            bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 3}, fontsize='x-small')

plt.xticks(rng, rotation=65)
plt.title(f'{len(output)} samples')
plt.legend()
plt.savefig(f'./csv_files/binsgraph.png')



rel_error_N = abs(mean_output.sum()-mean_target.sum()) / mean_target.sum()
t = mean_target.sum(axis=1)
o = mean_output.sum(axis=1)
with open(os.path.join('./csv_files', 'stats.txt'), 'a+') as f:
    f.write(f'The average results for all the above epochs\n')
    f.write('='*50)
    f.write(f'\nThe output average number of particles per event is: {o.mean():.2f}')
    f.write(f'\nThe target average number of particles per event is: {t.mean():.2f}')
    f.write(f'\nthe output N value is: {mean_output.sum()}'
    f'\nthe target N value is: {mean_target.sum()}')
    f.write(f'\nrelative error for total N: {rel_error_N*100:.2f}%\n')

    
