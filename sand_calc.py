import pandas as pd
import os

# path = './csv_files/class_2d_epochs_20energies/run_0/epoch_50'
path = './csv_files/epoch_30'
df = pd.read_csv(os.path.join(path, 'data_frame.csv'))
rel_error = df.rel_error

print(f'The rel error per event mean is: {rel_error.mean():.2f}, rel error per event std is: {rel_error.std():.2f}')
output = df.output
target = df.target
print(f'Average number of particles per event:\n'
f'output: {output.mean():.2f}, target: {target.mean():.2f}, rel error: {(output.mean()-target.mean())/target.mean()}')
