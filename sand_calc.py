import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def rely():
    # path = './csv_files/class_2d_epochs_20energies/run_0/epoch_50'
    path = './csv_files/epoch_30'
    df = pd.read_csv(os.path.join(path, 'data_frame.csv'))
    rel_error = df.rel_error

    print(f'The rel error per event mean is: {rel_error.mean():.2f}, rel error per event std is: {rel_error.std():.2f}')
    output = df.output
    target = df.target
    print(f'Average number of particles per event:\n'
    f'output: {output.mean():.2f}, target: {target.mean():.2f}, rel error: {(output.mean()-target.mean())/target.mean()}')

def my_rel():
    my_path = os.path.join('.', 'csv_files', '1class_negativeRel')
    for i in np.linspace(10, 100, 10, dtype='int'):
        df = pd.read_csv(os.path.join(my_path, f'epoch_{i}', 'data_frame.csv'))
        plt.figure(figsize=(12, 6))
        plt.clf()
        plt.ylabel('relative error in %')
        plt.xlabel('target value')
        y = df.rel_error
        y *= 100
        x = df.target
        x = x[y < 2]
        y = y[y < 2]
        plt.scatter(x, y)
        plt.savefig(os.path.join(my_path, f'epoch_{i}', 'rel_error_fig.png'))

if __name__ == '__main__':
    # rely()
    my_rel()
