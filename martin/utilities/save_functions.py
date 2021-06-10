import numpy as np
import pathlib as pl
from utilities.constants import *

def save_history_array(args, history_indices, history, history_name='loss'):
    file_name = history_name + '_' + TIME_STAMP + '.txt'
    
    with open(str(pl.Path(args.out_path) / file_name) , 'w')\
            as file:
        file.write(f'batch,{history_name}\n')
        for i in range(len(history)):
            file.write(f'{history_indices[i]},{history[i]}\n')

def save_conf_matrix(args, matrix):
    file_name = 'conf_matrix' + '_' + TIME_STAMP + '.txt'
    path = str(pl.Path(args.out_path) / file_name)
    
    with open(path, 'a+') as file:
        np.savetxt(file, matrix, fmt='%1.2e', delimiter=',')
