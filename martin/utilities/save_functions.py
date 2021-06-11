import numpy as np
import pathlib as pl
from utilities.constants import *

def save_history_array(args, history_indices, history, history_name='loss', data_set='val'):
    file_name = history_name + '_' + TIME_STAMP + '.txt'
    
    with open(str(pl.Path(args["out_path"]) / file_name) , 'w')\
            as file:
        file.write(f'batch,{history_name}\n')
        for i in range(len(history)):
            file.write(f'{history_indices[i]},{history[i]}\n')

def save_conf_matrix(args, matrix, data_set='val'):
    file_name = data_set +'_' + 'conf_matrix' + '_' + TIME_STAMP + '.txt'
    path = str(pl.Path(args["out_path"]) / file_name)
    
    with open(path, 'a+') as file:
        np.savetxt(file, matrix, fmt='%1.4e', delimiter=',')

def save_probs_and_truth(args, probs, truth, data_set='val'):

    file_name_prob = data_set +'_' + 'prob' + '_' + TIME_STAMP + '.txt'
    path_prob = str(pl.Path(args["out_path"]) / file_name_prob)

    file_name_truth = data_set +'_' + 'truth' + '_' + TIME_STAMP + '.txt'
    path_truth = str(pl.Path(args["out_path"]) / file_name_truth)
    
    with open(path_prob, 'w') as file:
        np.savetxt(file, probs, fmt='%1.4e', delimiter=',')
    
    with open(path_truth, 'w') as file:
        np.savetxt(file, truth, fmt='%d', delimiter=',')