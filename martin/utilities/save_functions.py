import pathlib as pl
from utilities.constants import *

def save_history_array(args, history_indices, history, history_name='loss'):
    file_name = history_name + '_' + TIME_STAMP + '.txt'
    
    with open(str(pl.Path(args.out_path) / file_name) , 'w')\
            as file:
        file.write(f'index,{history_name}\n')
        for i in range(len(history)):
            file.write(f'{history_indices[i]},{history[i]}\n')
