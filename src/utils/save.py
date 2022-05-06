import os
import torch
import shutil

def save_checkpoint(state, is_best, filepath1, filepath2, filename='checkpoint.pth'):
    r'''
    -exp
        -*
            -checkpoint
                -model0
                    checkpoint.pth
                -model1
                    checkpoint.pth
                ...
                -best
                    checkpoint.pth
    '''
    directory = filepath1 + filepath2
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        if not os.path.exists(filepath1 + 'best/'):
            os.makedirs(filepath1 + 'best/')
        shutil.copyfile(filename, filepath1 + 'best/' + 'model_best.pth')