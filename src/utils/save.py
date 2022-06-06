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
    if 'model_discriminator' in state.keys():
        for k in list(state["model_generator"].keys()):
            if "ops" in k or "params" in k:
                state["model_generator"].pop(k)
        for k in list(state["model_discriminator"].keys()):
            if "ops" in k or "params" in k:
                state["model_discriminator"].pop(k)
    else:
        for k in list(state["model"].keys()):
            if "ops" in k or "params" in k:
                state["model"].pop(k)
    torch.save(state, filename)
    if is_best:
        if not os.path.exists(filepath1 + 'best/'):
            os.makedirs(filepath1 + 'best/')
        shutil.copyfile(filename, filepath1 + 'best/' + 'model_best.pth')