import copy

import clip
import numpy as np
import torch

from embedding import get_delta_t
from manipulator import Manipulator
from wrapper import Generator


def get_delta_s(global_style_direction, delta_t, manipulator, beta_threshold):

    delta_s = np.dot(global_style_direction, delta_t)

    select = np.abs(delta_s) < beta_threshold # apply beta threshold (disentangle)
    num_channel = np.sum(~select)

    delta_s[select] = 0 # set the style direction below the threshold to 0
    absmax = np.abs(copy.deepcopy(delta_s)).max()
    delta_s /= absmax # normalize

    # delta_s -> style dict
    dic = dict()
    ind = 0
    for layer in manipulator.G.style_layers: # 26
        dim = manipulator.styles[layer].shape[-1]
        if layer in manipulator.manipulate_layers:
            dic[layer] = torch.from_numpy(delta_s[ind:ind+dim]).to(manipulator.device)
            ind += dim
        else:
            dic[layer] = torch.zeros([dim]).to(manipulator.device)
    return dic, num_channel

def get_delta_s_top(global_style_direction, delta_t, manipulator, num_top):
    delta_s = np.dot(global_style_direction, delta_t)

    select = np.argpartition(delta_s, -num_top)[-num_top:]
    mask = np.ones(delta_s.shape,dtype=bool) #np.ones_like(a,dtype=bool)
    mask[select] = False
    num_channel = np.sum(mask[~select])

    delta_s[mask] = 0 
    absmax = np.abs(copy.deepcopy(delta_s)).max()
    delta_s /= absmax # normalize

    # delta_s -> style dict
    dic = dict()
    ind = 0
    for layer in manipulator.G.style_layers: # 26
        dim = manipulator.styles[layer].shape[-1]
        if layer in manipulator.manipulate_layers:
            dic[layer] = torch.from_numpy(delta_s[ind:ind+dim]).to(manipulator.device)
            ind += dim
        else:
            dic[layer] = torch.zeros([dim]).to(manipulator.device)
    return dic, num_channel



if __name__ == "__main__":
    #manipulator = Manipulator
    device = torch.device('cuda:3')
    ckpt = 'pretrained/ffhq.pkl'
    G = Generator(ckpt, device)
    model, preprocess = clip.load("ViT-B/32", device=device)

    fs3 = np.load('./tensor/fs3.npy')

    # text space
    classnames=['face', 'face with glasses']
    delta_t = get_delta_t(classnames, model)

    lst_alpha = [0]
    num_images = 1
    manipulator = Manipulator(G, device, lst_alpha, num_images)

    # get style direction
    beta_threshold = 0.13
    delta_s, num_channel = get_delta_s(fs3, delta_t, manipulator, beta_threshold=beta_threshold)

    print(f'{num_channel} channels will be manipulated under the beta threshold {beta_threshold}')
