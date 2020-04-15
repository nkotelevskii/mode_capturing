import torch
import numpy as np
import random



class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def squared_distance_to_circle(pt, c):
    d = len(pt)
    res = 0
    for i in range(d):
        res += (pt[i] - c[i]) ** 2
    return res


def n_modes(args, samples, d, var):
    N = len(samples)
    n_modes = 0
    n_in_total = 0
    print("N = {}".format(N))

    for i, loc in enumerate(args["locs"]):
        nb_in = 0
        for s in samples:
            if squared_distance_to_circle(s, loc) < 4 * d * var:
                nb_in += 1
        print("nb_in = {}".format(nb_in))

        if 10 * nb_in > N / args["num_gauss"]:
            n_modes += 1
        n_in_total += nb_in

    print("n_out = {}".format(N - n_in_total))

    return n_modes


def make_gaussians(args, d, var):
    locs_list = []
    locs = []
    while len(locs) < args["num_gauss"]:
        loc = 4 * torch.ones(d, device=args['device'], dtype=args['dtype'])
        for i in range(d):
            if np.random.random() > 0.5:
                loc[i] = -4.
        if not list(loc.cpu().detach().numpy()) in locs_list:
            locs.append(loc)
            locs_list.append(list(loc.cpu().detach().numpy()))

    return locs


def set_seeds(rand_seed):
    torch.cuda.manual_seed_all(rand_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(rand_seed)
    np.random.seed(rand_seed)
    random.seed(rand_seed)