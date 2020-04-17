from aux_functions import dotdict, squared_distance_to_circle, n_modes, make_gaussians, set_seeds
import torch
import logging
import sys
from target import GMM_target2
from pyro.infer import MCMC, NUTS
import time
import numpy as np
from priors import get_prior
import pdb
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""


device = "cpu" #"cuda:0" if torch.cuda.is_available() else "cpu"
torchType = torch.float32


def find_n_modes(args):
    # GMM with arbitraty many components
    var = 1
    d = args["z_dim"]
    args["locs"] = make_gaussians(args, d, var)
    args['covs'] = [var * torch.eye(d, dtype=torch.float32, device=device)] * args[
        'num_gauss']  # list of covariance matrices for each of these gaussians

    target = GMM_target2(kwargs=args)

    #######################################################################################
    ###### NUTS ########
    def energy(z):
        z = z['points']
        return -target.get_logdensity(z)

    prior = get_prior(args, target)

    kernel = NUTS(potential_fn=energy)
    num_samples = 10000
    nuts = torch.tensor([], device=device)
    nuts_ungrouped = torch.tensor([], device=device)
    n_stop = 10  # number of time we stop to check n_modes (to allow fair comparison with other tests, we take the greatest number of modes retrieved)
    warmup_steps = 1500
    n_chains = args['n_chains']
    if n_chains > 1:
        best_n_modes = 0.
    else:
        best_n_modes = np.zeros(n_chains)
    start = time.time()

#     pdb.set_trace()
    init_samples = prior.sample((n_chains, args.z_dim))
    init_params = {'points': init_samples}
#     ## First we run warmup
#     mcmc = MCMC(kernel=kernel, num_samples=1,
#                 initial_params=init_params,
#                 num_chains=n_chains, warmup_steps=warmup_steps)
#     mcmc.run()
#     init_samples = mcmc.get_samples(group_by_chain=True)["points"].view(n_chains,
#                                                                         -1)  # group_by_chain = True allows us to retrieve the last sample of each chain
    pdb.set_trace()
    init_samples = torch.cat([chain[None] for chain in args.locs])
    init_params = {'points': init_samples}
    for i in range(n_stop): ## n_stop -- how often we check n modes
        mcmc = MCMC(kernel=kernel, num_samples=2,#num_samples // n_stop,
                    initial_params=init_params,
                    num_chains=n_chains, warmup_steps=0)
        mcmc.run()
        nuts = torch.cat([nuts, mcmc.get_samples(group_by_chain=True)['points'].view(n_chains, -1, d)], dim=1)
        nuts_ungrouped = torch.cat([nuts_ungrouped, mcmc.get_samples(group_by_chain=False)['points'].view(-1, d)],
                                   dim=0)
        init_samples = nuts[:, -1]  # last sample of each chain (shape = n_chains x z_dim)
        init_params = {'points': init_samples}
        # pdb.set_trace()
        if n_chains > 1:
            new_n_modes = n_modes(args, nuts_ungrouped, d, var)
            if new_n_modes > best_n_modes:
                best_n_modes = new_n_modes
            if best_n_modes == 8:
                break
        else:
            for k in range(n_chains):
                new_n_modes = n_modes(args, nuts[k], d, var)
                if new_n_modes > best_n_modes[k]:
                    best_n_modes[k] = new_n_modes
                if (best_n_modes == 8).all():
                    break
    end = time.time()

    print("time = {}".format(end - start))
    print(best_n_modes)

    return best_n_modes

def main(n_ch, prior_type):
    #################################################################################################################
    ################################################### Arguments ###################################################
    #################################################################################################################
    args = dotdict({})

    args['prior'] = prior_type
    args['device'] = device
    args['dtype'] = torchType
    args['num_gauss'] = 8
    args['p_gaussians'] = [torch.tensor(1. / args['num_gauss'], device=device, dtype=torchType)] * args['num_gauss']
    args['n_chains'] = n_ch

    logging.basicConfig(filename="./results_{}_{}.txt".format(args['n_chains'], args['prior']), level=logging.INFO)
    ################################################################################################
#     pdb.set_trace()
    dim_list = [100] #[20, 50, 100]
    res_list = []

    for _ in range(5):
        for d in dim_list:
            args["z_dim"] = d
            # NUTS parameters
            start = time.time()
            res = find_n_modes(args)
            finish = time.time()
            t = finish - start
            res_list.append(res)
            print("For model {} and dim {}, n_modes = {}, time = {}".format("NUTS", d, res, t))
            logging.info("For model {} and dim {}, n_modes = {}, time = {}".format("NUTS", d, res, t))

    print(res_list)

if __name__ == "__main__":
    n_ch = int(sys.argv[1])
    prior_type = str(sys.argv[2])
    main(n_ch, prior_type)
