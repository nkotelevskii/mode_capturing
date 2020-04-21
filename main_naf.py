from aux_functions import dotdict, squared_distance_to_circle, n_modes, make_gaussians, set_seeds
import torch
import torch.nn as nn
import logging
import sys
from target import GMM_target2
from tqdm import tqdm
import numpy as np
from src.transitions import NAF
from priors import get_prior
import pdb
import time

device = "cuda:1" if torch.cuda.is_available() else "cpu"
torchType = torch.float32


def find_n_modes(args):
    # GMM with arbitraty many components
    var = 1
    d = args["z_dim"]
    args["locs"] = make_gaussians(args, d, var)
    args['covs'] = [var * torch.eye(d, dtype=torch.float32, device=device)] * args[
        'num_gauss']  # list of covariance matrices for each of these gaussians
    K = args["K"]
    best_n_modes = 0

    target = GMM_target2(kwargs=args)

    std_normal = torch.distributions.Normal(loc=torch.tensor(0., dtype=torchType, device=device),
                                            scale=torch.tensor(1., dtype=torchType, device=device))

    transitions = nn.ModuleList([NAF(kwargs=args, device=device).to(device) for _ in range(args['K'])])
    
    mu_init = nn.Parameter(torch.zeros(args.z_dim, device=args.device, dtype=args.torchType),
                           requires_grad=args['train_prior'])
    sigma_init = nn.Parameter(torch.ones(args.z_dim, device=args.device, dtype=args.torchType),
                              requires_grad=args['train_prior'])
    
    params = list(transitions.parameters()) + [mu_init, sigma_init]
    optimizer = torch.optim.Adam(params=params)

    final_samples = torch.tensor([], device=args.device)
    iterator = tqdm(range(args.num_batches))
    for batch_num in iterator:  # cycle over batches
        u = std_normal.sample((args['batch_size_train'], args.z_dim))
        sum_log_jacobian = torch.zeros(u.shape[0], dtype=torchType, device=device)  # for log_jacobian accumulation
        
        if args['train_prior']:
            sum_log_sigma = torch.sum(nn.functional.softplus(sigma_init).log())
            z = mu_init + u * nn.functional.softplus(sigma_init)
        else:
            sum_log_sigma = 0.
            z = mu_init + u * sigma_init
        
        for k in range(K):
            z_upd, log_jac = transitions[k]._forward_step(z)
            sum_log_jacobian = sum_log_jacobian + log_jac - sum_log_sigma  # refresh log jacobian
            z = z_upd

        log_p = target.get_logdensity(z)
        log_q = std_normal.log_prob(u).sum(1) - sum_log_jacobian
        elbo = (log_p - log_q).mean()
        (-elbo).backward()
        optimizer.step()
        optimizer.zero_grad()

        if (batch_num) % args["print_info"] == 0:
            print('Current epoch:', (batch_num + 1), '\t', 'Current ELBO:', elbo.cpu().detach().numpy())
            
        if batch_num >= args.record_epoch:
            final_samples = torch.cat([final_samples, z.detach()], dim=0)
            
    with torch.no_grad():
        new_n_modes = n_modes(args, final_samples, d, var)
        print(new_n_modes)
        best_n_modes = new_n_modes
        
    return best_n_modes


def main(prior_type):
    #################################################################################################################
    ################################################### Arguments ###################################################
    #################################################################################################################
    args = dotdict({})
    args['train_prior'] = True if prior_type == 'train' else False
    args['prior'] = prior_type
    args['device'] = device
    args['dtype'] = torchType
    args['num_gauss'] = 8
    args['p_gaussians'] = [torch.tensor(1. / args['num_gauss'], device=device, dtype=torchType)] * args['num_gauss']

    args['K'] = 10

    args['num_batches'] = 15000  # number of batches
    args['batch_size_train'] = 400  # batch size for training
    args['repetitions'] = 0
    args["print_info"] = 250
    args.record_epoch = args['num_batches'] - 8000 // args['batch_size_train']

    logging.basicConfig(filename="./results_naf_{}.txt".format(args['prior']), level=logging.INFO)
    ################################################################################################
    dim_list = [3, 5, 7, 10, 20, 50, 100]
    res_list = []

    for _ in range(5):
        for d in dim_list:
            print('Current dimensionality: ', d)
            args["z_dim"] = d

            start = time.time()
            res = find_n_modes(args)
            finish = time.time()
            t = finish - start
            print("For model {} and dim {}, n_modes = {}, time = {}".format("algo_fixed", d, res, t))
            logging.info("For model {} and dim {}, n_modes = {}, time = {}".format("algo_fixed", d, res, t))
            res_list.append(res)

    print(res_list)


if __name__ == "__main__":
    prior_type = str(sys.argv[1])  # either train or fix
    main(prior_type)