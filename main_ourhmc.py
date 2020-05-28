from aux_functions import dotdict, squared_distance_to_circle, n_modes, make_gaussians, set_seeds
import torch
import torch.nn as nn
import logging
import sys
from target import GMM_target2
from tqdm import tqdm
import numpy as np
from src.transitions import RealNVP_new, HMC_our, Reverse_kernel_sampling
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

    repetitions = args['repetitions']
    target = GMM_target2(kwargs=args)

    if args.learnable_reverse:
        reverse_kernel = Reverse_kernel_sampling(kwargs=args).to(args.device)
        reverse_params = reverse_kernel.parameters()
    else:
        reverse_params = list([])

#     pdb.set_trace()
    if args.amortize:
        transitions = HMC_our(kwargs=args).to(args.device)
    else:
        transitions = nn.ModuleList([HMC_our(kwargs=args).to(args.device) for _ in range(args['K'])])

    if args.fix_transition_params:
        for p in transitions.parameters():
            transitions.requires_grad_(False)

    mu_init = nn.Parameter(torch.zeros(args.z_dim, device=args.device, dtype=args.torchType))
    sigma_init = nn.Parameter(torch.ones(args.z_dim, device=args.device, dtype=args.torchType))
    
    torch_log_2 = torch.tensor(np.log(2.), device=device, dtype=torchType)
    std_normal = torch.distributions.Normal(loc=torch.tensor(0., dtype=torchType, device=device),
                                            scale=torch.tensor(1., dtype=torchType, device=device))

    params = list(transitions.parameters()) + list(reverse_params) + [mu_init, sigma_init]

    optimizer = torch.optim.Adam(params=params)
    
    print_info_ = args["print_info"]

    def compute_loss(z, u, sum_log_jacobian, sum_log_alpha, all_directions=None, sum_log_sigma=0.):
        if args.learnable_reverse:
            log_r = reverse_kernel(z_fin=z.detach(), a=all_directions)
        else:
            log_r = -args.K * torch_log_2
        log_p = target.get_logdensity(z)
        log_m = std_normal.log_prob(u).sum(1) - sum_log_jacobian + sum_log_alpha - sum_log_sigma
        elbo_full = log_p + log_r - log_m
        grad_elbo = torch.mean(elbo_full + elbo_full.detach() * sum_log_alpha)
        return elbo_full, grad_elbo

    final_samples = torch.tensor([], device=args.device)
    iterator = tqdm(range(args.num_batches))
    for batch_num in iterator:  # cycle over batches
        cond_vectors = [std_normal.sample((args.batch_size_train, args.z_dim)) for k in range(args.K)]

        u = std_normal.sample(
            (args['batch_size_train'], args.z_dim))
        sum_log_alpha = torch.zeros(u.shape[0], dtype=args.torchType,
                                    device=args.device)
        sum_log_jacobian = torch.zeros(u.shape[0], dtype=args.torchType,
                                       device=args.device)
        if args['train_prior']:
            sum_log_sigma = torch.sum(nn.functional.softplus(sigma_init).log())
            z = mu_init + u * nn.functional.softplus(sigma_init)
        else:
            sum_log_sigma = 0.
            z = mu_init + u * sigma_init
            
        p = std_normal.sample((args.batch_size_train, args.z_dim))
        sum_log_sigma = torch.sum(nn.functional.softplus(sigma_init).log())
        
        if args.learnable_reverse:
            all_directions = torch.tensor([], device=args.device)
        else:
            all_directions = None
            
        for k in range(K):
            if args.amortize:
                z, p, log_jac, current_log_alphas, directions, q_prop = transitions.make_transition(q_old=z,
                                                p_old=p, k=cond_vectors[k], target_distr=target)
            else:
                z, p, log_jac, current_log_alphas, directions, q_prop = transitions[k].make_transition(q_old=z,
                                                            p_old=p, k=cond_vectors[k], target_distr=target)

            if (batch_num) % print_info_ == 0:
                print('On epoch number {} and on k = {} we have for 1: {}, for 0: {}'.format(
                    batch_num + 1, k + 1,
                    (directions == 1.).to(float).mean(), (directions == 0.).to(float).mean()))
            # Accumulate alphas
            sum_log_alpha = sum_log_alpha + current_log_alphas
            sum_log_jacobian = sum_log_jacobian + log_jac  # refresh log jacobian
            if args.learnable_reverse:
                all_directions = torch.cat([all_directions, directions.detach().view(-1, 1)], dim=1)
        if batch_num >= args.record_epoch:
            final_samples = torch.cat([final_samples, z.detach()], dim=0)

        elbo_full, grad_elbo = compute_loss(z, u, sum_log_jacobian, sum_log_alpha, sum_log_sigma)
        (-grad_elbo).backward()
        optimizer.step()
        optimizer.zero_grad()

        if (batch_num) % print_info_ == 0:
            print('Current epoch:', (batch_num + 1), '\t', 'Current ELBO:', elbo_full.data.mean().item())
    new_n_modes = n_modes(args, final_samples, d, var)
    print(new_n_modes)
    with torch.no_grad():
        if repetitions == 0:
            if new_n_modes > best_n_modes:
                best_n_modes = new_n_modes
            if best_n_modes == len(args["locs"]):
                return best_n_modes
    ##########################################Repetitions############################################

#     increment = 200

#     with torch.no_grad():
#             new_n_modes = n_modes(args, z, d, var)
#             print(new_n_modes)
#             best_n_modes = new_n_modes

    return best_n_modes


def main():
    #################################################################################################################
    ################################################### Arguments ###################################################
    #################################################################################################################
    args = dotdict({})
    args['train_prior'] = True
    args.fix_transition_params = False  # whether to freeze transition params
    args.amortize = False # whether to amortize transitions
    args.learnable_reverse = True  # whether to learn reverse
    args['device'] = device
    args['dtype'] = torchType
    args['num_gauss'] = 8
    args['p_gaussians'] = [torch.tensor(1. / args['num_gauss'], device=device, dtype=torchType)] * args['num_gauss']

    # HMC params

    
    args.K = 5 # How many different kernels to train
    args.N = 1 ## Number of Leapfrogs
    args.gamma = 0.1 ## Stepsize
    args.alpha = 0.5  ## For partial momentum refresh
    args['use_barker'] = True  # If True, we are using Barker's ratios, if not -- vanilla MH
    args['num_batches'] = 1201  # number of batches
    args['batch_size_train'] = 400  # batch size for training
    args['repetitions'] = 0
    args["print_info"] = 250
    
    args.record_epoch = args['num_batches'] - 8000 // args['batch_size_train']

    logging.basicConfig(filename="./results_hmc.txt", level=logging.INFO)
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
    main()