from aux_functions import dotdict, squared_distance_to_circle, n_modes, make_gaussians, set_seeds
import torch
import torch.nn as nn
import logging
import sys
from target import GMM_target2
from tqdm import tqdm
import numpy as np
from src.transitions import RealNVP_new
from priors import get_prior
import pdb

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

    if args.step_conditioning is None:
        transitions = nn.ModuleList([RealNVP_new(kwargs=args,
                                                 device=args.device).to(args.device) for _ in range(args['K'])])
    else:
        transitions = RealNVP_new(kwargs=args, device=args.device).to(args.device)

    torch_log_2 = torch.tensor(np.log(2.), device=device, dtype=torchType)
    std_normal = torch.distributions.Normal(loc=torch.tensor(0., dtype=torchType, device=device),
                                            scale=torch.tensor(1., dtype=torchType, device=device))

    mu_init = nn.Parameter(torch.zeros(args.z_dim, device=args.device, dtype=args.torchType),
                           requires_grad=args['train_prior'])
    sigma_init = nn.Parameter(torch.ones(args.z_dim, device=args.device, dtype=args.torchType),
                              requires_grad=args['train_prior'])

    params = list(transitions.parameters()) + [mu_init, sigma_init]
    optimizer = torch.optim.Adam(params=params)
    print_info_ = args["print_info"]

    if args.step_conditioning == 'fixed':
        cond_vectors = [[std_normal.sample((args.z_dim,)), args.noise_aggregation] for _ in range(K)]
    else:
        cond_vectors = [None] * K

    def compute_loss(z, u, sum_log_jacobian, sum_log_alpha, sum_log_probs):
        log_p = target.get_logdensity(z)
        log_r = -args.K * torch_log_2
        log_m_tilde = std_normal.log_prob(u).sum(1) - sum_log_jacobian
        log_m = log_m_tilde + sum_log_alpha
        elbo_full = log_p + log_r - log_m
        grad_elbo = torch.mean(elbo_full + elbo_full.detach() * (sum_log_alpha + sum_log_probs))
        return elbo_full, grad_elbo

    iterator = tqdm(range(args.num_batches))
    for batch_num in iterator:  # cycle over batches
        if args.step_conditioning == 'free':
            cond_vectors = [[std_normal.sample((args.z_dim,)), args.noise_aggregation] for _ in range(K)]

        u = std_normal.sample(
            (args['batch_size_train'], args.z_dim))
        sum_log_alpha = torch.zeros(u.shape[0], dtype=args.torchType,
                                    device=args.device)
        sum_log_probs = torch.zeros(u.shape[0], dtype=args.torchType,
                                    device=args.device)
        sum_log_jacobian = torch.zeros(u.shape[0], dtype=args.torchType,
                                       device=args.device)
        z = mu_init + u * sigma_init
        for k in range(K):
            if args.step_conditioning is None:
                z, log_jac, current_log_alphas, \
                current_log_probs, directions = transitions[k].make_transition(z_old=z, k=cond_vectors[k],
                                                                               target_distr=target)
            else:
                z, log_jac, current_log_alphas, \
                current_log_probs, directions = transitions.make_transition(z_old=z, k=cond_vectors[k],
                                                                            target_distr=target)

            if (batch_num) % print_info_ == 0:
                print('On epoch number {} and on k = {} we have for -1: {}, for 0: {} and for +1: {}'.format(
                    batch_num + 1, k + 1,
                    (directions == -1.).to(float).mean(), (directions == 0.).to(float).mean(),
                    (directions == 1.).to(float).mean()))
            # Accumulate alphas
            sum_log_alpha = sum_log_alpha + current_log_alphas
            sum_log_probs = sum_log_probs + current_log_probs
            sum_log_jacobian = sum_log_jacobian + log_jac  # refresh log jacobian

        elbo_full, grad_elbo = compute_loss(z, u, sum_log_jacobian, sum_log_alpha, sum_log_probs)
        (-grad_elbo).backward()
        optimizer.step()
        optimizer.zero_grad()

        if (batch_num) % print_info_ == 0:
            print('Current epoch:', (batch_num + 1), '\t', 'Current ELBO:', elbo_full.data.mean().item())
            new_n_modes = n_modes(args, z, d, var)
            print(new_n_modes)
            if repetitions == 0:
                if new_n_modes > best_n_modes:
                    best_n_modes = new_n_modes
                if best_n_modes == len(args["locs"]):
                    return best_n_modes
    ##########################################Repetitions############################################

    increment = 200

    with torch.no_grad():
        if repetitions:
            for rep in tqdm(range(repetitions)):
                for k in range(K):
                    if args.step_conditioning == 'free':
                        cond_vectors = [[std_normal.sample((args.z_dim,)), args.noise_aggregation] for k in range(K)]
                    # sample alpha - transition probabilities
                    if args.step_conditioning is None:
                        z, _, _, \
                        _, _ = transitions[k].make_transition(z_old=z, k=cond_vectors[k],
                                                              target_distr=target)
                    else:
                        z, _, _, \
                        _, _ = transitions.make_transition(z_old=z, k=cond_vectors[k],
                                                           target_distr=target)
                if rep % increment == 0:
                    new_n_modes = n_modes(args, z, d, var)
                    print(new_n_modes)
                    if new_n_modes > best_n_modes:
                        best_n_modes = new_n_modes
                    if best_n_modes == len(args["locs"]):
                        return best_n_modes
                print('Current epoch:', (rep + 1))
                n_modes(args, z, d, var)
        else:
            new_n_modes = n_modes(args, z, d, var)
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

    # MetFlow parameters
    args['K'] = 7
    args['p'] = 0.5  # probability of forward transition

    args['step_conditioning'] = None  # fixed, free, None
    args['noise_aggregation'] = 'stacking'  # addition, stacking

    args['use_barker'] = True  # If True, we are using Barker's ratios, if not -- vanilla MH
    args['num_batches'] = 10000  # number of batches
    args['batch_size_train'] = 300  # batch size for training
    args['repetitions'] = 0
    args["print_info"] = 1000

    logging.basicConfig(filename="./results_metflow_{}.txt".format(args['prior']), level=logging.INFO)
    ################################################################################################
    dim_list = [3, 5, 7, 10, 20, 50, 100]
    res_list = []

    for _ in range(5):
        for d in dim_list:
            print('Current dimensionality: ', d)
            args["z_dim"] = d
            # RNVP parameters
            args['hidden_dim'] = 2 * args["z_dim"]
            if args['step_conditioning'] is None:
                args['nets'] = lambda: nn.Sequential(nn.Linear(args['z_dim'], args['hidden_dim']), nn.LeakyReLU(),
                                                     nn.Linear(args['hidden_dim'], args['hidden_dim']),
                                                     nn.LeakyReLU(), nn.Linear(args['hidden_dim'], args['z_dim']),
                                                     nn.Tanh())
                args['nett'] = lambda: nn.Sequential(nn.Linear(args['z_dim'], args['hidden_dim']), nn.LeakyReLU(),
                                                     nn.Linear(args['hidden_dim'], args['hidden_dim']),
                                                     nn.LeakyReLU(), nn.Linear(args['hidden_dim'], args['z_dim']))
            else:
                if args['noise_aggregation'] == 'addition':
                    args['nets'] = lambda: nn.Sequential(nn.Linear(args['z_dim'], args['hidden_dim']), nn.LeakyReLU(),
                                                         nn.Linear(args['hidden_dim'], args['hidden_dim']),
                                                         nn.LeakyReLU(), nn.Linear(args['hidden_dim'], args['z_dim']),
                                                         nn.Tanh())
                    args['nett'] = lambda: nn.Sequential(nn.Linear(args['z_dim'], args['hidden_dim']), nn.LeakyReLU(),
                                                         nn.Linear(args['hidden_dim'], args['hidden_dim']),
                                                         nn.LeakyReLU(), nn.Linear(args['hidden_dim'], args['z_dim']))
                elif args['noise_aggregation'] == 'stacking':
                    args['nets'] = lambda: nn.Sequential(nn.Linear(args['z_dim'] * 2, args['hidden_dim']),
                                                         nn.LeakyReLU(),
                                                         nn.Linear(args['hidden_dim'], args['hidden_dim']),
                                                         nn.LeakyReLU(), nn.Linear(args['hidden_dim'], args['z_dim']),
                                                         nn.Tanh())
                    args['nett'] = lambda: nn.Sequential(nn.Linear(args['z_dim'] * 2, args['hidden_dim']),
                                                         nn.LeakyReLU(),
                                                         nn.Linear(args['hidden_dim'], args['hidden_dim']),
                                                         nn.LeakyReLU(), nn.Linear(args['hidden_dim'], args['z_dim']))

            args['masks'] = np.array(
                [[i % 2 for i in range(args["z_dim"])], [(i + 1) % 2 for i in range(args["z_dim"])]]).astype(np.float32)

            res = find_n_modes(args)
            print("For model {} and dim {}, n_modes = {}".format("algo_fixed", d, res))
            logging.info("For model {} and dim {}, n_modes = {}".format("algo_fixed", d, res))
            res_list.append(res)

    print(res_list)


if __name__ == "__main__":
    prior_type = str(sys.argv[1])  # either train or fix
    main(prior_type)