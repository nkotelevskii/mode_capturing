import torch
import torch.nn as nn
from tqdm import tqdm
import pdb

def standard_normal(args, target):
    return torch.distributions.Normal(loc=torch.tensor(0., device=args['device'], dtype=args['dtype']),
                                      scale=torch.tensor(1., device=args['device'], dtype=args['dtype']))

def wide_normal(args, target):
    max_loc = 2 * torch.abs(args['locs'][0].max())
    return torch.distributions.Normal(loc=torch.tensor(0., device=args['device'], dtype=args['dtype']),
                                      scale=torch.tensor(max_loc, device=args['device'], dtype=args['dtype']))

def hoffman_prior(args, target):
    ##### Minimize KL first

    mu_init_hoff = nn.Parameter(torch.zeros(args.z_dim, device=args.device, dtype=args.torchType))
    sigma_init_hoff = nn.Parameter(torch.ones(args.z_dim, device=args.device, dtype=args.torchType))
    optimizer = torch.optim.Adam(params=[mu_init_hoff, sigma_init_hoff], lr=1e-3)
    std_normal = standard_normal(args, target)
    for i in tqdm(range(20000)):
        u_init = std_normal.sample((500, args.z_dim))
        q_init = mu_init_hoff + nn.functional.softplus(sigma_init_hoff) * u_init

        current_kl = std_normal.log_prob(u_init).sum(1) - torch.sum(nn.functional.softplus(sigma_init_hoff).log()) - target.get_logdensity(z=q_init)
        torch.mean(current_kl).backward()  ## minimize KL
        optimizer.step()
        optimizer.zero_grad()
        if i % 2000 == 0:
            print(current_kl.mean().cpu().detach().numpy())
    mu_init_hoff.requires_grad_(False)
    sigma_init_hoff.requires_grad_(False)

    class hoff_prior():
        def __init__(self, mu, sigma, std_normal):
            self.mu = mu
            self.sigma = nn.functional.softplus(sigma)
            self.std_normal = std_normal
        def sample(self, shape):
            return self.mu + self.std_normal.sample(shape) * self.sigma

    return hoff_prior(mu_init_hoff, nn.functional.softplus(sigma_init_hoff), std_normal)

def get_prior(args, target):
    if args['prior'] == 'standard_normal':
        return standard_normal(args, target)
    elif args['prior'] == 'wide_normal':
        return wide_normal(args, target)
    elif args['prior'] == 'hoffman':
        return hoffman_prior(args, target)
    else:
        print('No such prior!')
        raise AttributeError