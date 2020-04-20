import torch
import numpy as np
import torch.nn as nn
import pdb

from pyro.distributions.transforms import BlockAutoregressive, NeuralAutoregressive
torchType = torch.float32

from pyro.nn import AutoRegressiveNN
from pyro.distributions.transforms import NeuralAutoregressive


class Transition_new(nn.Module):
    """
    Base class for custom transitions of new type
    """
    def __init__(self, kwargs, device):
        super(Transition_new, self).__init__()
        self.use_barker = kwargs.use_barker  # If false, we are using standard MH ration, otherwise Barker ratio
        self.device = device
        self.device_zero = torch.tensor(0., dtype=torchType, device=self.device)
        self.device_one = torch.tensor(1., dtype=torchType, device=self.device)
        self.direction_distr = torch.distributions.Uniform(low=self.device_zero,
                                                            high=self.device_one)  # distribution for transition making
        self.logit = nn.Parameter(torch.tensor(np.log(kwargs['p']) - np.log(1 - kwargs['p']),
                                               dtype=torchType, device=self.device))  # probability of forward transition


    def _forward_step(self, z_old, x=None, k=None, target=None):
        """
        The function makes forward step
        Also, this function computes log_jacobian of the transformation
        Input:
        z_old - current position
        x - data object (optional)
        k - number of transition (optional)
        Output:
        z_new - new position = T(z_old)
        log_jac - log_jacobian of the transformation
        """
        # You should define the class for your custom transition
        raise NotImplementedError

    def _backward_step(self, z_old, x=None, k=None, target=None):
        """
        The function makes backward step
        Input:
        z_old - current position
        x - data object (optional)
        k - number of transition (optional)
        Output:
        z_new - new position T^(-1)(z_old)
        log_jac - log_jacobian of the transformation
        """
        # You should define the class for your custom transition
        raise NotImplementedError

    def make_transition(self, z_old, target_distr, k=None, x=None, detach=False, p_old=None):
        """
        The function returns directions (-1, 0 or +1), sampled in the current positions
        Input:
        z_old - point of evaluation
        target_distr - target distribution
        x - data object (optional)
        k - number of transition (optional)
        detach - whether to detach target (decoder) from computational graph or not
        p_old - auxilary variables for some types of transitions
        Output:
        z_new - new points
        log_jac - log jacobians of transformations
        current_log_alphas - current log_alphas, corresponding to sampled decision variables
        current_log_probs - current log probabilities of forward transition
        a - decision variables (-1, 0 or +1)
        """
        ############ Sample v_1 -- either +1 or -1 ############
        p = torch.sigmoid(self.logit)
        probs = self.direction_distr.sample((z_old.shape[0], ))
        v_1 = torch.where(probs < p, self.device_one, -self.device_one)

        ############ Then we compute new points and densities ############
        # pdb.set_trace()
        z_f, log_jac_f = self._forward_step(z_old=z_old, k=k)
        z_b, log_jac_b = self._backward_step(z_old=z_old, k=k)

        target_log_density_f = target_distr.get_logdensity(z=z_f, x=x)
        target_log_density_b = target_distr.get_logdensity(z=z_b, x=x)
        target_log_density_old = target_distr.get_logdensity(z=z_old, x=x)
        ############ Then we select only those which correspond to selected direction ############
        target_log_density_new = torch.where(v_1 == 1., target_log_density_f, target_log_density_b)
        current_probs = torch.where(v_1 == 1., p, 1 - p)
        new_log_jacobian = torch.where(v_1 == 1., log_jac_f, log_jac_b)
        ############### Compute acceptance ratio ##############
        log_t = target_log_density_new + torch.log(1. - current_probs) + new_log_jacobian\
                - target_log_density_old - torch.log(current_probs)
        ############### Two expressions for performing transition  ##############
        log_1_t = torch.logsumexp(torch.cat([torch.zeros_like(log_t).view(-1, 1),
                                                                        log_t.view(-1, 1)], dim=-1), dim=-1)
        if self.use_barker:
            current_log_alphas_pre = log_t - log_1_t
        else:
            current_log_alphas_pre = torch.min(self.device_zero, log_t)
        log_probs = torch.log(self.direction_distr.sample((z_old.shape[0],)))
        a = torch.where(log_probs < current_log_alphas_pre, v_1, self.device_zero)

        if self.use_barker:
            current_log_alphas = torch.where((a == 0), -log_1_t, current_log_alphas_pre)
        else:
            expression = 1. - torch.exp(log_t)
            expression = torch.where(expression <= self.device_zero, self.device_one * 1e-8, expression)
            corr_expression = torch.log(expression)
            current_log_alphas = torch.where((a == 0), corr_expression, current_log_alphas_pre)

        current_log_probs = torch.log(current_probs)

        z_new = torch.where((a == -self.device_one)[:, None], z_b,
                            torch.where((a == self.device_zero)[:, None], z_old, z_f))

        same_log_jac = torch.zeros_like(new_log_jacobian)
        log_jac = torch.where((a == self.device_zero), same_log_jac, new_log_jacobian)

        return z_new, log_jac, current_log_alphas, current_log_probs, a



class RealNVP_new(Transition_new):
    def __init__(self, kwargs, device):
        super(RealNVP_new, self).__init__(kwargs, device)
        if "n_layers" in kwargs.keys():
            self.n_layers = kwargs["n_layers"]
        else:
            self.n_layers = 1

        if "n_flows" in kwargs.keys(): # number of RNVP steps
            self.n_flows = kwargs["n_flows"]
        elif "masks" in kwargs.keys():
            self.n_flows = len(kwargs["masks"])
        else:
            self.n_flows = 1


        self.hidden_dim = kwargs["hidden_dim"]
        self.z_dim = kwargs["z_dim"]

        if "masks" in kwargs.keys():
            self.masks = torch.tensor(kwargs['masks'], device=self.device).to(device)
        else:
            self.masks = []
            for i in range(self.n_flows):
                if i % 2:
                    self.masks.append(torch.cat((torch.ones((self.z_dim//2), device=device),
                                                 torch.zeros((self.z_dim - self.z_dim//2), device=device))))
                else:
                    self.masks.append(
                        torch.cat((torch.zeros((self.z_dim // 2), device=device),
                                   torch.ones((self.z_dim - self.z_dim // 2), device=device))))


        if len(self.masks) != self.n_flows:
            message = "mask list length : {}, n_flows : {}".format(len(self.masks), self.n_flows)
            print("WARNING : {}".format(message))


        nett, nets = kwargs["nett"], kwargs["nets"] # custom nets for t and s. Overwritten if not

        if nets == "linear": # if nets is not custom, we build a linear network with the right number of layers (n_layers)
            nets = lambda: nn.Sequential(
                *[nn.Linear(self.z_dim, self.hidden_dim), nn.LeakyReLU()] + [nn.Linear(self.hidden_dim, self.hidden_dim), nn.LeakyReLU()] * self.n_layers + [nn.Linear(self.hidden_dim, self.z_dim), nn.Tanh()] )
        if nett == "linear":
            if kwargs['step_conditioning']:
                if kwargs['noise_aggregation'] in ['addition',  None]:
                    nett = lambda : nn.Sequential(*[nn.Linear(self.z_dim, self.hidden_dim), nn.LeakyReLU()] + [nn.Linear(self.hidden_dim, self.hidden_dim), nn.LeakyReLU()] * self.n_layers + [nn.Linear(self.hidden_dim, self.z_dim)])
                elif kwargs['noise_aggregation'] == 'stacking':
                    nett = lambda : nn.Sequential(*[nn.Linear(2 * self.z_dim, 2 * self.hidden_dim), nn.LeakyReLU()] + [nn.Linear(2 * self.hidden_dim, 2 * self.hidden_dim), nn.LeakyReLU()] * (self.n_layers - 1) + [nn.Linear(2 * self.hidden_dim, self.hidden_dim), nn.Linear(self.hidden_dim, self.z_dim)]) # n_layers - 1


        if nets == "convolutional":
            raise NotImplementedError

        if nett == "convolutional":
            raise NotImplementedError


        self.t = nn.ModuleList([nett() for _ in range(self.n_flows)]).to(device)
        self.s = nn.ModuleList([nets() for _ in range(self.n_flows)]).to(device)

    def _forward_step(self, z_old, x=None, k=None, target=None, p_old=None):
        """
        The function makes forward step
        Also, this function computes log_jacobian of the transformation
        Input:
        z_old - current position
        x - data object (optional)
        k - number of transition (optional) k[0] - value, k[1] - method
        target - target class (optional)
        p_old - auxilary variables for some types of transitions (like momentum for HMC)
        Output:
        z_new - new position = T(z_old)
        log_jac - log_jacobian of the transformation
        """
        log_det_J = z_old.new_zeros(z_old.shape[0])
        if k is not None:
            noise = torch.ones_like(z_old) * k[0]
        for i in range(self.n_flows):
            z_old_ = z_old * self.masks[i]
            if k is not None:
                if k[1] == 'stacking':
                    stacked_vector = torch.cat([z_old_, noise], dim=1)
                    s = self.s[i](stacked_vector) * (self.device_one - self.masks[i])
                    t = self.t[i](stacked_vector) * (self.device_one - self.masks[i])
                elif k[1] == 'addition':
                    added_vector = z_old_ + noise
                    s = self.s[i](added_vector) * (self.device_one - self.masks[i])
                    t = self.t[i](added_vector) * (self.device_one - self.masks[i])
            else:
                s = self.s[i](z_old_) * (self.device_one - self.masks[i])
                t = self.t[i](z_old_) * (self.device_one - self.masks[i])
            z_old = z_old_ + (self.device_one - self.masks[i]) * (z_old * torch.exp(s) + t)
            if len(s.squeeze()) == 2:
                log_det_J += s.squeeze().sum(dim=1)
            else:
                log_det_J += s.sum(dim=1)
        z_new = z_old
        return z_new, log_det_J

    def _backward_step(self, z_old, x=None, k=None, target=None, p_old=None):
        """
        The function makes backward step
        Input:
        z_old - current position
        x - data object (optional)
        k - number of transition (optional)
        target - target class (optional)
        p_old - auxilary variables for some types of transitions (like momentum for HMC)
        Output:
        z_new - new position T^(-1)(z_old)
        log_jac - log_jacobian of the transformation
        """
        log_det_J, z = z_old.new_zeros(z_old.shape[0]), z_old
        if k is not None:
            noise = torch.ones_like(z_old) * k[0]
        for i in reversed(range(self.n_flows)):
            z_ = self.masks[i] * z
            if k is not None:
                if k[1] == 'stacking':
                    stacked_vector = torch.cat([z_, noise], dim=1)
                    t = self.t[i](stacked_vector) * (self.device_one - self.masks[i])
                    s = self.s[i](stacked_vector) * (self.device_one - self.masks[i])
                elif k[1] == 'addition':
                    added_vector = z_ + noise
                    t = self.t[i](added_vector) * (self.device_one - self.masks[i])
                    s = self.s[i](added_vector) * (self.device_one - self.masks[i])
            else:
                t = self.t[i](z_) * (self.device_one - self.masks[i])
                s = self.s[i](z_) * (self.device_one - self.masks[i])
            z = (self.device_one - self.masks[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        z_new = z
        return z_new, log_det_J

    def log_prob(self, x):
        z, logp = self.f(x)
        return self.prior.log_prob(z) + logp

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        x, log_det_J = self.g(z)
        return x

class BNAF(Transition_new):
    def __init__(self, kwargs, device):
        super(BNAF, self).__init__(kwargs, device)
        self.z_dim = kwargs["z_dim"]
        self.flow = BlockAutoregressive(self.z_dim)


    def _forward_step(self, z_old, x=None, k=None):
        """
        The function makes forward step
        Also, this function computes log_jacobian of the transformation
        Input:
        z_old - current position
        x - data object (optional)
        k - number of transition (optional) k[0] - value, k[1] - method
        Output:
        z_new - new position = T(z_old)
        log_jac - log_jacobian of the transformation
        """
        z_new = self.flow(z_old)
        log_det_J = self.flow.log_abs_det_jacobian(z_old, z_new)
        return z_new, log_det_J

    def _backward_step(self, z_old, x=None, k=None):
        """
        The function makes backward step
        Input:
        z_old - current position
        x - data object (optional)
        k - number of transition (optional)
        Output:
        z_new - new position T^(-1)(z_old)
        log_jac - log_jacobian of the transformation
        """
        raise NotImplementedError
    def log_prob(self, x):
        raise NotImplementedError

    def sample(self, batchSize):
        raise NotImplementedError

class NAF(Transition_new):
    def __init__(self, kwargs, device):
        super(NAF, self).__init__(kwargs, device)
        self.z_dim = kwargs["z_dim"]
        self.flow = NeuralAutoregressive(
                    AutoRegressiveNN(self.z_dim, [2 * self.z_dim], param_dims=[self.z_dim] * 3),
                    hidden_units=self.z_dim)


    def _forward_step(self, z_old, x=None, k=None):
        """
        The function makes forward step
        Also, this function computes log_jacobian of the transformation
        Input:
        z_old - current position
        x - data object (optional)
        k - number of transition (optional) k[0] - value, k[1] - method
        Output:
        z_new - new position = T(z_old)
        log_jac - log_jacobian of the transformation
        """
        z_new = self.flow(z_old)
        log_det_J = self.flow.log_abs_det_jacobian(z_old, z_new)
        return z_new, log_det_J

    def _backward_step(self, z_old, x=None, k=None):
        """
        The function makes backward step
        Also, this function computes log_jacobian of the transformation
        Input:
        z_old - current position
        x - data object (optional)
        k - number of transition (optional)
        Output:
        z_new - new position T^(-1)(z_old)
        log_jac - log_jacobian of the transformation
        """
        raise NotImplementedError
    def log_prob(self, x):
        raise NotImplementedError

    def sample(self, batchSize):
        raise NotImplementedError

        
class HMC_our(nn.Module):
    def __init__(self, kwargs):
        super(HMC_our, self).__init__()
        self.device = kwargs.device
#         self.gamma_logit = nn.Parameter(torch.tensor(np.log(kwargs.gamma) - np.log(1. - kwargs.gamma), device=self.device))
        self.gamma = nn.Parameter(torch.tensor(np.log(kwargs.gamma), device=self.device))
        self.N = kwargs.N # num leapfrogs
        self.alpha_logit = nn.Parameter(torch.tensor(np.log(kwargs.alpha) - np.log(1. - kwargs.alpha), device=self.device), requires_grad=True)
        self.use_barker = kwargs.use_barker  # If false, we are using standard MH ration, otherwise Barker ratio
        self.device_zero = torch.tensor(0., dtype=kwargs.torchType, device=self.device)
        self.device_one = torch.tensor(1., dtype=kwargs.torchType, device=self.device)
        self.uniform = torch.distributions.Uniform(low=self.device_zero,
                                                   high=self.device_one)  # distribution for transition making
        self.std_normal = torch.distributions.Normal(loc=self.device_zero, scale=self.device_one)
        self.naf = None
        if kwargs.neutralizing_idea:
            self.naf = kwargs.naf

    def _forward_step(self, q_old, x=None, k=None, target=None, p_old=None):
        """
        The function makes forward step
        Also, this function computes log_jacobian of the transformation
        Input:
        q_old - current position
        x - data object (optional)
        k - number of transition (optional) k[0] - value, k[1] - method
        target - target class (optional)
        p_old - auxilary variables for some types of transitions (like momentum for HMC)
        Output:
        q_new - new position = T(q_old)
        log_jac - log_jacobian of the transformation
        """
#         gamma = torch.sigmoid(self.gamma_logit)  # to make eps positive
#         pdb.set_trace()
        gamma = torch.exp(self.gamma)
        p_flipped = -p_old

        p_ = p_flipped + gamma / 2. * self.get_grad(q=q_old, target=target, x=x)  # NOTE that we are using log-density, not energy!
        q_ = q_old
        for l in range(self.N):
            q_ = q_ + gamma * p_
            if (l != self.N - 1):
                p_ = p_ + gamma * self.get_grad(q=q_, target=target, x=x)  # NOTE that we are using log-density, not energy!
        p_ = p_ + gamma / 2. * self.get_grad(q=q_, target=target, x=x)  # NOTE that we are using log-density, not energy!
        return q_, p_

    def _backward_step(self, q_old, x=None, k=None, target=None, p_old=None):
        """
        The function makes backward step
        Input:
        q_old - current position
        x - data object (optional)
        k - number of transition (optional)
        target - target class (optional)
        p_old - auxilary variables for some types of transitions (like momentum for HMC)
        Output:
        q_new - new position T^(-1)(q_old)
        log_jac - log_jacobian of the transformation
        """
        pass
        # alpha = torch.sigmoid(self.alpha_logit) # to make alpha within (0, 1)
        # eps = torch.exp(self.eps) # to make eps positive
        # z_ = z_old
        # p_ = p_old - eps / 2. * target.get_logdensity(x=x, z=z_)  # NOTE that we are using log-density, not energy!
        # for l in range(self.L):
        #     z_ = z_ - eps * p_
        #     if (l != self.L - 1):
        #         p_ = p_ - eps * target.get_logdensity(x=x, z=z_)# NOTE that we are using log-density, not energy!
        # p_ = p_ - eps / 2. * target.get_logdensity(x=x, z=z_old)  # NOTE that we are using log-density, not energy!
        # p_old_refreshed = (p_ - torch.sqrt(1. - alpha**2) * k[0]) / alpha
        # log_jac = -p_old.shape[1] * torch.log(alpha)
        # return [z_, p_, p_old_refreshed, log_jac]

    def make_transition(self, q_old, p_old, target_distr, k=None, x=None, flows=None, args=None, get_prior=None, prior_flow=None):
        """
        The function returns directions (-1, 0 or +1), sampled in the current positions
        Input:
        q_old - point of evaluation
        target_distr - target distribution
        x - data object (optional)
        k - number of transition (optional)
        detach - whether to detach target (decoder) from computational graph or not
        Output:
        q_new - new points
        log_jac - log jacobians of transformations
        current_log_alphas - current log_alphas, corresponding to sampled decision variables
        current_log_probs - current log probabilities of forward transition
        a - decision variables (-1, 0 or +1)
        p_new (optional) - new momentum (for HMC, if p_old is not None)
        """
        ### Partial momentum refresh
        alpha = torch.sigmoid(self.alpha_logit)
        p_ref = p_old * alpha + torch.sqrt(1. - alpha ** 2) * k
        log_jac = p_old.shape[1] * torch.log(alpha) * torch.ones(q_old.shape[0], device=self.device)
        ############ Then we compute new points and densities ############
        q_upd, p_upd = self._forward_step(q_old=q_old, p_old=p_ref, k=k, target=target_distr, x=x)

        target_log_density_f = target_distr.get_logdensity(z=q_upd, x=x, prior=get_prior, args=args, prior_flow=prior_flow) + self.std_normal.log_prob(p_upd).sum(1)
        target_log_density_old = target_distr.get_logdensity(z=q_old, x=x, prior=get_prior, args=args, prior_flow=prior_flow) + self.std_normal.log_prob(p_ref).sum(1)

        log_t = target_log_density_f - target_log_density_old
        log_1_t = torch.logsumexp(torch.cat([torch.zeros_like(log_t).view(-1, 1),
                                             log_t.view(-1, 1)], dim=-1), dim=-1)  # log(1+t)
        if self.use_barker:
            current_log_alphas_pre = log_t - log_1_t
        else:
            current_log_alphas_pre = torch.min(self.device_zero, log_t)

        log_probs = torch.log(self.uniform.sample((q_upd.shape[0],)))
        a = torch.where(log_probs < current_log_alphas_pre, self.device_one, self.device_zero)

        if self.use_barker:
            current_log_alphas = torch.where((a == 0.), -log_1_t, current_log_alphas_pre)
        else:
            expression = 1. - torch.exp(log_t)
            expression = torch.where(expression <= self.device_one * 1e-8, self.device_one * 1e-8, expression)
            corr_expression = torch.log(expression)
            current_log_alphas = torch.where((a == 0), corr_expression, current_log_alphas_pre)

        q_new = torch.where((a == self.device_zero)[:, None], q_old, q_upd)
        p_new = torch.where((a == self.device_zero)[:, None], p_ref, p_upd)

        return q_new, p_new, log_jac, current_log_alphas, a, q_upd

    def get_grad(self, q, target, x=None):
        q_init = q.detach().requires_grad_(True)
        if self.naf:
            sum_log_jac = torch.zeros(q_init.shape[0], device=self.device)
            q_prev = q_init
            for naf in self.naf:
                q = naf(q_prev)
                sum_log_jac = sum_log_jac + naf.log_abs_det_jacobian(q_prev, q)
                q_prev = q
            grad = torch.autograd.grad((target.get_logdensity(x=x, z=q) + sum_log_jac).sum(), q_init)[
                0]
        else:
            grad = torch.autograd.grad(target.get_logdensity(x=x, z=q_init).sum(), q_init)[
        0]
        return grad
    
    
class HMC_vanilla(nn.Module):
    def __init__(self, kwargs):
        super(HMC_vanilla, self).__init__()
        self.device = kwargs.device
        self.N = kwargs.N
        
        self.alpha_logit = torch.tensor(np.log(kwargs.alpha) - np.log(1. - kwargs.alpha), device=self.device)
#         self.gamma_logit = torch.tensor(np.log(kwargs.gamma) - np.log(1. - kwargs.gamma), device=self.device)
        self.gamma = torch.tensor(np.log(kwargs.gamma), device=self.device)
        self.use_partialref = kwargs.use_partialref  # If false, we are using full momentum refresh
        self.use_barker = kwargs.use_barker  # If false, we are using standard MH ration, otherwise Barker ratio
        self.device_zero = torch.tensor(0., dtype=kwargs.torchType, device=self.device)
        self.device_one = torch.tensor(1., dtype=kwargs.torchType, device=self.device)
        self.uniform = torch.distributions.Uniform(low=self.device_zero,
                                                   high=self.device_one)  # distribution for transition making
        self.std_normal = torch.distributions.Normal(loc=self.device_zero, scale=self.device_one)

    def _forward_step(self, q_old, x=None, k=None, target=None, p_old=None, flows=None):
        """
        The function makes forward step
        Also, this function computes log_jacobian of the transformation
        Input:
        q_old - current position
        x - data object (optional)
        k - number of transition (optional) k[0] - value, k[1] - method
        target - target class (optional)
        p_old - auxilary variables for some types of transitions (like momentum for HMC)
        Output:
        q_new - new position = T(q_old)
        log_jac - log_jacobian of the transformation
        """
#         gamma = torch.sigmoid(self.gamma_logit)
        gamma = torch.exp(self.gamma)
        p_flipped = -p_old
        q_old.requires_grad_(True)
        p_ = p_flipped + gamma / 2. * self.get_grad(q=q_old, target=target, x=x, flows=flows)  # NOTE that we are using log-density, not energy!
        q_ = q_old
        for l in range(self.N):
            q_ = q_ + gamma * p_
            if (l != self.N - 1):
                p_ = p_ + gamma * self.get_grad(q=q_, target=target, x=x, flows=flows)  # NOTE that we are using log-density, not energy!
        p_ = p_ + gamma / 2. * self.get_grad(q=q_, target=target, x=x, flows=flows)  # NOTE that we are using log-density, not energy!

        p_ = p_.detach()
        q_ = q_.detach()
        q_old.requires_grad_(False)

        return q_, p_

    def _backward_step(self, q_old, x=None, k=None, target=None, p_old=None):
        """
        The function makes backward step
        Input:
        q_old - current position
        x - data object (optional)
        k - number of transition (optional)
        target - target class (optional)
        p_old - auxilary variables for some types of transitions (like momentum for HMC)
        Output:
        q_new - new position T^(-1)(q_old)
        log_jac - log_jacobian of the transformation
        """
        pass
        # alpha = torch.sigmoid(self.alpha_logit) # to make alpha within (0, 1)
        # eps = torch.exp(self.eps) # to make eps positive
        # z_ = z_old
        # p_ = p_old - eps / 2. * target.get_logdensity(x=x, z=z_)  # NOTE that we are using log-density, not energy!
        # for l in range(self.L):
        #     z_ = z_ - eps * p_
        #     if (l != self.L - 1):
        #         p_ = p_ - eps * target.get_logdensity(x=x, z=z_)# NOTE that we are using log-density, not energy!
        # p_ = p_ - eps / 2. * target.get_logdensity(x=x, z=z_old)  # NOTE that we are using log-density, not energy!
        # p_old_refreshed = (p_ - torch.sqrt(1. - alpha**2) * k[0]) / alpha
        # log_jac = -p_old.shape[1] * torch.log(alpha)
        # return [z_, p_, p_old_refreshed, log_jac]

    def make_transition(self, q_old, p_old, target_distr, k=None, x=None, flows=None, args=None, get_prior=None, prior_flow=None):
        """
        The function returns directions (-1, 0 or +1), sampled in the current positions
        Input:
        q_old - point of evaluation
        target_distr - target distribution
        x - data object (optional)
        k - number of transition (optional)
        detach - whether to detach target (decoder) from computational graph or not
        Output:
        q_new - new points
        log_jac - log jacobians of transformations
        current_log_alphas - current log_alphas, corresponding to sampled decision variables
        current_log_probs - current log probabilities of forward transition
        a - decision variables (-1, 0 or +1)
        p_new (optional) - new momentum (for HMC, if p_old is not None)
        """
        # pdb.set_trace()
        ### Partial momentum refresh
        alpha = torch.sigmoid(self.alpha_logit)
        if self.use_partialref:
            p_ref = p_old * alpha + torch.sqrt(1. - alpha ** 2) * self.std_normal.sample(p_old.shape)
        else:
            p_ref = self.std_normal.sample(p_old.shape)
        
        ############ Then we compute new points and densities ############
        q_upd, p_upd = self._forward_step(q_old=q_old, p_old=p_ref, k=k, target=target_distr, x=x, flows=flows)

        target_log_density_f = target_distr.get_logdensity(z=q_upd, x=x) + self.std_normal.log_prob(p_upd).sum(1)
        target_log_density_old = target_distr.get_logdensity(z=q_old, x=x) + self.std_normal.log_prob(p_ref).sum(1)

        log_t = target_log_density_f - target_log_density_old
        log_1_t = torch.logsumexp(torch.cat([torch.zeros_like(log_t).view(-1, 1),
                                             log_t.view(-1, 1)], dim=-1), dim=-1)  # log(1+t)
        if self.use_barker:
            current_log_alphas_pre = log_t - log_1_t
        else:
            current_log_alphas_pre = torch.min(self.device_zero, log_t)

        log_probs = torch.log(self.uniform.sample((q_upd.shape[0],)))
        a = torch.where(log_probs < current_log_alphas_pre, self.device_one, self.device_zero)

        q_new = torch.where((a == self.device_zero)[:, None], q_old, q_upd)
        p_new = torch.where((a == self.device_zero)[:, None], p_ref, p_upd)

        return q_new, p_new, None, None, a, q_upd

    def get_grad(self, q, target, x=None, flows=None):
        q_init = q.detach().requires_grad_(True)
        if flows:
            log_jacobian = 0.
            q_prev = q_init
            q_new = q_init
            for i in range(len(flows)):
                q_new = flows[i](q_prev)
                log_jacobian += flows[i].log_abs_det_jacobian(q_prev, q_new)
                q_prev = q_new
            s = target.get_logdensity(x=x, z=q_new) + log_jacobian
            grad = torch.autograd.grad(s.sum(), q_init)[0]
        else:
            s = target.get_logdensity(x=x, z=q_init)
            grad = torch.autograd.grad(s.sum(), q_init)[0]
        return grad



class Reverse_kernel(nn.Module):
    def __init__(self, kwargs):
        super(Reverse_kernel, self).__init__()
        self.device = kwargs.device
        self.device_one = torch.tensor(1., dtype=kwargs.torchType, device=self.device)
        self.z_dim = kwargs.z_dim
        self.K = kwargs.K
        #self.linear_a = nn.Linear(in_features=self.K, out_features=2*self.K)
        self.linear_z = nn.Linear(in_features=self.z_dim, out_features=5*self.K)
        self.linear_mu = nn.Linear(in_features=self.z_dim, out_features=5*self.K)
        self.linear_hidden = nn.Linear(in_features=10*self.K, out_features=5*self.K)
        self.linear_out = nn.Linear(in_features=5*self.K, out_features=self.K)

    def forward(self, z_fin, mu, a):
        z_ = torch.relu(self.linear_z(z_fin))
        mu_ = torch.relu(self.linear_mu(mu))
        #a_ = torch.relu(self.linear_a(a))
        #cat_z_mu_a = torch.cat([z_, mu_, a_], dim=1)
        cat_z_mu = torch.cat([z_, mu_], dim=1)
        h1 = torch.relu(self.linear_hidden(cat_z_mu))
        probs = torch.sigmoid(self.linear_out(h1))
        probs = torch.where(a == self.device_one, probs, self.device_one-probs)
        log_prob = torch.sum(torch.log(probs), dim=1)
        return log_prob

class Reverse_kernel_sampling(nn.Module):
    def __init__(self, kwargs):
        super(Reverse_kernel_sampling, self).__init__()
        self.device = kwargs.device
        self.device_one = torch.tensor(1., dtype=kwargs.torchType, device=self.device)
        self.z_dim = kwargs.z_dim
        self.K = kwargs.K

        self.linear_z = nn.Linear(in_features=self.z_dim, out_features=5*self.K)
        self.linear_hidden = nn.Linear(in_features=5*self.K, out_features=5*self.K)
        self.linear_out = nn.Linear(in_features=5*self.K, out_features=self.K)

    def forward(self, z_fin, a):
        z_ = torch.relu(self.linear_z(z_fin))
        h1 = torch.relu(self.linear_hidden(z_))
        probs = torch.sigmoid(self.linear_out(h1))
        probs = torch.where(a == self.device_one, probs, self.device_one - probs)
        log_prob = torch.sum(torch.log(probs), dim=1)
        return log_prob

