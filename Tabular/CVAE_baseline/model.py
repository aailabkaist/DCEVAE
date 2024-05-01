import torch
from torch import nn
import torch.distributions as dists
import torch.nn.functional as F
import math

import random

class CVAE(nn.Module):
    def __init__(self, r_dim, d_dim, sens_dim, label_dim, args):
        super(CVAE, self).__init__()
        '''random seed'''
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        if args.device == 'cuda':
            print("Current CUDA random seed", torch.cuda.initial_seed())
        else:
            print("Current CPU random seed", torch.initial_seed())

        """model structure"""
        self.device = args.device
        self.args = args
        self.r_dim = r_dim
        self.d_dim = d_dim
        self.label_dim = label_dim
        self.sens_dim = sens_dim
        u_dim = args.u_dim
        self.u_dim = u_dim
        if args.act_fn == 'ReLU':
            act_fn = nn.LeakyReLU()
        elif args.act_fn == 'Tanh':
            act_fn = nn.Tanh()

        i_dim = r_dim + d_dim + sens_dim + label_dim

        """encoder"""
        self.encoder_ia_to_u = nn.Sequential(nn.Linear(i_dim, i_dim), act_fn)
        self.mu_ia_to_u = nn.Sequential(nn.Linear(i_dim, u_dim))
        self.logvar_ia_to_u = nn.Sequential(nn.Linear(i_dim, u_dim))

        """decoder"""
        self.decoder_ua_to_r = nn.Sequential(nn.Linear(u_dim + sens_dim, u_dim), act_fn, nn.Linear(u_dim, r_dim))
        self.decoder_ua_to_d = nn.Sequential(nn.Linear(u_dim + sens_dim, u_dim), act_fn, nn.Linear(u_dim, d_dim))
        self.p_ua_to_y = nn.Sequential(nn.Linear(u_dim + sens_dim, u_dim), act_fn, nn.Linear(u_dim, label_dim))

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)

    def rearrange(self, prev, index):
        new = torch.ones_like(prev)
        new[index, :] = prev
        return new

    @staticmethod
    def reparameterize(mu, logvar):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std).to(device)
        return eps.mul(std).add_(mu)

    def q_u(self, r, d, a, y):
        i = torch.cat((r, d, a, y), 1)

        # q(z|r,d,y)
        intermediate = self.encoder_ia_to_u(i)
        u_mu = self.mu_ia_to_u(intermediate)
        u_logvar = self.logvar_ia_to_u(intermediate)

        return u_mu, u_logvar

    def p_i(self, u, a):
        ua = torch.cat([u, a], 1)

        y = self.p_ua_to_y(ua)

        r_p = self.decoder_ua_to_r(ua)

        d_p = self.decoder_ua_to_d(ua)

        return r_p, d_p, y

    def reconstruct(self, u, a):
        r_p, d_p, y_p = self.p_i(u, a)
        a_cf = torch.where(a == 0, torch.ones_like(a), torch.zeros_like(a))
        _, _, y_p_cf = self.p_i(u, a_cf)

        return r_p, d_p, y_p, y_p_cf

    def reconstruct_hard(self, u, a):
        r_p, d_p, y_p, _ = self.reconstruct(u, a)

        r = nn.Sigmoid()(r_p)
        d = nn.Sigmoid()(d_p)
        y_p = nn.Sigmoid()(y_p)

        r_hard = dists.bernoulli.Bernoulli(r)
        r_hard = r_hard.sample()
        d_hard = dists.bernoulli.Bernoulli(d)
        d_hard = d_hard.sample()
        y_hard = dists.bernoulli.Bernoulli(y_p)
        y_hard = y_hard.sample()

        return r_hard, d_hard, y_hard

    def diagonal(self, M):
        new_M = torch.where(torch.abs(M) < 1e-05, M + 1e-05 * torch.abs(M), M)
        return new_M


    def calculate_loss(self, r, d, a, y, test=True):
        MB = self.args.batch_size
        a_cf = torch.where(a == 1, torch.zeros_like(a), torch.ones_like(a))

        u_mu, u_logvar = self.q_u(r, d, a, y)
        u = self.reparameterize(u_mu, u_logvar)
        r_mu, d_mu, y_p = self.p_i(u, a)

        _, _, y_p_cf = self.p_i(u, a_cf)

        "reconstruction"
        if self.args.loss_fn == 'MSE':
            loss_fn = nn.MSELoss(reduction='sum')
        elif self.args.loss_fn == 'BCE':
            loss_fn = nn.BCEWithLogitsLoss(reduction='sum')

        d_recon = loss_fn(d_mu, d)/MB
        r_recon = loss_fn(r_mu, r)/MB
        recon = self.args.a_d * d_recon + self.args.a_r * r_recon
        y_recon = loss_fn(y_p, y)/MB

        """KL loss"""
        #Prohibiting cholesky error
        u_logvar = self.diagonal(u_logvar)

        assert (torch.sum(torch.isnan(u_logvar)) == 0), 'u_logvar'

        u_dist = dists.MultivariateNormal(u_mu.flatten(), torch.diag(u_logvar.flatten().exp()))
        u_prior = dists.MultivariateNormal(torch.zeros(self.u_dim * u_mu.size()[0]).to(self.device),\
                                           torch.eye(self.u_dim * u_mu.size()[0]).to(self.device))
        u_kl = dists.kl.kl_divergence(u_dist, u_prior)/MB

        """fair loss"""
        y_cf_sig = nn.Sigmoid()(y_p_cf)
        y_p_sig = nn.Sigmoid()(y_p)
        fair_l = torch.sum(torch.norm(y_cf_sig - y_p_sig, p=2, dim=1))/MB

        assert (torch.sum(torch.isnan(recon)) == 0), 'x_recon'
        assert (torch.sum(torch.isnan(y_recon)) == 0), 'y_recon'
        assert (torch.sum(torch.isnan(u_kl)) == 0), 'u_kl'

        ELBO = recon + self.args.a_y * y_recon + self.args.u_kl * u_kl + self.args.a_f * fair_l

        assert (torch.sum(torch.isnan(ELBO)) == 0), 'ELBO'

        return ELBO, recon, y_recon, y_p, y_p_cf, u_kl, fair_l