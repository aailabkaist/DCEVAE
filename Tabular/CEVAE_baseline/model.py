import torch
from torch import nn
import torch.distributions as dists
import torch.nn.functional as F
import math

import random

class CSVAE(nn.Module):
    def __init__(self, r_dim, d_dim, sens_dim, label_dim, args):
        super(CSVAE, self).__init__()
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
        h_dim = args.h_dim

        i_dim = (r_dim + d_dim)

        self.encoder_i_to_a = nn.Sequential(nn.Linear(i_dim, h_dim), act_fn, nn.Linear(h_dim, sens_dim))

        """encoder"""
        self.encoder_i0_to_y = nn.Sequential(nn.Linear(i_dim, h_dim), act_fn, nn.Linear(h_dim, label_dim))
        self.encoder_i1_to_y = nn.Sequential(nn.Linear(i_dim, h_dim), act_fn, nn.Linear(h_dim, label_dim))

        i_dim = r_dim + d_dim + label_dim

        self.encoder_i0_to_u = nn.Sequential(nn.Linear(i_dim, h_dim), act_fn)
        self.mu_i0_to_u = nn.Sequential(nn.Linear(h_dim, u_dim), act_fn)
        self.logvar_i0_to_u = nn.Sequential(nn.Linear(h_dim, u_dim), act_fn)

        self.encoder_i1_to_u = nn.Sequential(nn.Linear(i_dim, h_dim), act_fn)
        self.mu_i1_to_u = nn.Sequential(nn.Linear(h_dim, u_dim), act_fn)
        self.logvar_i1_to_u = nn.Sequential(nn.Linear(h_dim, u_dim), act_fn)

        """decoder"""
        self.decoder_u_to_r = nn.Sequential(nn.Linear(u_dim, h_dim), act_fn, nn.Linear(h_dim, r_dim))
        self.decoder_u_to_d = nn.Sequential(nn.Linear(u_dim, h_dim), act_fn, nn.Linear(h_dim,  d_dim))

        self.p_u0_to_y = nn.Sequential(nn.Linear(u_dim, h_dim), act_fn, nn.Linear(h_dim, label_dim))
        self.p_u1_to_y = nn.Sequential(nn.Linear(u_dim, h_dim), act_fn, nn.Linear(h_dim, label_dim))

        self.p_u_to_a = nn.Sequential(nn.Linear(u_dim, h_dim), act_fn, nn.Linear(h_dim, sens_dim))

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)

    @staticmethod
    def reparameterize(mu, logvar):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std).to(device)
        return eps.mul(std).add_(mu)

    def D(self, z):
        return self.discriminator(z).squeeze()

    def q_u(self, r, d, a, y, test=True):
        i = torch.cat((r, d), 1)
        if test == False:
            qa = self.encoder_i_to_a(i)
            qa_logit = nn.Sigmoid()(qa)
            a = dists.bernoulli.Bernoulli(qa_logit)
            a = a.sample()

            qy_a0 = self.encoder_i0_to_y(i)
            qy_a1 = self.encoder_i1_to_y(i)

            qy = torch.where(a == 1, qy_a1, qy_a0)
            qy_logit = nn.Sigmoid()(qy)

            y = dists.bernoulli.Bernoulli(qy_logit)
            y = y.sample()

        i = torch.cat((r, d, y), 1)

        # q(z|r,d,y)
        intermediate = self.encoder_i0_to_u(i)
        u0_mu = self.mu_i0_to_u(intermediate)
        u0_logvar = self.logvar_i0_to_u(intermediate)

        intermediate = self.encoder_i1_to_u(i)
        u1_mu = self.mu_i1_to_u(intermediate)
        u1_logvar = self.logvar_i1_to_u(intermediate)

        u_mu = torch.where(a == 1, u1_mu, u0_mu)
        u_logvar = torch.where(a == 1, u1_logvar, u0_logvar)
        
        if test == False:
            return u_mu, u_logvar, qa, qy
        else:
            return u_mu, u_logvar

    def p_i(self, u, a, test=True):
        if test==False:
            pa = self.p_u_to_a(u)
            pa_logit = nn.Sigmoid()(pa)
            a = dists.bernoulli.Bernoulli(pa_logit)
            a = a.sample()

        y_p0 = self.p_u0_to_y(u)
        y_p1 = self.p_u1_to_y(u)
        y = torch.where(a == 1, y_p1, y_p0)
        y_cf = torch.where(a == 1, y_p0, y_p1)

        r_mu = self.decoder_u_to_r(u)
        d_mu = self.decoder_u_to_d(u)
        d_mu_cf = d_mu

        return r_mu, d_mu, y, d_mu_cf, y_cf

    def reconstruct(self, u, a):
        r_mu, d_mu, y_p, d_mu_cf, y_p_cf = self.p_i(u, a)

        # r_p = self.reparameterize(r_mu, r_logvar)
        # d_p = self.reparameterize(d_mu, d_logvar)
        # d_p_cf = self.reparameterize(d_mu_cf, d_logvar_cf)
        return r_mu, d_mu, y_p, d_mu_cf, y_p_cf

    def reconstruct_hard(self, u, a):
        r_p, d_p, y_p, d_p_cf, y_p_cf = self.reconstruct(u, a)

        r = nn.Sigmoid()(r_p)
        d = nn.Sigmoid()(d_p)
        d_cf = nn.Sigmoid()(d_p_cf)
        r_hard = dists.bernoulli.Bernoulli(r)
        r_hard = r_hard.sample()
        d_hard = dists.bernoulli.Bernoulli(d)
        d_hard = d_hard.sample()
        d_cf_hard = dists.bernoulli.Bernoulli(d_cf)
        d_cf_hard = d_cf_hard.sample()

        y_p = nn.Sigmoid()(y_p)
        y_p_cf = nn.Sigmoid()(y_p_cf)
        y_hard = dists.bernoulli.Bernoulli(y_p)
        y_hard = y_hard.sample()
        y_cf_hard = dists.bernoulli.Bernoulli(y_p_cf)
        y_cf_hard = y_cf_hard.sample()

        return r_hard, d_hard, y_hard, d_cf_hard, y_cf_hard

    def diagonal(self, M):
        new_M = torch.where(torch.abs(M) < 1e-05, M + 1e-05 * torch.abs(M), M)
        return new_M

    def calculate_loss(self, r, d, a, y, test=True):
        MB = self.args.batch_size

        if test == True:
            u_mu, u_logvar = self.q_u(r, d, a, y, test=test)
            r_mu, d_mu, y_p, d_mu_cf, y_p_cf = self.p_i(u_mu, a)
            u = u_mu
        else:
            u_mu, u_logvar, qa, qy = self.q_u(r, d, a, y, test=test)
            u = self.reparameterize(u_mu, u_logvar)
            r_mu, d_mu, y_p, d_mu_cf, y_p_cf = self.p_i(u, a)

        a_pred = self.p_u_to_a(u)


        "reconstruction"
        if self.args.loss_fn == 'MSE':
            loss_fn = nn.MSELoss(reduction='sum')
        elif self.args.loss_fn == 'BCE':
            loss_fn = nn.BCEWithLogitsLoss(reduction='sum')

        d_recon = loss_fn(d_mu, d)/MB
        r_recon = loss_fn(r_mu, r)/MB
        a_recon = loss_fn(a_pred, a)/MB
        recon = d_recon * self.args.a_d + r_recon * self.args.a_r + a_recon * self.args.a_a
        y_recon = loss_fn(y_p, y)/MB

        if test == False:
            qa_recon = loss_fn(qa, a) / MB
            qy_recon = loss_fn(qy, y) / MB

            aux = qa_recon + qy_recon
            recon += aux

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