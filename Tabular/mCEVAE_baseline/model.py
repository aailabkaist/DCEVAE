import torch
from torch import nn
import torch.distributions as dists
from utils import compute_mmd

class CEVAE(nn.Module):
    def __init__(self, x_dim, o_dim, sens_dim, label_dim, args):
        super(CEVAE, self).__init__()
        '''random seed'''
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        if args.device == 'cuda':
            print("Current CUDA random seed", torch.cuda.initial_seed())
        else:
            print("Current CPU random seed", torch.initial_seed())

        '''model structure'''

        self.device = args.device
        self.args = args
        self.x_dim = x_dim
        self.o_dim = o_dim
        self.label_dim = label_dim
        self.sens_dim = sens_dim
        u_dim = args.u_dim
        self.u_dim = u_dim
        if args.act_fn == 'ReLU':
            act_fn = nn.LeakyReLU()
        elif args.act_fn == 'Tanh':
            act_fn = nn.Tanh()

        i_dim = o_dim + x_dim #input_dim

        #dropout = nn.Dropout(p=args.dp_ratio)

        self.encoder_ox0_to_u = nn.Sequential(nn.Linear(i_dim, i_dim), nn.BatchNorm1d(i_dim), act_fn, \
                                              nn.Linear(i_dim, i_dim), nn.BatchNorm1d(i_dim), act_fn)
        self.mu_ox0_to_u = nn.Linear(i_dim, u_dim)
        self.logvar_ox0_to_u = nn.Linear(i_dim, u_dim)

        self.encoder_ox1_to_u = nn.Sequential(nn.Linear(i_dim, i_dim), nn.BatchNorm1d(i_dim), act_fn,\
                                             nn.Linear(i_dim, i_dim), nn.BatchNorm1d(i_dim), act_fn)
        self.mu_ox1_to_u = nn.Linear(i_dim, u_dim)
        self.logvar_ox1_to_u = nn.Linear(i_dim, u_dim)

        self.decoder_au_to_ox = nn.Sequential(nn.Linear(u_dim + sens_dim, u_dim + sens_dim), nn.BatchNorm1d(u_dim + sens_dim), \
                                             act_fn, nn.Linear(u_dim + sens_dim, i_dim), nn.BatchNorm1d(i_dim), act_fn)
        self.mu_au_to_x = nn.Sequential(nn.Linear(i_dim, i_dim), act_fn, nn.Linear(i_dim, x_dim))
        self.logvar_au_to_x = nn.Linear(i_dim, x_dim)

        self.mu_au_to_o = nn.Sequential(nn.Linear(i_dim, i_dim), act_fn, nn.Linear(i_dim, o_dim))
        self.logvar_au_to_o = nn.Linear(i_dim, o_dim)

        self.p_au_to_y = nn.Sequential(nn.Linear(u_dim + sens_dim, u_dim+sens_dim), nn.BatchNorm1d(u_dim+sens_dim), \
                                       act_fn, nn.Linear(u_dim+sens_dim, label_dim))

        self.decoder_u_to_a = nn.Sequential(nn.Linear(u_dim, u_dim), nn.BatchNorm1d(u_dim), act_fn,\
                                            nn.Linear(u_dim, sens_dim))

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)

    def q_u(self, o, a, x):

        # divide a
        a0_index = (a == 0).nonzero()[:, 0].to(self.device)
        a1_index = (a == 1).nonzero()[:, 0].to(self.device)
        a_index = torch.cat((a0_index, a1_index), 0).to(self.device)

        ''' q(u|o,a)'''
        ox = torch.cat((o, x), 1)
        # q(u|o,x,a=0)
        ox0 = ox[a0_index, :]
        intermediate = self.encoder_ox0_to_u(ox0)
        u0_mu_encoder = self.mu_ox0_to_u(intermediate)
        u0_logvar_encoder = self.logvar_ox0_to_u(intermediate)

        # q(u|o,x,a=1)
        ox1 = ox[a1_index, :]
        intermediate = self.encoder_ox1_to_u(ox1)
        u1_mu_encoder = self.mu_ox1_to_u(intermediate)
        u1_logvar_encoder = self.logvar_ox1_to_u(intermediate)

        mu_encoder = torch.cat((u0_mu_encoder, u1_mu_encoder), 0)
        logvar_encoder = torch.cat((u0_logvar_encoder, u1_logvar_encoder), 0)
        u_mu_encoder = self.rearrange(mu_encoder, a_index)
        u_logvar_encoder = self.rearrange(logvar_encoder, a_index)

        u0 = self.reparameterize(u0_mu_encoder, u0_logvar_encoder)
        u1 = self.reparameterize(u1_mu_encoder, u1_logvar_encoder)

        return u_mu_encoder, u_logvar_encoder, u0, u1

    def rearrange(self, prev, index):
        new = torch.ones_like(prev)
        new[index, :] = prev
        return new

    def p_oxy(self, a, u):
        au = torch.cat((a, u), 1)

        intermediate = self.decoder_au_to_ox(au)

        x_mu = self.mu_au_to_x(intermediate)
        x_logvar = self.logvar_au_to_x(intermediate)

        o_mu = self.mu_au_to_o(intermediate)
        o_logvar = self.logvar_au_to_o(intermediate)

        y_p = self.p_au_to_y(au)

        return o_mu, o_logvar, x_mu, x_logvar, y_p

    def forward(self, o, a, x):
        # z, s, u posterior
        u_mu_encoder, u_logvar_encoder, u0, u1 = self.q_u(o, a, x)

        u = self.reparameterize(u_mu_encoder, u_logvar_encoder)

        return u_mu_encoder, u_logvar_encoder, u, u0, u1

    def reconstruct(self, a, u):
        o_mu, o_logvar, x_mu, x_logvar, y_p = self.p_oxy(a, u)
        a_cf = torch.where(a == 1, torch.zeros_like(a), torch.ones_like(a))
        o_counter_mu, o_counter_logvar, x_counter_mu, x_counter_logvar, y_counter_p = self.p_oxy(a_cf, u)
        a_pred_from_u = self.decoder_u_to_a(u)

        return o_mu, x_mu, y_p, y_counter_p, a_pred_from_u

    def predict(self, a, u):
        _, _, _, _, y_p = self.p_oxy(a, u)
        a_cf = torch.where(a == 1, torch.zeros_like(a), torch.ones_like(a))
        _, _, _, _, y_counter_p = self.p_oxy(a_cf, u)
        y_p = nn.Sigmoid()(y_p)
        y_cf_p = nn.Sigmoid()(y_counter_p)
        return y_p, y_cf_p

    def reconstruct_hard(self, a, u):
        o_mu, o_logvar, x_mu, x_logvar, y_p = self.p_oxy(a, u)
        a_cf = torch.where(a == 1, torch.zeros_like(a), torch.ones_like(a))
        o_cf_mu, o_cf_logvar, x_cf_mu, x_cf_logvar, y_counter_p = self.p_oxy(a_cf, u)

        x = x_mu
        x_cf = x_cf_mu

        o = o_mu
        o_cf = o_cf_mu

        # you should modify this code
        x = nn.Sigmoid()(x)
        x_cf = nn.Sigmoid()(x_cf)
        x_hard = dists.bernoulli.Bernoulli(x)
        x_hard = x_hard.sample().type(torch.float64)
        x_cf_hard = dists.bernoulli.Bernoulli(x_cf)
        x_cf_hard = x_cf_hard.sample().type(torch.float64)

        o = nn.Sigmoid()(o)
        o_cf = nn.Sigmoid()(o_cf)
        o_hard = dists.bernoulli.Bernoulli(o)
        o_hard = o_hard.sample().type(torch.float64)
        #o_cf_hard = (o_cf >= 0.5).type(torch.float64)
        o_cf_hard = dists.bernoulli.Bernoulli(o_cf).sample().type(torch.float64)

        y_p = nn.Sigmoid()(y_p)
        y_counter_p = nn.Sigmoid()(y_counter_p)
        # y_hard = (y_p >= 0.5).type(torch.float64)
        # y_cf_hard = (y_counter_p >= 0.5).type(torch.float64)
        y_hard = dists.bernoulli.Bernoulli(y_p)
        y_hard = y_hard.sample().type(torch.float64)
        y_cf_hard = dists.bernoulli.Bernoulli(y_counter_p)
        y_cf_hard = y_cf_hard.sample().type(torch.float64)

        return o_hard, o_cf_hard, x_hard, x_cf_hard, y_hard, y_cf_hard

    def te_sampling(self, num):

        u_mu = torch.zeros(num, self.u_dim).to(self.device)
        u_logvar = torch.ones(num, self.u_dim).to(self.device)
        u = self.reparameterize(u_mu, u_logvar)

        a0 = torch.zeros(num, 1).to(self.device)

        o_hard, o_cf_hard, x_hard, x_cf_hard, y_hard, y_cf_hard = self.reconstruct_hard(a0, u)

        cf_effect = y_cf_hard - y_hard

        mask1 = torch.all(torch.eq(o_hard, torch.Tensor([0, 0]).to(self.device)), 1)
        mask2 = torch.all(torch.eq(o_hard, torch.Tensor([0, 1]).to(self.device)), 1)
        mask3 = torch.all(torch.eq(o_hard, torch.Tensor([1, 0]).to(self.device)), 1)
        mask4 = torch.all(torch.eq(o_hard, torch.Tensor([1, 1]).to(self.device)), 1)

        o1 = cf_effect[mask1 == True]
        o2 = cf_effect[mask2 == True]
        o3 = cf_effect[mask3 == True]
        o4 = cf_effect[mask4 == True]

        y_a0 = torch.mean(y_hard)
        y_a1 = torch.mean(y_cf_hard)
        o1 = torch.mean(o1)
        o2 = torch.mean(o2)
        o3 = torch.mean(o3)
        o4 = torch.mean(o4)

        return y_a0, y_a1, y_a1 - y_a0, o1, o2, o3, o4

    def diagonal(self, M):
        new_M = torch.where(torch.abs(M) < 1e-05, M + 1e-05 * torch.abs(M), M)
        return new_M

    def calculate_loss(self, x, o, a, y):

        MB = self.args.batch_size

        u_mu_encoder, u_logvar_encoder, u, u0, u1 = self.forward(o, a, x)

        o_p, x_p, y_p, y_counter_p, a_pred_from_u = self.reconstruct(a, u)


        if self.args.loss_fn == 'MSE':
            loss_fn = nn.MSELoss(reduction='sum')
        elif self.args.loss_fn == 'BCE':
            loss_fn = nn.BCEWithLogitsLoss(reduction='sum')

        recon = loss_fn(x_p, x)
        recon += loss_fn(o_p, o)
        recon = recon/MB
        y_recon = loss_fn(y_p, y)/MB

        '''Prohibiting cholesky error'''
        u_logvar_encoder = self.diagonal(u_logvar_encoder)
        assert (torch.sum(torch.isnan(u_logvar_encoder)) == 0), 'u_logvar'

        u_sample = torch.rand(u_mu_encoder.shape).to(self.device)
        u0_sample = torch.rand(u0.shape).to(self.device)
        u1_sample = torch.rand(u1.shape).to(self.device)
        mmd = compute_mmd(u_sample, u)
        mmd_A0 = compute_mmd(u0_sample, u0) * u0.shape[0]
        mmd_A1 = compute_mmd(u1_sample, u1) * u1.shape[0]
        mmd_a = mmd_A0 + mmd_A1

        y_cf_sig = nn.Sigmoid()(y_counter_p)
        y_p_sig = nn.Sigmoid()(y_p)
        fair_l = torch.sum(torch.norm(y_cf_sig - y_p_sig, p=2, dim=1))/MB

        assert (torch.sum(torch.isnan(recon)) == 0), 'x_recon'
        assert (torch.sum(torch.isnan(y_recon)) == 0), 'y_recon'

        ELBO = self.args.a_x * recon + self.args.a_y * y_recon + self.args.mmd * mmd + self.args.mmd_a * mmd_a + self.args.a_f * fair_l

        assert (torch.sum(torch.isnan(ELBO)) == 0), 'ELBO'

        return ELBO, recon, y_recon, y_p, y_counter_p, mmd, mmd_a, fair_l

    @staticmethod
    def reparameterize(mu, logvar):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(device)
        return eps.mul(std).add_(mu)
