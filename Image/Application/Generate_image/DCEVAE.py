import torch
from torch import nn
from block import Reshape, Flatten, Conv_block
import torch.distributions as dists
import torch.nn.functional as F

class DCEVAE(nn.Module):
    def __init__(self, args, sens_dim, rest_dim, des_dim, ur_dim, ud_dim, KOF=64, p=0.04, batch_size=64):
        super(DCEVAE, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.int = args.int

        self.sens_dim = sens_dim
        self.rest_dim = rest_dim
        self.des_dim = des_dim

        self.ur_dim = ur_dim
        self.ud_dim = ud_dim

        u_dim = ur_dim + ud_dim
        self.u_dim = u_dim

        self.batch_size = batch_size

        """Encoder Network"""
        # xa to ur
        KOF = 32
        ka_size = int(KOF / 2)
        kx_size = KOF - ka_size

        self.encoder_x_to_kx = nn.Sequential()
        self.encoder_x_to_kx.add_module("block01", Conv_block(kx_size, 3, kx_size, 4, 2, 1, p=p))
        # self.encoder_x_to_kx.add_module("block02", Conv_block(kx_size, kx_size, kx_size, 4, 2, 1, p=p))

        self.encoder_a_to_ka = nn.Sequential()
        # Bx2 -> Bx2x1x1
        self.encoder_a_to_ka.add_module("reshape", Reshape((-1, 2, 1, 1)))
        # Bx2x1x1 -> Bxkax4x4
        self.encoder_a_to_ka.add_module("block00", Conv_block(ka_size, 2, ka_size, 32, 1, 0, p=p, transpose=True))

        self.encoder = nn.Sequential()
        # self.encoder.add_module("block01", Conv_block(KOF, 3, KOF, 4, 2, 1, p=p))
        self.encoder.add_module("block02", Conv_block(KOF, KOF, KOF, 4, 2, 1, p=p))
        self.encoder.add_module("block03", Conv_block(KOF * 2, KOF, KOF * 2, 4, 2, 1, p=p))
        self.encoder.add_module("block04", Conv_block(KOF * 2, KOF * 2, KOF * 2, 4, 2, 1, p=p))
        self.encoder.add_module("flatten", Flatten())
        self.encoder.add_module("FC01", nn.Linear(KOF * 32, 256))
        self.encoder.add_module("ReLU", nn.ReLU())
        self.encoder.add_module("FC02", nn.Linear(256, 2 * u_dim))

        """Classifier"""
        self.decoder_ur_to_rest = nn.Sequential(
            nn.Linear(ur_dim, ur_dim),
            nn.ReLU(),
            nn.Linear(ur_dim, rest_dim)
        )

        self.decoder_aud_to_des = nn.Sequential(
            nn.Linear(sens_dim * 2 + ud_dim, sens_dim * 2 + ud_dim),
            nn.ReLU(),
            nn.Linear(sens_dim * 2 + ud_dim, des_dim)
        )

        """Decoder Network"""
        ku_size = KOF  # int(KOF * 2 * (u2_dim)/u1_dim)
        ka_size = 2 * KOF - ku_size  # 2 * KOF - ku_size

        self.decoder_a_to_ka = nn.Sequential()
        # Bx2 -> Bx2x1x1
        self.decoder_a_to_ka.add_module("reshape", Reshape((-1, 2, 1, 1)))
        # Bx2x1x1 -> Bxkax4x4
        self.decoder_a_to_ka.add_module("block00", Conv_block(ka_size, 2, ka_size, 4, 1, 0, p=p, transpose=True))

        self.decoder_urud_to_ku = nn.Sequential()
        self.decoder_urud_to_ku.add_module("block00", nn.Sequential(
            nn.Linear(u_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2)
        ))  # Bx(ur+ud) -> Bxku
        self.decoder_urud_to_ku.add_module("reshape", Reshape((-1, 256, 1, 1)))  # Bxkux1X1
        self.decoder_urud_to_ku.add_module("block01", \
                                           Conv_block(ku_size, 256, ku_size, 4, 1, 0, p=p,
                                                      transpose=True))  # Bxkux4X4

        self.decoder_u_to_x = nn.Sequential()
        self.decoder_u_to_x.add_module("block01", Conv_block(KOF * 2, KOF * 2, KOF * 2, 4, 2, 1, p=p, transpose=True))
        self.decoder_u_to_x.add_module("block02", Conv_block(KOF, KOF * 2, KOF, 4, 2, 1, p=p, transpose=True))
        self.decoder_u_to_x.add_module("block03", Conv_block(KOF, KOF, KOF, 4, 2, 1, p=p, transpose=True))
        self.decoder_u_to_x.add_module("block04", Conv_block(3, KOF, 3, 4, 2, 1, p=p, transpose=True))

        """Discriminator Network"""
        self.discriminator = nn.Sequential(
            nn.Linear(u_dim + sens_dim * 2, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 2),
        )
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)

    def D(self, z):
        return self.discriminator(z).squeeze()

    def q_u(self, x, a):
        """
        VARIATIONAL POSTERIOR
        :param x: input image
        :return: parameters of q(z|x), (MB, hid_dim)
        """
        intervention = torch.cat([1 - a, a], 1)
        ka = self.encoder_a_to_ka(intervention)
        kx = self.encoder_x_to_kx(x)
        xa = torch.cat([kx, ka], 1)
        stats = self.encoder(xa)
        u_mu = stats[:, :self.u_dim]
        u_logvar = stats[:, self.u_dim:]

        return u_mu, u_logvar

    def p_x(self, u, a):
        """
        GENERATIVE DISTRIBUTION
        :param z: latent vector          (MB, hid_dim)
        :return: parametersc of p(x|z)    (MB, inp_dim)
        """

        intervention = torch.cat([1 - a, a], 1)
        ku = self.decoder_urud_to_ku(u)
        ka = self.decoder_a_to_ka(intervention)
        u = torch.cat([ku, ka], 1)
        x_hat = self.decoder_u_to_x(u)

        intervention_cf = 1 - intervention
        ka_cf = self.decoder_a_to_ka(intervention_cf)
        u = torch.cat([ku, ka_cf], 1)
        x_cf_hat = self.decoder_u_to_x(u)

        return x_hat, x_cf_hat

    def classifier(self, a, u):
        """classifier"""
        ur, ud = torch.split(u, [self.ur_dim, self.ud_dim], 1)

        rest_ur = self.decoder_ur_to_rest(ur)

        if self.int == 'M':
            return rest_ur  # des_ud
        elif self.int == 'S':
            a = torch.cat([1-a, a], 1)
            aud = torch.cat([a, ud], 1)
            des_ud = self.decoder_aud_to_des(aud)
            return rest_ur, des_ud

    def forward(self, x, a):
        """
        Encode the image, sample z and decode
        :param x: input image
        :return: parameters of p(x|z_hat), z_hat, parameters of q(z|x)
        """
        u_mu, u_logvar = self.q_u(x, a)
        u = self.reparameterize(u_mu, u_logvar)

        x_hat, x_cf_hat = self.p_x(u, a)
        # x_rec = self.reparameterize(x_mu, x_logvar)
        # x_cf = self.reparameterize(x_mu_cf, x_logvar_cf)

        return x_hat, x_cf_hat, u_mu, u_logvar, u

    def sampling_intervention(self, a):
        num = a.shape[0]
        urud_mu = torch.zeros(num, self.ur_dim + self.ud_dim).to(self.device)
        urud_logvar = torch.ones(num, self.ur_dim + self.ud_dim).to(self.device)
        urud = self.reparameterize(urud_mu, urud_logvar)

        ku = self.decoder_urud_to_ku(urud)
        intervention = torch.cat([1 - a, a], 1)

        ka = self.decoder_a_to_ka(intervention)
        u = torch.cat([ku, ka], 1)

        x_hat = self.decoder_u_to_x(u)
        x_hat = nn.Sigmoid()(x_hat)
        return x_hat

    def sampling_counterfactual(self, x, a):
        u_mu, u_logvar = self.q_u(x, a)
        u = self.reparameterize(u_mu, u_logvar)

        x_hat, x_cf_hat = self.p_x(u, a)
        x_cf_hat = nn.Sigmoid()(x_cf_hat)
        return x_cf_hat

    # def sampling_conditional(self, x):
    #     num = x.shape[0]
    #     u_mu = torch.zeros(num, self.u1_dim).to(self.device)
    #     u_logvar = torch.ones(num, self.u1_dim).to(self.device)
    #     u = self.reparameterize(u_mu, u_logvar)
    #
    #     ur, ud, ua = torch.split(u, [self.ur_dim, self.ud_dim, self.ua_dim], 1)
    #     a_pred = self.decoder_ua_to_a(ua)
    #     a_pred = nn.Sigmoid()(a_pred)
    #     a_pred = torch.where(a_pred > 1, torch.ones_like(a_pred), torch.zeros_like(a_pred))
    #
    #     x_hat, _ = self.p_x(u, a_pred)
    #     x_hat = nn.Sigmoid()(x_hat)
    #     return x_hat, a_pred

    def cov(self, x, a):
        _, u_logvar = self.q_u(x, a)
        u_logvar.unsqueeze_(2)
        u_logvar_T = u_logvar.permute(0, 2, 1)
        cov = torch.matmul(u_logvar, u_logvar_T)
        return cov

    def reconstruct_x(self, x, a):
        x_hat, _, _, _, _ = self.forward(x, a)
        x_hat = nn.Sigmoid()(x_hat)
        return x_hat

    def diagonal(self, M):
        """
        If diagonal value is close to 0, it could makes cholesky decomposition error.
        To prevent this, I add some diagonal value which won't affect the original value too much.
        """
        new_M = torch.where(torch.isnan(M), (torch.ones_like(M) * 1e-02).to(self.device), M)
        new_M = torch.where(torch.abs(new_M) < 1e-02, (torch.ones_like(M) * 1e-02).to(self.device), new_M)

        return new_M

    def negentropy(self, a_pred):
        a_pred = nn.Sigmoid()(a_pred)
        a_pred_negentropy = ((a_pred + 1e-5).log() * a_pred + (1 - a_pred + 1e-5).log() * (1 - a_pred)).mean()
        return a_pred_negentropy

    def permute_dims(self, ua):
        assert ua.dim() == 2

        B, _ = ua.size()
        perm_ua = []
        idx = 0
        for u_j in torch.split(ua, [self.ur_dim, self.ud_dim, self.sens_dim * 2], 1):
            perm = torch.randperm(B).to(self.device)
            if idx == 0:
                perm_ur = perm
            elif idx == 2:
                perm = perm_ur
            perm_u_j = u_j[perm]
            perm_ua.append(perm_u_j)
            idx += 1

        return torch.cat(perm_ua, 1)

    def image(self, x, sens):
        x_fc, x_cf, u_mu, u_logvar, u = self.forward(x, sens)
        #         x_fc = self.reparameterize(x_mu, x_logvar)
        #         x_cf = self.reparameterize(x_mu_cf, x_logvar_cf)
        x_fc = nn.Sigmoid()(x_fc)
        x_cf = nn.Sigmoid()(x_cf)
        return x_fc, x_cf

    def calculate_loss(self, x, sens, rest, des, x2, sens2, beta1=1, beta2=1, beta3=1, beta4=1):
        """
        Given the input batch, compute the negative ELBO
        :param x:   (MB, inp_dim)
        :param beta: Float
        :param average: Compute average over mini batch or not, bool
        :return: -RE + beta * KL  (MB, ) or (1, )
        """

        # divide a
        MB = x.shape[0]

        x_p, x_cf_p, u_mu, u_logvar, u = self.forward(x, sens)

        #         x_logvar = self.diagonal(x_logvar)
        u_logvar = self.diagonal(u_logvar)

        #         assert (torch.sum(torch.isnan(x_logvar)) == 0), 'x_logvar'
        assert (torch.sum(torch.isnan(u_logvar)) == 0), 'u_logvar'

        assert (torch.sum(torch.isnan(x_p)) == 0), 'x_p'
        assert (torch.sum(torch.isnan(x_cf_p)) == 0), 'x_cf_p'

        """x_reconstruction_loss"""
        x_recon = nn.BCEWithLogitsLoss(reduction='sum')(x_p, x) / MB
        x_cf_recon = nn.BCEWithLogitsLoss(reduction='sum')(x_cf_p, x) / MB
        assert (torch.sum(torch.isnan(x_recon)) == 0), 'x_recon'
        assert (torch.sum(torch.isnan(x_cf_recon)) == 0), 'x_cf_recon'

        """kl loss"""
        u_dist = dists.MultivariateNormal(u_mu.flatten(), torch.diag(u_logvar.flatten().exp()))
        u_prior = dists.MultivariateNormal(torch.zeros(self.u_dim * u_mu.size()[0]).to(u_mu),
                                           torch.eye(self.u_dim * u_mu.size()[0]).to(u_mu))
        u_kl = dists.kl.kl_divergence(u_dist, u_prior) / MB


        """Classifier loss"""
        if self.int == 'M':
            rest_ur = self.classifier(sens, u)
            recon_rest_ur = nn.BCEWithLogitsLoss(reduction='sum')(rest_ur, rest) / MB
            l_recon = recon_rest_ur
        elif self.int == 'S':
            rest_ur, des_ud = self.classifier(sens, u)
            recon_rest_ur = nn.BCEWithLogitsLoss(reduction='sum')(rest_ur, rest) / MB
            recon_des_ud = nn.BCEWithLogitsLoss(reduction='sum')(des_ud, des) / MB
            l_recon = recon_rest_ur + recon_des_ud

        """TC Loss"""
        ones = torch.ones(u.shape[0], dtype=torch.long, device=self.device)
        zeros = torch.zeros(u.shape[0], dtype=torch.long, device=self.device)

        ua = torch.cat([u, 1-sens, sens], 1)
        D_u = self.D(ua)
        vae_tc_loss = (D_u[:, :1] - D_u[:, 1:]).mean()

        _, _, _, _, u2 = self.forward(x2, sens2)
        ua2 = torch.cat([u2, 1 - sens2, sens2], 1)
        u_perm = self.permute_dims(ua2).detach()
        D_u_perm = self.D(u_perm)
        D_tc_loss = 0.5 * (F.cross_entropy(D_u, zeros) + F.cross_entropy(D_u_perm, ones))

        ELBO = beta1 * x_recon + l_recon * beta2 + beta3 * u_kl + beta4 * vae_tc_loss

        return ELBO, x_recon, recon_rest_ur, u_kl, vae_tc_loss, D_tc_loss, x_cf_recon

    @staticmethod
    def reparameterize(mu, logvar):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(device)
        return eps.mul(std).add_(mu)