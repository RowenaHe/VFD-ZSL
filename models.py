from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable

class VAE(nn.Module):
    # 这里的改动仅为 create-Encoder、create-Decoder 中 input 和 output shape 从args.X_dim 变为 args.S_dim
    def __init__(self, args, gen_size = None):
        super(VAE, self).__init__()
        self.z_size = args.Z_dim
        if gen_size is not None:
            self.input_size = gen_size
        elif args.gen_size is not None:
            self.input_size = args.gen_size
        else:
            self.input_size = args.S_dim
        self.args = args
        self.q_z_nn_output_dim = args.q_z_nn_output_dim #128, 隐藏层输出 dim
        self.q_z_nn, self.q_z_mean, self.q_z_var = self.create_encoder()
        self.p_x_nn, self.p_x_mean = self.create_decoder()
        self.FloatTensor = torch.FloatTensor

    def create_encoder(self):
        q_z_nn = nn.Sequential(
            nn.Linear(self.input_size + self.args.C_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1024),
            nn.Dropout(self.args.vae_enc_drop),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, self.q_z_nn_output_dim)
        )
        # 输入图像 dim + emb dim -->1024 -->1024 
        q_z_mean = nn.Linear(self.q_z_nn_output_dim, self.z_size)

        q_z_var = nn.Sequential(
            nn.Linear(self.q_z_nn_output_dim, self.z_size),
            nn.Dropout(0.2),
            nn.Softplus(),
        )
        return q_z_nn, q_z_mean, q_z_var


    def create_decoder(self):
        p_x_nn = nn.Sequential(
            nn.Linear(self.z_size + self.args.C_dim, 1024),
            nn.Dropout(self.args.vae_dec_drop),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.Dropout(self.args.vae_dec_drop),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.Dropout(self.args.vae_dec_drop),
            nn.LeakyReLU(0.2, inplace=True)
        )
        p_x_mean = nn.Sequential(
            nn.Linear(1024, self.input_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        return p_x_nn, p_x_mean


    def reparameterize(self, mu, var):
        std = var.sqrt()
        eps = self.FloatTensor(std.size()).normal_().to(self.args.gpu)
        eps = Variable(eps)
        z = eps.mul(std).add_(mu)
        return z

    def encode(self, x, c):
        input = torch.cat((x,c),1)
        h = self.q_z_nn(input)
        h = h.view(h.size(0), -1)
        mean = self.q_z_mean(h)
        var = self.q_z_var(h)
        return mean, var

    def decode(self, z, c):
        input = torch.cat((z, c), 1)
        h = self.p_x_nn(input)
        x_mean = self.p_x_mean(h)
        return x_mean

    def forward(self, x, c, weights=None):
        z_mu, z_var = self.encode(x, c)
        z = self.reparameterize(z_mu, z_var) # 根据均值和方差，采样 normal noise，得到 latent z
        x_mean = self.decode(z, c)
        return x_mean, z_mu, z_var, z


class alter_VAE(nn.Module):
    # 这里的改动仅为 create-Encoder、create-Decoder 中 input 和 output shape 从args.X_dim 变为 args.S_dim
    def __init__(self, args, gen_size = None):
        super(alter_VAE, self).__init__()
        self.z_size = args.Z_dim
        if gen_size is not None:
            self.input_size = gen_size
        elif args.gen_size is not None:
            self.input_size = args.gen_size
        else:
            self.input_size = args.S_dim
        self.args = args
        self.q_z_nn_output_dim = args.q_z_nn_output_dim
        self.q_z_nn, self.q_z_mean, self.q_z_var = self.create_encoder()
        self.p_x_nn, self.p_x_mean = self.create_decoder()
        self.FloatTensor = torch.FloatTensor

    def create_encoder(self):
        q_z_nn = nn.Sequential(
            nn.Linear(self.input_size + self.args.C_dim, 1536),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1536, 1536),
            nn.Dropout(self.args.vae_enc_drop),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1536, self.q_z_nn_output_dim)
        )
        q_z_mean = nn.Linear(self.q_z_nn_output_dim, self.z_size)

        q_z_var = nn.Sequential(
            nn.Linear(self.q_z_nn_output_dim, self.z_size),
            nn.Dropout(0.2),
            nn.Softplus(),
        )
        return q_z_nn, q_z_mean, q_z_var


    def create_decoder(self):
        p_x_nn = nn.Sequential(
            nn.Linear(self.z_size + self.args.C_dim, 1536),
            nn.Dropout(self.args.vae_dec_drop),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1536, 1536),
            nn.BatchNorm1d(1536, 0.8),
            nn.Dropout(self.args.vae_dec_drop),
            nn.Linear(1536, 1536),
            nn.BatchNorm1d(1536, 0.8),
            nn.Dropout(self.args.vae_dec_drop),
            nn.LeakyReLU(0.2, inplace=True)
        )
        p_x_mean = nn.Sequential(
            nn.Linear(1536, self.input_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        return p_x_nn, p_x_mean


    def reparameterize(self, mu, var):
        std = var.sqrt()
        eps = self.FloatTensor(std.size()).normal_().to(self.args.gpu)
        eps = Variable(eps)
        z = eps.mul(std).add_(mu)
        return z

    def encode(self, x, c):
        input = torch.cat((x,c),1)
        h = self.q_z_nn(input)
        h = h.view(h.size(0), -1)
        mean = self.q_z_mean(h)
        var = self.q_z_var(h)
        return mean, var

    def decode(self, z, c):
        input = torch.cat((z, c), 1)
        h = self.p_x_nn(input)
        x_mean = self.p_x_mean(h)
        return x_mean

    def forward(self, x, c, weights=None):
        z_mu, z_var = self.encode(x, c)
        z = self.reparameterize(z_mu, z_var) # 根据均值和方差，采样 normal noise，得到 latent z
        x_mean = self.decode(z, c)
        return x_mean, z_mu, z_var, z

class Classifier(nn.Module):
    def __init__(self, S_dim, dataset):
        super(Classifier, self).__init__()
        self.cls = nn.Linear(S_dim, dataset.ntrain_class) #FLO 82
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, s):
        return self.logic(self.cls(s))

class R_AE(nn.Module):
    def __init__(self, args):
        super(R_AE, self).__init__()
        self.args = args
        self.encoder = nn.Sequential(
            nn.Linear(args.X_dim, args.S_dim + args.D_dim + args.R_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(args.ae_drop)
        )
        self.decoder = nn.Sequential(
            nn.Linear(args.S_dim + args.D_dim + args.R_dim, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(args.ae_drop),
            nn.Linear(2048, args.X_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(args.ae_drop),
        )

    def forward(self, x):
        z = self.encoder(x)
        s = z[:, :self.args.S_dim]#这部分是 semantic的，待保存
        d = z[:, self.args.S_dim:self.args.S_dim+self.args.D_dim]
        r = z[:, self.args.S_dim+self.args.D_dim:self.args.S_dim+self.args.D_dim+self.args.R_dim]
        x1 = self.decoder(z)
        return x1, z, s, d, r


class R_AE_new(nn.Module):
    def __init__(self, args):
        super(R_AE_new, self).__init__()
        self.args = args
        self.encoder = nn.Sequential(
            nn.Linear(args.X_dim, args.S_dim + args.D_dim + args.R_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(args.ae_drop)
        )
        self.S = nn.Sequential(
            nn.Linear(args.S_dim + args.D_dim, args.S_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(args.ae_drop)
        )

        self.D = nn.Sequential(
            nn.Linear(args.S_dim + args.D_dim, args.D_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(args.ae_drop)
        )
        self.decoder = nn.Sequential(
            nn.Linear(args.S_dim + args.D_dim + args.R_dim, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(args.ae_drop),
            nn.Linear(2048, args.X_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(args.ae_drop),
        )

    def forward(self, x):
        z = self.encoder(x)
        hn = z[:, :self.args.S_dim+self.args.D_dim]
        r = z[:, self.args.S_dim+self.args.D_dim:self.args.S_dim+self.args.D_dim+self.args.R_dim]
        s = self.S(z[:, :self.args.S_dim+self.args.D_dim])#这部分是 semantic的，待保存
        d = self.D(z[:, :self.args.S_dim+self.args.D_dim])
        z = torch.cat((s, d, r), 1)
        x1 = self.decoder(z)
        return x1, z, hn, s, d, r

class AE(nn.Module):
    def __init__(self, args):
        super(AE, self).__init__()
        self.args = args
        self.encoder = nn.Sequential(
            nn.Linear(args.X_dim, args.S_dim + args.NS_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(args.ae_drop)
        )
        self.decoder = nn.Sequential(
            nn.Linear(args.S_dim + args.NS_dim, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(args.ae_drop),
            nn.Linear(2048, args.X_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(args.ae_drop),
        )

    def forward(self, x):
        z = self.encoder(x)
        s = z[:, :self.args.S_dim]#这部分是 semantic的，待保存
        ns = z[:, self.args.S_dim:]
        x1 = self.decoder(z)
        return x1, z, s, ns


class RelationNet(nn.Module):
    def __init__(self, args):
        super(RelationNet, self).__init__()
        self.fc1 = nn.Linear(args.C_dim + args.S_dim, 2048) # C_dim 为 semantic attribute, S_dim 为 semantic-related visual
        self.fc2 = nn.Linear(2048, 1)

    def forward(self, s, c):

        c_ext = c.unsqueeze(0).repeat(s.shape[0], 1, 1)
        cls_num = c_ext.shape[1]

        s_ext = torch.transpose(s.unsqueeze(0).repeat(cls_num, 1, 1), 0, 1)
        relation_pairs = torch.cat((s_ext, c_ext), 2).view(-1, c.shape[1] + s.shape[1])
        relation = nn.ReLU()(self.fc1(relation_pairs))
        relation = nn.Sigmoid()(self.fc2(relation))
        return relation


class Discriminator(nn.Module):
    """此处提供 adversarial training，这里的 Discriminator 本质上是一个分类器，
    它试图捕捉 Residual Encoder 得到的对分类有价值的 feature 并进行正确分类
    Residual Encoder 就相当于一个 Generator，试图 encode 无法被正确分类的冗余信息
    """
    def __init__(self, R_dim, ntrain_class):
        super(Discriminator, self).__init__()
        self.cls = nn.Linear(R_dim, ntrain_class) #FLO 82
        self.logic = nn.LogSoftmax(dim=1)#Log

    def forward(self, s):
        o = self.logic(self.cls(s))
        return o
