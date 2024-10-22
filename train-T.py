import torch.optim as optim
import argparse
import random
import math
from time import gmtime, strftime
from models import *
from dataset_GBU import FeatDataLayer, DATA_LOADER
from my_utils import *
import torch.backends.cudnn as cudnn
import classifier  #
from triplet_loss import *
import matplotlib.pyplot as plt
from mi_estimators import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='SUN', help='dataset: CUB, AWA2, APY, FLO, SUN')
parser.add_argument('--dataroot', default='./SDGZSL_data', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--image_embedding', default='res101', type=str)
parser.add_argument('--class_embedding', default='att', type=str)

parser.add_argument('--gen_nepoch', type=int, default=400, help='number of epochs to train for')
parser.add_argument('--discriminative_VAE_lr', type=float, default=0.0001, help='learning rate to train generater')
parser.add_argument('--semantic_VAE_lr', type=float, default=0.0001, help='learning rate to train generater')

parser.add_argument('--zsl', type=bool, default=False, help='Evaluate ZSL or GZSL')
parser.add_argument('--ga', type=float, default=15, help='relationNet weight')
parser.add_argument('--residual', type=float, default=0.1, help='residual weight')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight_decay')
parser.add_argument('--kl_warmup', type=float, default=0.01, help='kl warm-up for VAE')

parser.add_argument('--vae_dec_drop', type=float, default=0.5, help='dropout rate in the VAE decoder')
parser.add_argument('--vae_enc_drop', type=float, default=0.4, help='dropout rate in the VAE encoder')
parser.add_argument('--ae_drop', type=float, default=0.2, help='dropout rate in the auto-encoder')

parser.add_argument('--ae_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--classifier_steps', type=int, default=40, help='training steps of the classifier')

parser.add_argument('--batchsize', type=int, default=64, help='AE input batch size')
parser.add_argument('--nSample', type=int, default=1200, help='number features to generate per class')

parser.add_argument('--disp_interval', type=int, default=200)
parser.add_argument('--evl_interval', type=int, default=400)
parser.add_argument('--evl_start', type=int, default=0)
parser.add_argument('--manualSeed', type=int, default=5606, help='manual seed')

parser.add_argument('--latent_dim', type=int, default=20, help='dimention of latent z')
parser.add_argument('--q_z_nn_output_dim', type=int, default=128, help='dimention of hidden layer in encoder')
parser.add_argument('--S_dim', type=int, default=1024)
parser.add_argument('--D_dim', type=int, default=512)
parser.add_argument('--R_dim', type=int, default=512)

parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
parser.add_argument('--name', type=str, default='')
parser.add_argument('--part', type=float, default=0.3, help='from which part do we start the training of VAE')
parser.add_argument('--margin', type=float, default=0.1, help='margin for triplet loss')
parser.add_argument('--dis', type=float, default=0.1, help='hyper-parameter for loss of discriminative visual feature')
parser.add_argument('--out', type=float, default=2.0, help='hyper-parameter for loss of outlier strength')
parser.add_argument('--mi', type=float, default=0.1, help='MI minimization strength')
parser.add_argument('--alter', type=int, default=1, help='Discriminator 多少 iters 训练一次')
parser.add_argument('--mi_epc', type=int, default=10, help='MI-estimator 每个 iters 训练多少次')

opt = parser.parse_args()

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
np.random.seed(opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True
print('Running parameters:')
print(opt)
opt.gpu = torch.device("cuda:" + opt.gpu if torch.cuda.is_available() else "cpu")  # 'cpu'#


def train():
    dataset = DATA_LOADER(opt)
    opt.C_dim = dataset.att_dim
    opt.X_dim = dataset.feature_dim
    opt.Z_dim = opt.latent_dim
    opt.y_dim = dataset.ntrain_class

    dataset.feature_dim = dataset.train_feature.shape[1]
    opt.adv_lr = opt.classifier_lr
    data_layer = FeatDataLayer(dataset.train_label.numpy(), dataset.train_feature.cpu().numpy(), opt)

    opt.niter = int(dataset.ntrain / opt.batchsize) * opt.gen_nepoch 
    opt.vae_start = opt.part * opt.niter
    VAE_model = VAE(opt, opt.S_dim+opt.D_dim).to(opt.gpu)
    relationNet = RelationNet(opt).to(opt.gpu)
    DiscriminatorNet = Discriminator(opt.R_dim, dataset.ntrain_class).to(opt.gpu)
    DiscriminatorNet.apply(weights_init)
    dis_criterion = nn.NLLLoss()
    ae = R_AE(opt).to(opt.gpu)
    estimator_name = 'CLUB'
    mi_estimator = eval(estimator_name)(x_dim=opt.S_dim+opt.D_dim, y_dim=opt.R_dim, hidden_size=2048).to(opt.gpu)
    mi_optimizer = optim.Adam(mi_estimator.parameters(), lr=opt.ae_lr)

    start_step = 0
    VAE_optimizer = optim.Adam(VAE_model.parameters(), lr=opt.semantic_VAE_lr, weight_decay=opt.weight_decay)
    relation_optimizer = optim.Adam(relationNet.parameters(), lr=opt.ae_lr, weight_decay=opt.weight_decay)
    ae_optimizer = optim.Adam(ae.parameters(), lr=opt.ae_lr, weight_decay=opt.weight_decay)
    adversarial_optimizer = optim.Adam(DiscriminatorNet.parameters(), lr=opt.adv_lr, betas=(0.5, 0.999))
    mse = nn.MSELoss().to(opt.gpu)

    iters = math.ceil(dataset.ntrain / opt.batchsize) 
    beta = 0.01
    best_H = 0
    best_T = 0
    m = []
    H = []
    T = []
    for it in range(start_step, opt.niter + 1):
        blobs = data_layer.forward()
        feat_data = blobs['data']
        labels_numpy = blobs['labels'].astype(int)
        labels = torch.from_numpy(labels_numpy.astype('int')).to(opt.gpu)
        C = np.array([dataset.train_att[i, :] for i in labels])
        C = torch.from_numpy(C.astype('float32')).to(opt.gpu)
        X = torch.from_numpy(feat_data).to(opt.gpu)
        sample_C = torch.from_numpy(np.array([dataset.train_att[i, :] for i in labels.unique()])).to(opt.gpu)
        sample_C_n = labels.unique().shape[0]
        sample_label = labels.unique().cpu()
        sample_labels = np.array(sample_label)
        re_batch_labels = []
        for label in labels_numpy:
            index = np.argwhere(sample_labels == label)
            re_batch_labels.append(index[0][0])
        re_batch_labels = torch.LongTensor(re_batch_labels)
        one_hot_labels = torch.zeros(opt.batchsize, sample_C_n).scatter_(1, re_batch_labels.view(-1, 1), 1).to(opt.gpu)

        mi_estimator.eval()
        ae.train()
        x, h, hs, hd, hr = ae(X)
        sampler_loss = opt.mi * mi_estimator(torch.cat((hs, hd), 1), hr)

        if it % opt.alter == 0:
            pred_logits = DiscriminatorNet(hr.detach())
            adv_loss = dis_criterion(pred_logits, labels)
            adversarial_optimizer.zero_grad()
            adv_loss.backward()
            adversarial_optimizer.step()


        relations = relationNet(hs, sample_C)
        relations = relations.view(-1, labels.unique().cpu().shape[0])
        p_loss = opt.ga * mse(relations, one_hot_labels)
        # triplet-loss
        dis_loss = opt.dis * batch_all_triplet_loss(re_batch_labels.to(opt.gpu), hd, opt.margin)[0]
        # reconstruction loss
        rec = mse(x, X)
        # residual loss
        residual_loss = opt.residual * -1 * dis_criterion(DiscriminatorNet(hr), labels)

        loss = p_loss + rec + dis_loss + residual_loss + sampler_loss
        relation_optimizer.zero_grad()
        ae_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        relation_optimizer.step()
        ae_optimizer.step()

        mi_estimator.train()
        ae.eval()
        for j in range(opt.mi_epc):
            x_samples, y_samples = torch.cat((hs, hd), 1), hr
            mi_loss = mi_estimator.learning_loss(x_samples, y_samples)
            mi_optimizer.zero_grad()
            mi_loss.backward(retain_graph=True)
            mi_optimizer.step()

        # VAE training
        if it > opt.vae_start: 
            to_gen = torch.cat((hs, hd), 1).detach()
            s_mean, z_mu, z_var, z = VAE_model(to_gen, C)
            if it % iters == 0:
                beta = min(opt.kl_warmup * ((it - opt.part * opt.niter) / iters), 1)
            loss_VAE, ce, kl = multinomial_loss_function(s_mean, to_gen, z_mu, z_var, z, beta=beta)

            VAE_optimizer.zero_grad()
            loss_VAE.backward()
            VAE_optimizer.step()
        # eval
        if it % opt.evl_interval == 0 and it >= opt.evl_start and it > opt.vae_start:
            m.append(it)
            VAE_model.eval()
            ae.eval()
            gen_unseen, gen_label = synthesize_feature_test(VAE_model, dataset, opt, opt.S_dim+opt.D_dim)
            gen_unseen_semantic = gen_unseen[:, :opt.S_dim]
            with torch.no_grad():
                train_feature = ae.encoder(dataset.train_feature.to(opt.gpu)).cpu().detach()
                train_feature_semantic = train_feature[:, :opt.S_dim]
                test_unseen_feature = ae.encoder(dataset.test_unseen_feature.to(opt.gpu)).cpu().detach()
                test_seen_feature = ae.encoder(dataset.test_seen_feature.to(opt.gpu)).cpu().detach()



            test_seen = test_seen_feature[:, :opt.S_dim]
            test_unseen = test_unseen_feature[:, :opt.S_dim]

            """ ZSL """
            cls = classifier.CLASSIFIER(opt, gen_unseen_semantic, gen_label,
                                        dataset, test_seen, test_unseen, dataset.ntest_class,
                                        True, opt.classifier_lr, 0.5,
                                        opt.classifier_steps, 1024, False)
            T.append(cls.T)
            print(f'iter {it}:  T={cls.T:.2f}')
            if cls.T > best_T:
                best_T = cls.T
            VAE_model.train()
            ae.train()
    print(f'=========================BEST acc: T = {best_T:.2f}')


if __name__ == "__main__":
    train()








