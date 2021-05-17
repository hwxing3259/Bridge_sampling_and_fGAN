import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sklearn.model_selection

np.random.seed(314159)
torch.manual_seed(314159)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ############################################################################################
# ######################### Part 1: Define Real-NVP ##########################################
# ############################################################################################
# modified from https://github.com/kamenbliznashki/normalizing_flows/blob/master/maf.py
# now the transforming part

# one Coupling layer of real-nvp
class Coupling(nn.Module):
    def __init__(self, old_par_size, hidden_size, hidden_layers, mask, additional_par_size=None):
        super(Coupling, self).__init__()
        # mask, masked entries will be transformed, the rest stays the same
        self.register_buffer("mask", mask)
        # scaling transformation for masked pars
        self.scale = [nn.Linear(old_par_size if additional_par_size is None else old_par_size+additional_par_size,
                                hidden_size)]
        for _ in range(hidden_layers):
            self.scale += [nn.Tanh(), nn.Linear(hidden_size, hidden_size)]
        self.scale += [nn.Linear(hidden_size,
                                 old_par_size if additional_par_size is None else old_par_size+additional_par_size)]
        self.scale = nn.Sequential(*self.scale)

        # shift transformation for masked pars
        self.shift = [nn.Linear(old_par_size if additional_par_size is None else old_par_size+additional_par_size,
                                hidden_size)]
        for _ in range(hidden_layers):
            self.shift += [nn.ReLU(), nn.Linear(hidden_size, hidden_size)]
        self.shift += [nn.Linear(hidden_size,
                                 old_par_size if additional_par_size is None else old_par_size+additional_par_size)]
        self.shift = nn.Sequential(*self.shift)

    def forward(self, x):
        mx = x * self.mask  # masked input will be transformed
        # get scale and shift
        s = self.scale(mx)
        t = self.shift(mx)
        # apply real-nvp
        u = mx + (1 - self.mask) * (torch.exp(-s)*(x-t))
        logdet = (1 - self.mask) * (-s)

        return u, logdet  # both u and log det has shape batch size*(old_par_size+additional_par_size)

    def inverse(self, u):
        mu = u * self.mask  # note that mu = mx, they are inputs that are not masked i.e. stay the same
        s = self.scale(mu)
        t = self.shift(mu)
        x = mu + (1 - self.mask) * (u * torch.exp(s) + t)
        logdet = (1 - self.mask) * s
        return x, logdet  # both x and log det has shape batch size*(old_par_size+additional_par_size)


class nvpSeq(nn.Sequential):
    def forward(self, x):
        logdet = 0
        for mod in self:
            x, det = mod(x)
            logdet += det
        return x, logdet

    def inverse(self, u):
        logdet = 0
        for mod in reversed(self):
            u, det = mod.inverse(u)
            logdet += det
        return u, logdet


# actually batch norm is not a good idea x_x, did not use
class BatchNorm(nn.Module):
    """ RealNVP BatchNorm layer """
    # taken from https://github.com/kamenbliznashki/normalizing_flows/blob/master/maf.py
    def __init__(self, input_size, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.log_gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))

        self.register_buffer('running_mean', torch.zeros(input_size))
        self.register_buffer('running_var', torch.ones(input_size))

    def inverse(self, x, cond_y=None):
        if self.training:
            self.batch_mean = x.mean(0)
            self.batch_var = x.var(0)  # note MAF paper uses biased variance estimate; ie x.var(0, unbiased=False)

            # update running mean
            self.running_mean.mul_(self.momentum).add_(self.batch_mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(self.batch_var.data * (1 - self.momentum))

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        # compute normalized input (cf original batch norm paper algo 1)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.log_gamma.exp() * x_hat + self.beta

        # compute log_abs_det_jacobian (cf RealNVP paper)
        log_abs_det_jacobian = self.log_gamma - 0.5 * torch.log(var + self.eps)
        # print('in sum log var {:6.3f} ; out sum log var {:6.3f}; sum log det {:8.3f}; mean log_gamma {:5.3f};
        # mean beta {:5.3f}'.format(
        # (var + self.eps).log().sum().data.numpy(), y.var(0).log().sum().data.numpy(),
        # log_abs_det_jacobian.mean(0).item(), self.log_gamma.mean(), self.beta.mean()))
        return y, log_abs_det_jacobian.expand_as(x)

    def forward(self, y, cond_y=None):
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (y - self.beta) * torch.exp(-self.log_gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean

        log_abs_det_jacobian = 0.5 * torch.log(var + self.eps) - self.log_gamma

        return x, log_abs_det_jacobian.expand_as(x)


# this real nvp aims to map samples from base density to a target.
# forward evaluation: it takes samples from target, ideally it should be mapped back to the base
# inverse: take samples from the base, map to target
# our aim is to make net.inverse(base samples) to be as close to the target as possible
# log prob: get a sample x, map it back to u, return base_density(u)+log_det
class mynvp(nn.Module):
    def __init__(self, input_par_size, n_block, hidden_size, unnorm_base_density,
                 hidden_layers,  extra_base_mean=None, extra_base_cov=None, additional_dim=None,
                 final_transformation=None, batch_norm=False):
        """
        :param input_par_size: dimension of the density with lower dimension, will be augmented by standard normal
        to mach the dimension of the density with higher dimension
        :param n_block: number of coupling layers of the real-NVP
        :param hidden_size: number of hidden layers for the scaling/shifting neural nets in one coupling layer
        :param unnorm_base_density: unnormalized log density of the lower dimensional one, does not matter if
        q1,q2 have the same dimension, should be a nn.Module, assume it has been send to CUDA if necessary
        :param hidden_layers: number of hidden layers for the scaling/shifting neural nets in one coupling layer
        :param extra_base_mean: mean of the augmenting density, default is 0
        :param extra_base_cov: cov of the augmenting density, default is identity matrix
        :param additional_dim: difference in dimension of q1,q2, 0 if same, positive integer otherwise, keep in mind
        that q1 is the one with lower dimension
        :param final_transformation: we may need a final transformation that maps the rvs back to the constrained
        supports, not always necessary
        """
        # base density should be a nn.module, gotta send it to device manually tho
        super(mynvp, self).__init__()
        self.input_par_size = input_par_size
        self.additional_dim = additional_dim
        self.register_buffer('mask', torch.arange(self.input_par_size if self.additional_dim is None else
                                                  self.input_par_size+self.additional_dim, dtype=torch.float) % 2)
        batch_norm = False
        # build real nvp
        net = [Coupling(self.input_par_size, hidden_size, hidden_layers, self.mask, self.additional_dim)]
        for _ in range(n_block-1):
            net += batch_norm * [BatchNorm(self.input_par_size if self.additional_dim is None else
                                           self.input_par_size+self.additional_dim)]
            self.mask = 1-self.mask
            net += [Coupling(self.input_par_size, hidden_size, hidden_layers, self.mask, self.additional_dim)]
        if final_transformation is not None:
            net += [final_transformation]
            # we may need a final transformation that maps the rvs back to the constrained support
            # should be another nn.Module
        # specify the base density for extra dims
        if extra_base_mean is None:
            self.register_buffer('extra_base_mean', torch.zeros(additional_dim))
        else:
            self.register_buffer('extra_base_mean', extra_base_mean)
        if extra_base_cov is None:
            self.register_buffer('extra_base_cov', torch.diag(torch.ones(additional_dim)))
        else:
            self.register_buffer('extra_base_cov', extra_base_cov)

        # assume base_density has been send to CUDA if necessary
        self.unnorm_base_density = unnorm_base_density
        self.net = nvpSeq(*net)

    def _extra_density(self, x_extra_dim):  # handles the augmented parts
        if self.additional_dim == 0 or self.additional_dim is None:
            return 0
        else:
            m = torch.distributions.normal.Normal(loc=self.extra_base_mean,
                                                  scale=torch.sqrt(torch.diag(self.extra_base_cov)))
            return m.log_prob(x_extra_dim).sum(dim=1)

    def forward(self, x):  # maps obs x to base u
        u, det = self.net(x)
        return u, det  # both x and lod ett has shape batch size*(old_par_size+additional_par_size)

    def inverse(self, u):  # maps base u to target x
        x, det = self.net.inverse(u)
        return x, det  # both x and log det has shape batch size*(old_par_size+additional_par_size)

    def log_prob(self, x):  # returns un normalized, transformed log density
        u, det = self.net(x)
        u1 = u[:, :self.input_par_size]  # base density part
        u2 = u[:, self.input_par_size:]  # augmented density part
        return torch.sum(det, dim=1) + self.unnorm_base_density(u1) + self._extra_density(u2)

    def augmented_base_density(self, u):  # return the augmented base log density
        u1 = u[:, :self.input_par_size]
        u2 = u[:, self.input_par_size:]
        if self.additional_dim == 0 or self.additional_dim is None:
            return self.unnorm_base_density(u1)
        else:
            return self.unnorm_base_density(u1) + self._extra_density(u2)


# ###########################################################################################
# ######################### Part 2: define f-GAN objectives #################################
# ###########################################################################################

# These functions take 6 inputs. q1(w1), q1(w2), q2(w1), q2(w2), logb=estimated log Bayes factor,
# pi=n1/(n1+n2) is the weight parameter
# implemented using log-sum-exp trick
# loss for discriminator (max part) will be be defined later

def loss_generator_js(transformed_sample_at_transformed_density, transformed_sample_at_target_density,
                      target_sample_at_transformed_density, target_sample_at_target_density, logb, pi):
    if pi.device.type == "cpu":
        pi = pi.to("cuda")
    logit_score = torch.cat(
        (transformed_sample_at_transformed_density - transformed_sample_at_target_density - logb,
         target_sample_at_transformed_density - target_sample_at_target_density - logb))
    logit_score += torch.log(pi/(1-pi))
    mylabel = torch.cat((torch.ones(transformed_sample_at_transformed_density.shape[0], device=device),
                         torch.zeros(target_sample_at_target_density.shape[0], device=device)))
    loss = torch.nn.BCEWithLogitsLoss(reduction="sum")
    return -1 * loss(logit_score, mylabel)
    # recall that f-GAN objective is the classification lkd = -1*BCE_loss


def loss_generator_hellinger(transformed_sample_at_transformed_density, transformed_sample_at_target_density,
                             target_sample_at_transformed_density, target_sample_at_target_density, logb, pi):
    if pi.device.type == "cpu":
        pi = pi.to("cuda")
    log_density_ratio_with_n1n2adj = 0.5*torch.cat(
        (-transformed_sample_at_transformed_density + transformed_sample_at_target_density + logb,
         target_sample_at_transformed_density - target_sample_at_target_density - logb + 2*torch.log(pi/(1-pi))))
    maxxx = torch.max(log_density_ratio_with_n1n2adj)  # max of all n1+n2 log ratio components
    lse_trick_remaining = torch.exp(log_density_ratio_with_n1n2adj - maxxx).sum()  # sum_exp part
    return -1*maxxx - torch.log(lse_trick_remaining)
    # log_sum_exp, times negative 1
    # so minimize original f-GAN hellinger is equivalent to minimizing the log_sum_exp form


def loss_generator_harmonic(transformed_sample_at_transformed_density, transformed_sample_at_target_density,
                            target_sample_at_transformed_density, target_sample_at_target_density, logb, pi):
    if pi.device.type == "cpu":
        pi = pi.to("cuda")
    logit_score = torch.cat(  # logit score is log(q1/q2r)
        (transformed_sample_at_transformed_density - transformed_sample_at_target_density - logb,
         target_sample_at_transformed_density - target_sample_at_target_density - logb))
    logit_score += torch.log(pi/(1-pi))  # logit score is log(s1q1/s2q2r)
    loss = torch.nn.BCEWithLogitsLoss(reduction="none")
    mylabel = torch.cat((torch.zeros(transformed_sample_at_transformed_density.shape[0], device=device),
                         torch.ones(target_sample_at_target_density.shape[0], device=device)))
    mylogsigmoid = -2*loss(logit_score, mylabel)  # this gives 2logs2q2r/s1q1+s2q2r for w1, 2logs1q1/s1q1+s2q2r for w2
    maxxx = torch.max(mylogsigmoid)  # find the max lij
    lse_trick_remaining = torch.exp(mylogsigmoid - maxxx).sum()  # lse
    # approximately log sum (s2q2r/s1q1+s2q2r)^2 + sum (s1q1/s1q1+s2q2r)^2
    return -1*maxxx - torch.log(lse_trick_remaining)

# ##############################################################################
# ########################## Discriminator (max part) ##########################
# ##############################################################################


class Discriminator(nn.Module):
    def __init__(self, logb, fdiv="harmonic"):
        super(Discriminator, self).__init__()
        self.register_parameter("logb", nn.Parameter(logb))
        self.fdiv = fdiv

    def forward(self, transformed_sample_at_transformed_density, transformed_sample_at_target_density,
                target_sample_at_transformed_density, target_sample_at_target_density, pi):
        """
        :param transformed_sample_at_transformed_density: transformed_q1(transformed_w1),
        transformed_w1=nvp.inverse(w1), transformed_q1 = nvp.logprob(x), recall that forward call maps given x to base
        distribution, inverse call maps base sample to target. This also means transformed_q1(transformed_w1) =
        nvp.logprob(nvp.inverse(w1)) = base_density(nvp(nvp.inverse(w1)) + logdet_prob = base_density(w1)+logdet_prob;
        on the other hand, compute nvp.inverse(w1) will also give us a logdet_inv term, which is precisely the negative
        of logdet_prob we need for density evaluation. So nvp.logprob(nvp.inverse(w1)) = base_density(w1) - log_det
        :param transformed_sample_at_target_density: q2(w1)
        :param target_sample_at_transformed_density: transformed_q1(w2), transformed_q1 is net.log_prob(w2)
        :param target_sample_at_target_density: q2(w2)
        :param pi: n1/n1+n2, equivalent to s1 in Meng and Wong
        :return: if fdiv=JS, return weighted BCE, or the NEGATIVE classification log likelihood,
        aim to find optimal logb;  if fdiv=hellinger, then optimize Hellinger distance using f-GAN, same for harmonic
        """
        if pi.device.type == "cpu":
            pi = pi.to("cuda")
        if self.fdiv == "js":
            fhat = loss_generator_js(transformed_sample_at_transformed_density,
                                     transformed_sample_at_target_density,
                                     target_sample_at_transformed_density, target_sample_at_target_density,
                                     self.logb, pi)
            return -1*fhat  # recall that this is the max step, so minimize the negative of f-GAN objective
        elif self.fdiv == "hellinger":
            fhat = loss_generator_hellinger(transformed_sample_at_transformed_density,
                                            transformed_sample_at_target_density,
                                            target_sample_at_transformed_density, target_sample_at_target_density,
                                            self.logb, pi)
            return -1*fhat
        elif self.fdiv == "harmonic":
            fhat = loss_generator_harmonic(transformed_sample_at_transformed_density,
                                           transformed_sample_at_target_density,
                                           target_sample_at_transformed_density, target_sample_at_target_density,
                                           self.logb, pi)
            return -1*fhat
        else:
            print("sorry boss not implemented, try js, harmonic or hellinger")


# ##############################################################################################################
# ################################## Data loader ###############################################################
# ##############################################################################################################

class TargetProposalDataset(torch.utils.data.Dataset):
    """
    q1, q1 are samples from q1,q2, stack them, give label 1 if it's from q1, 0 if it's from q2,
    don't actually need the weight, too lazy to remove it lol
    """
    def __init__(self, q1, q2):
        super(TargetProposalDataset, self).__init__()
        self.mytrainingdata = torch.cat((q1, q2), dim=0)
        self.mytraininglabel = torch.cat((torch.ones(q1.shape[0]), torch.zeros(q2.shape[0])))
        self.mytrainingweight = torch.cat((torch.ones(q1.shape[0]), torch.ones(q2.shape[0])*(q1.shape[0]/q2.shape[0])))

    def __getitem__(self, item):
        return self.mytrainingdata[item], self.mytraininglabel[item], self.mytrainingweight[item]

    def __len__(self):  # returns n1+n2
        return self.mytrainingdata.shape[0]


# #############################################################################################################
# ############################### f-GAN Bridge estimate, the main part ########################################
# #############################################################################################################
class fGANBridge:
    def __init__(self, p1_sample, p1_log_density, p2_sample, p2_log_density,
                 p1_estimating, p2_estimating, fdiv="harmonic"):
        """
        :param p1_sample: samples from p1, np array will do
        :param p1_log_density: log density of p1, torch.nn.Module, so we can move to cuda
        :param p2_sample: samples from p2, np array will do
        :param p2_log_density: log density of p2, torch.nn.Module, so we can move to cuda
        :param p1_estimating: separate estimating set, samples from p1, independent to p1_sample
        :param p2_estimating: separate estimating set, samples from p2, independent to p2_sample
        :param fdiv: type of f-div, default is harmonic
        """
        # get sample sizes and dimensions
        self.n1 = p1_sample.shape[0]
        self.n2 = p2_sample.shape[0]
        self.input_size = p1_sample.shape[-1]
        self.d1 = p1_sample.shape[-1]
        self.d2 = p2_sample.shape[-1]

        # define samples from p1,p2, augmenting if dims are different
        self.dim_diff = self.d2 - self.d1
        if self.dim_diff == 0:
            self.augmented_p1_sample = p1_sample
            self.augmented_p1_estimating = p1_estimating
        else:
            try:
                self.augmented_p1_sample = np.c_[p1_sample, np.random.randn(self.n1, self.dim_diff)]
                self.augmented_p1_estimating = np.c_[p1_estimating,
                                                     np.random.randn(p1_estimating.shape[0], self.dim_diff)]
            except ValueError:
                print("p1 should have smaller dim u dummy")
        self.p2_sample = p2_sample
        self.p2_estimating = p2_estimating

        # set device, send relative funcs to GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.my_base_density = p1_log_density.to(self.device)  # base density
        self.my_target_density = p2_log_density.to(self.device)  # target density
        self.myGenerator = "not yet specified"  # the min part of f-GAN, real-nvp
        self.myDiscriminator = "not yet specified"  # the max part of f-GAN
        self.training_loader = "not yet specified"  # training data loader
        self.testing_loader = "not yet specified"  # testing data loader
        self.fitted_r = "not yet specified"  # fitted bayes factor r
        self.raw_r = "not yet specified"  # raw output from the min-max problem,needs to be refined using optimal Bridge
        self.geometric_bridge = 0.
        self.training_seq = "run .fit first"  # list of training error for each iteration
        self.estimated_RMSE = "run .fit first"
        self.B_seq = "run .fit first"  # list of log r for each iteration
        self.testing_seq = "run .fit first"  # list of testing error for each iteration
        self.training_seq_iter = np.zeros(1)  # ???
        self.B_seq_iter = np.zeros(1)  # ???

        # choose f-divergence, set the objective func for generator, will set discriminator later
        self.fdiv = fdiv
        self.minimized_rmse = 0.
        if self.fdiv == "js":
            self.loss_generator = loss_generator_js
        elif self.fdiv == "hellinger":
            self.loss_generator = loss_generator_hellinger
        elif self.fdiv == "harmonic":
            self.loss_generator = loss_generator_harmonic
        else:
            print("sorry boss not implemented yet, try js, harmonic or hellinger")

    # FIT!
    def fit(self, n_block, hidden_size, hidden_layers, epoch=10, lr_generator=8e-4, lr_discriminator=5.,
            beta_for_kl1=0.2, beta_for_kl2=0.2, batch_norm=False, rinit=0., wait=8, k=1, max_iter=1000, opt="SGD"):
        """

        :param n_block: number of coupling layers
        :param hidden_size: # of nodes for each nn-layer of the coupling layer
        :param hidden_layers: # of nn-layers for each coupling layer
        :param epoch: # of iterations of training
        :param lr_generator: learning rate of generator
        :param lr_discriminator: learning rate of discriminator
        :param beta_for_kl1: additional forward log lkd term to stabilize f-GAN training
        :param beta_for_kl2: additional backward log lkd term to stabilize f-GAN training
        :param batch_norm: do we need batch norm? probably not lol please leave it as False
        :param rinit: initial value of r, the Bayes factor to estimate
        :param wait: patience of the the lr scheduler for generator
        :param k: update r every k batches, default is 1
        :param max_iter: max iter of Meng and Wong iter seq
        :param opt: choice of discriminator's optimizer, sgd or adam?
        :return: hopefully a good estimate of the Bayes factor
        and the f-divergence between the transformed p1 and original p2
        """
        self.myGenerator = mynvp(input_par_size=self.d1, n_block=n_block, hidden_size=hidden_size,
                                 hidden_layers=hidden_layers, unnorm_base_density=self.my_base_density,
                                 additional_dim=self.dim_diff, batch_norm=batch_norm)
        self.myGenerator = self.myGenerator.to(self.device)  # move to cuda if needed
        optimizer_generator = torch.optim.Adam(params=self.myGenerator.parameters(), lr=lr_generator)  # choose Adam
        schlr_generator = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_generator, mode="min",
                                                                     verbose=True, patience=wait)

        self.myDiscriminator = Discriminator(torch.tensor(rinit), fdiv=self.fdiv)
        self.myDiscriminator = self.myDiscriminator.to(self.device)  # move to cuda if needed
        if opt == "SGD":
            optimizer_discriminator = torch.optim.SGD(params=self.myDiscriminator.parameters(), lr=lr_discriminator)
        else:
            optimizer_discriminator = torch.optim.Adam(params=self.myDiscriminator.parameters(), lr=lr_discriminator)

        lambda_kl_backward = torch.tensor(beta_for_kl1, device=self.device)  # governs KL(q2, q1T)
        lambda_kl_forward = torch.tensor(beta_for_kl2, device=self.device)  # governs KL(q1T, q2)

        data_set = TargetProposalDataset(torch.tensor(self.augmented_p1_sample, dtype=torch.float),
                                         torch.tensor(self.p2_sample, dtype=torch.float))
        # stratified split
        splitter = sklearn.model_selection.StratifiedKFold()
        training_idx, testing_idx = splitter.split(np.zeros((self.n1+self.n2, 2)),
                                                   np.concatenate((np.ones(self.n1), np.zeros(self.n2)))).__next__()

        training_idx_sampler = torch.utils.data.SubsetRandomSampler(training_idx)
        testing_idx_sampler = torch.utils.data.SubsetRandomSampler(testing_idx)
        # get training set and testing set that has the same proportion of samples from p1,p2
        self.training_loader = torch.utils.data.DataLoader(data_set, batch_size=128, sampler=training_idx_sampler)
        self.testing_loader = torch.utils.data.DataLoader(data_set, batch_size=128, sampler=testing_idx_sampler)

        # send both augmented proposal and target samples to device, need this for discriminator
        self.augmented_p1_sample = torch.tensor(self.augmented_p1_sample, dtype=torch.float, device=self.device)
        self.p2_sample = torch.tensor(self.p2_sample, dtype=torch.float, device=self.device)
        self.augmented_p1_estimating = torch.tensor(self.augmented_p1_estimating, dtype=torch.float, device=self.device)
        self.p2_estimating = torch.tensor(self.p2_estimating, dtype=torch.float, device=self.device)
        s1 = torch.tensor(self.n1/(self.n1+self.n2), device=self.device)

        # training process
        training_obj = np.zeros(epoch)  # record the training error
        test_obj = np.zeros(epoch)  # record the testing error
        Bseq = np.zeros(epoch)  # record the estimated Bayes factor r
        for EPOCH in range(epoch):
            training_loss = 0  # training error for one epoch
            test_loss = 0  # testing error for one epoch
            test_mixture_kl = 0  # monitor kl divergence as a benchmark of fitting?
            self.myGenerator.train()  # switch on training mode, does not do much if batch_norm disabled
            for i, (data_batch, label_batch, weight_batch) in enumerate(self.training_loader):
                label_batch = label_batch.type(dtype=torch.bool)  # labels are used to identify samples from p1 or p2
                aug_q1_batch = data_batch[label_batch]  # those from q1
                q2_batch = data_batch[~label_batch]  # those from q2
                aug_q1_batch = aug_q1_batch.to(self.device)  # send to device
                q2_batch = q2_batch.to(self.device)  # send to device
                # weight and label is handled inside generator and discriminator loss, don't need them for now

                # standard training steps for generator
                aug_q1_transformed_batch, aug_q1_transformed_det_batch = self.myGenerator.inverse(aug_q1_batch)
                # this will compute the transformed proposals and the corresponding logdet,
                # former will be used to compute transformed_sample_at_target_density,
                # later for transformed_sample_at_transformed_density, we do not want to recompute
                q2_sample_at_q2_density_batch = self.my_target_density(q2_batch)
                q2_sample_at_transformed_q1_density_batch = self.myGenerator.log_prob(q2_batch)
                transformed_q1_sample_at_q2_density_batch = self.my_target_density(aug_q1_transformed_batch)
                transformed_q1_sample_at_transformed_q1_density_batch = \
                    self.myGenerator.augmented_base_density(aug_q1_batch) - \
                    torch.sum(aug_q1_transformed_det_batch, dim=1)

                # logb fixed, negative loglkd is backward KL, also have forward KL,
                # so we are minimizing a mixture of JS,KL and backward KL
                # generator update steps
                loss_g = self.loss_generator(transformed_q1_sample_at_transformed_q1_density_batch,
                                             transformed_q1_sample_at_q2_density_batch,
                                             q2_sample_at_transformed_q1_density_batch,
                                             q2_sample_at_q2_density_batch,
                                             self.myDiscriminator.logb.detach(), s1) - \
                         lambda_kl_backward * q2_sample_at_transformed_q1_density_batch.mean() - \
                         lambda_kl_forward * (transformed_q1_sample_at_q2_density_batch.mean() -
                                              transformed_q1_sample_at_transformed_q1_density_batch.mean())

                optimizer_generator.zero_grad()
                loss_g.backward()
                optimizer_generator.step()
                training_loss += loss_g.detach().cpu().item()*128
                # monitor the forward and backward KL as benchmark for fitting
                if i % 50 == 0:
                    with torch.no_grad():
                        print("THIS ALSO MATTERS: ", "Monitoring backward KL:",
                              -1 * torch.mean(q2_sample_at_transformed_q1_density_batch.detach()).cpu(),
                              "monitoring forward KL",
                              -1 * (transformed_q1_sample_at_q2_density_batch.mean()
                                    - transformed_q1_sample_at_transformed_q1_density_batch.mean()).detach().cpu()
                              )

                # for every k batches, update discriminator use this batch
                if i % k == 0:  # discriminator updates using batch data
                    # ##############One step Gradient Descent using batch
                    # recall that discriminator loss is flipped, so we still minimize wrt b
                    loss_d = self.myDiscriminator(transformed_q1_sample_at_transformed_q1_density_batch.detach(),
                                                  transformed_q1_sample_at_q2_density_batch.detach(),
                                                  q2_sample_at_transformed_q1_density_batch.detach(),
                                                  q2_sample_at_q2_density_batch.detach(), s1)
                    # logb is now the parameter, does not appear here
                    # UPDATE B
                    optimizer_discriminator.zero_grad()
                    loss_d.backward()
                    optimizer_discriminator.step()
                    # monitor current fitted value
                    if i % 50 == 0:
                        print("Curr estimate of r: ", self.myDiscriminator.logb.detach().cpu())

            # once one epoch is finished, compute averaged training and test error
            training_obj[EPOCH] = training_loss/(0.8*self.n1+0.8*self.n2)
            Bseq[EPOCH] = self.myDiscriminator.logb.detach().cpu().item()

            # ##########################start evaluation part################################################
            self.myGenerator.eval()  # turn on evaluation mode
            with torch.no_grad():
                for j, (data_test, label_test, weight_test) in enumerate(self.testing_loader):
                    # same se before, but use the test set (20% of p1,p2_sample)
                    label_test = label_test.type(dtype=torch.bool)
                    q1_test_sample = data_test[label_test]
                    q2_test_sample = data_test[~label_test]
                    q1_test_sample = q1_test_sample.to(self.device)
                    q2_test_sample = q2_test_sample.to(self.device)
                    # again, same as before
                    aug_q1_transformed_test, aug_q1_transformed_det_test = self.myGenerator.inverse(q1_test_sample)
                    # this will compute the transformed q1 samples and the corresponding logdet,
                    # former will be used to compute transformed_q1_sample_at_q2_density,
                    # later for transformed_q1_sample_at_transformed_q1_density, we do not want to recompute
                    q2_sample_at_q2_density_test = self.my_target_density(q2_test_sample)
                    q2_sample_at_transformed_q1_density_test = self.myGenerator.log_prob(q2_test_sample)
                    transformed_q1_sample_at_q2_density_test = self.my_target_density(aug_q1_transformed_test)
                    transformed_q1_sample_at_transformed_q1_density_test = \
                        self.myGenerator.augmented_base_density(q1_test_sample) - \
                        torch.sum(aug_q1_transformed_det_test, dim=1)
                    # compute test loss, mixture of target f-div, forward and backward KL
                    loss_g_test = self.loss_generator(transformed_q1_sample_at_transformed_q1_density_test,
                                                      transformed_q1_sample_at_q2_density_test,
                                                      q2_sample_at_transformed_q1_density_test,
                                                      q2_sample_at_q2_density_test,
                                                      self.myDiscriminator.logb.detach(), s1) - \
                                  lambda_kl_backward * q2_sample_at_transformed_q1_density_test.mean() - \
                                  lambda_kl_forward * (transformed_q1_sample_at_q2_density_test.mean() -
                                                       transformed_q1_sample_at_transformed_q1_density_test.mean())
                    # recall that log_b fixed at current value, stored in myDiscriminator
                    test_loss += loss_g_test*128
                    # a differnet benchmark, only involves forward and backward KL
                    test_mixture_kl += -1*q2_sample_at_transformed_q1_density_test.mean().cpu() + \
                                       -1*(transformed_q1_sample_at_q2_density_test.mean().cpu() -
                                           transformed_q1_sample_at_transformed_q1_density_test.mean().cpu())
            # record averaged test loss
            test_obj[EPOCH] = test_loss.cpu().item()/(0.2*self.n1+0.2*self.n2)
            print("EPOCH: ", EPOCH, "TEST LOSS: ", test_obj[EPOCH], "Current r: ", Bseq[EPOCH])
            self.myGenerator.train()
            schlr_generator.step(test_mixture_kl)  # activate learning rate scheduler

        # ###################################### Learning iteration done!#####################################
        # record the training, testing error and fitted r sequences
        self.B_seq = Bseq
        self.training_seq = training_obj
        self.testing_seq = test_obj

        # ################################## use the estimating set to find the optimal Bridge estimate ##########
        self.myGenerator.eval()
        with torch.no_grad():
            q2_sample_at_q2_density_all = self.my_target_density(self.p2_estimating)
            q2_sample_at_transformed_q1_density_all = self.myGenerator.log_prob(self.p2_estimating)
            aug_q1_transformed_all, aug_q1_transformed_det_all = self.myGenerator.inverse(self.augmented_p1_estimating)
            transformed_q1_sample_at_q2_density_all = self.my_target_density(aug_q1_transformed_all)
            transformed_q1_sample_at_transformed_q1_density_all = \
                self.myGenerator.augmented_base_density(self.augmented_p1_estimating) - \
                torch.sum(aug_q1_transformed_det_all, dim=1)
        # output r from the min max problem, recall that it is a biased estimate of r
        # will use it as the initial value of the iterative procedure of finding optimal Bridge estimate
        old_r = self.myDiscriminator.logb.detach()
        self.raw_r = old_r
        rel_diff = 1.
        curr_t = 0
        print(old_r)
        ll1 = q2_sample_at_transformed_q1_density_all - q2_sample_at_q2_density_all + torch.log(s1/(1-s1))
        ll2 = transformed_q1_sample_at_transformed_q1_density_all - transformed_q1_sample_at_q2_density_all + \
              torch.log(s1/(1-s1))

        while curr_t < max_iter and rel_diff > 1e-4:
            # update rule on log scale, for numeric stability, use log sum exp here
            l1 = F.logsigmoid(ll1 - old_r)
            l2 = F.logsigmoid(old_r - ll2)
            l1max = torch.max(l1)
            part1 = l1max + torch.log(torch.sum(torch.exp(l1 - l1max)))
            l2max = torch.max(l2)
            part2 = l2max + torch.log(torch.sum(torch.exp(l2 - l2max)))
            new_r = old_r + part1 - part2
            rel_diff = torch.abs(new_r-old_r)/torch.abs(old_r)
            rel_diff = rel_diff.cpu().item()
            old_r = new_r
            curr_t += 1
        # if optimal Bridge does not converge... TRY GEOMETRIC BRIDGE
        # tired of implementing specific Bridge estimate, just search for the max lol 1d optimization not so bad

        # candidates = torch.linspace(old_r-20., old_r+20., 2000, device=self.device)
        # objective_value = np.zeros(3000)
        # for i, j in enumerate(candidates):
        #     objective_value[i] = self.loss_generator(transformed_q1_sample_at_transformed_q1_density_all,
        #                                              transformed_q1_sample_at_q2_density_all,
        #                                              q2_sample_at_transformed_q1_density_all,
        #                                              q2_sample_at_q2_density_all, j, s1).cpu().item()

        # minimized_f_div = np.max(objective_value)
        # estimating_n2 = self.p2_estimating.shape[0]
        # estimating_n1 = self.augmented_p1_estimating.shape[0]
        # n1s2 = (estimating_n1*estimating_n2)/(estimating_n1+estimating_n2)
        # self.minimized_rmse = [(np.exp(minimized_f_div+np.log(n1s2))-1)/n1s2, minimized_f_div]

        if curr_t == max_iter:
            print("optimal Bridge estimate did not converge, use sub-optimal Geometric bridge")
            a1 = 0.5*(transformed_q1_sample_at_q2_density_all - transformed_q1_sample_at_transformed_q1_density_all)
            a2 = 0.5*(q2_sample_at_transformed_q1_density_all - q2_sample_at_q2_density_all)
            a1max = torch.max(a1)
            a2max = torch.max(a2)
            part1 = a1max + torch.log(torch.sum(torch.exp(a1-a1max)))  # log-sum-exp again
            part2 = a2max + torch.log(torch.mean(torch.exp(a2-a2max)))
            self.geometric_bridge = (part2 - part1 + torch.log(s1/(1-s1))).cpu().item()
            self.fitted_r = old_r.cpu().item()
        else:
            self.fitted_r = old_r.cpu().item()

        mydiv = self.divergence_estimation()['harmonic']
        mys1 = self.augmented_p1_estimating.shape[0]/(self.p2_estimating.shape[0]+self.augmented_p1_estimating.shape[0])
        self.estimated_RMSE = \
            (1/(1-mydiv)-1)/(mys1*(1-mys1)*(self.p2_estimating.shape[0]+self.augmented_p1_estimating.shape[0]))
        return self.fitted_r

    def divergence_estimation(self, p1_sample=None, p2_sample=None, fdiv='harmonic'):
        """
        estimating the f-divergence between transformed q1 and q2 using the variational framework
        :return: estimated Squared Hellinger distance, weighted JS divergence and weighted Harmonic divergence
        """
        if p1_sample is None and p2_sample is None:
            augmented_p1_sample = self.augmented_p1_estimating
            p2_sample = self.p2_estimating
        else:
            augmented_p1_sample = torch.tensor(p1_sample, dtype=torch.float, device=self.device)
            if self.dim_diff != 0:
                augmented_p1_sample = torch.cat((augmented_p1_sample,
                                                 torch.randn((p1_sample.shape[0],
                                                              self.dim_diff), device=self.device)), dim=1)
            p2_sample = torch.tensor(p2_sample, dtype=torch.float, device=self.device)

        self.myGenerator.eval()
        with torch.no_grad():
            q2_sample_at_q2_density_all = self.my_target_density(p2_sample)
            q2_sample_at_transformed_q1_density_all = self.myGenerator.log_prob(p2_sample)
            aug_q1_transformed_all, aug_q1_transformed_det_all = self.myGenerator.inverse(augmented_p1_sample)
            transformed_q1_sample_at_q2_density_all = self.my_target_density(aug_q1_transformed_all)
            transformed_q1_sample_at_transformed_q1_density_all = \
                self.myGenerator.augmented_base_density(augmented_p1_sample) - \
                torch.sum(aug_q1_transformed_det_all, dim=1)

        candidate = torch.linspace(self.fitted_r-20., self.fitted_r+20., 5000, device=self.device)
        s1 = torch.tensor(self.n1/(self.n1+self.n2), device=self.device)
        final_estimate = 0
        # estimate JS:
        if fdiv == "js":
            js_candidate = torch.zeros(5000, device=device)
            for i, j in enumerate(candidate):
                a1 = F.logsigmoid(torch.log(s1/(1-s1)) + transformed_q1_sample_at_transformed_q1_density_all -
                                  transformed_q1_sample_at_q2_density_all - j).mean()
                a2 = F.logsigmoid(q2_sample_at_q2_density_all + j - q2_sample_at_transformed_q1_density_all -
                                  torch.log(s1/(1-s1))).mean()
                js_candidate[i] = s1*a1 + (1-s1)*a2 - s1*torch.log(s1) - (1-s1)*torch.log(1-s1)
            js_estimate = torch.max(js_candidate).cpu().item()
            final_estimate = js_estimate
        if fdiv == "hellinger":
            # estimate Hellinger
            h_candidate = torch.zeros(5000, device=device)
            for i, j in enumerate(candidate):
                a1 = transformed_q1_sample_at_q2_density_all + j - transformed_q1_sample_at_transformed_q1_density_all
                a2 = q2_sample_at_transformed_q1_density_all - j - q2_sample_at_q2_density_all
                h_candidate[i] = 2 - torch.exp(0.5*a1).mean() - torch.exp(0.5*a2).mean()
            h_estimate = torch.max(h_candidate).cpu().item()
            final_estimate = h_estimate
        if fdiv == "harmonic":
            # estimate Harmonic
            ha_candidate = torch.zeros(5000, device=device)
            for i, j in enumerate(candidate):
                a1 = F.logsigmoid(torch.log((1-s1)/s1) + transformed_q1_sample_at_q2_density_all + j -
                                  transformed_q1_sample_at_transformed_q1_density_all)
                a2 = F.logsigmoid(torch.log(s1/(1-s1)) + q2_sample_at_transformed_q1_density_all -
                                  q2_sample_at_q2_density_all - j)
                ha_candidate[i] = 1 - torch.exp(2*a1).mean()/(1-s1) - torch.exp(2*a2).mean()/s1
            ha_estimate = torch.max(ha_candidate).cpu().item()
            final_estimate = ha_estimate
        # return {"JS": js_estimate, "Hellinger": h_estimate, "Harmonic": ha_estimate}
        return {fdiv: final_estimate}

    def generate_plots(self, truth=None, lim=20):
        # same as before, get all ingredients first
        self.myGenerator.eval()
        with torch.no_grad():
            q2_sample_at_q2_density_all = self.my_target_density(self.p2_estimating)
            q2_sample_at_transformed_q1_density_all = self.myGenerator.log_prob(self.p2_estimating)
            aug_q1_transformed_all, aug_q1_transformed_det_all = self.myGenerator.inverse(self.augmented_p1_estimating)
            transformed_q1_sample_at_q2_density_all = self.my_target_density(aug_q1_transformed_all)
            transformed_q1_sample_at_transformed_q1_density_all = \
                self.myGenerator.augmented_base_density(self.augmented_p1_estimating) - \
                torch.sum(aug_q1_transformed_det_all, dim=1)
        lossss_js = torch.zeros(1000)  # we want this since optimal Bridge should minimize weighted JS divergence
        lossss_ha = torch.zeros(1000)  # best transformated q1 should minimizes the weighted harmonic divergence
        # and the maximizer is also a Bridge estimate of Bayes factor
        bbb = torch.linspace(self.fitted_r-lim, self.fitted_r+lim, 1000, device=self.device)
        s1 = torch.tensor(self.n1/(self.n1+self.n2), device=self.device)
        for i, b in enumerate(bbb):
            with torch.no_grad():
                lossss_js[i] = loss_generator_js(transformed_q1_sample_at_transformed_q1_density_all,
                                                 transformed_q1_sample_at_q2_density_all,
                                                 q2_sample_at_transformed_q1_density_all,
                                                 q2_sample_at_q2_density_all, b, s1).cpu()
                lossss_ha[i] = loss_generator_harmonic(transformed_q1_sample_at_transformed_q1_density_all,
                                                       transformed_q1_sample_at_q2_density_all,
                                                       q2_sample_at_transformed_q1_density_all,
                                                       q2_sample_at_q2_density_all, b, s1).cpu()
        plt.subplot(1, 2, 1)
        plt.plot(bbb.cpu(), lossss_js)
        plt.axvline(x=self.fitted_r, label="Estimated", c="k")
        if truth is not None:
            plt.axvline(x=truth, label="Truth", c="r")
            print("MSE of log r is {}".format((self.fitted_r-truth)**2))
        plt.legend(loc="upper right")
        plt.subplot(1, 2, 2)
        plt.plot(bbb.cpu(), lossss_ha)
        plt.axvline(x=self.fitted_r, label="Estimated", c="k")
        if truth is not None:
            plt.axvline(x=truth, label="Truth", c="r")
        plt.legend(loc="upper right")

    def new_bridge(self, q1_sample, q2_sample):
        new_n1 = q1_sample.shape[0]
        new_n2 = q2_sample.shape[0]
        if self.dim_diff != 0:
            q1_sample = np.c_[q1_sample, np.random.randn(new_n1, self.dim_diff)]
        q1_sample = torch.tensor(q1_sample, device=device, dtype=torch.float)
        q2_sample = torch.tensor(q2_sample, device=device, dtype=torch.float)

        self.myGenerator.eval()
        with torch.no_grad():
            q2_sample_at_q2_density_all = self.my_target_density(q2_sample)
            q2_sample_at_transformed_q1_density_all = self.myGenerator.log_prob(q2_sample)
            aug_q1_transformed_all, aug_q1_transformed_det_all = self.myGenerator.inverse(q1_sample)
            transformed_q1_sample_at_q2_density_all = self.my_target_density(aug_q1_transformed_all)
            transformed_q1_sample_at_transformed_q1_density_all = \
                self.myGenerator.augmented_base_density(q1_sample) - \
                torch.sum(aug_q1_transformed_det_all, dim=1)

        s1 = torch.tensor(new_n1/(new_n2+new_n1), dtype=torch.float, device=device)
        old_r = self.raw_r
        rel_diff = 1.
        curr_t = 0
        while curr_t < 1500 and rel_diff > 1e-4:
            l1 = q2_sample_at_transformed_q1_density_all - q2_sample_at_q2_density_all + \
                 torch.log(s1/(1-s1)) - old_r
            l2 = transformed_q1_sample_at_transformed_q1_density_all - transformed_q1_sample_at_q2_density_all - \
                 old_r + torch.log(s1/(1-s1))
            l1 = F.logsigmoid(l1)
            l2 = F.logsigmoid(-1*l2)
            l1max = torch.max(l1)
            l1 = l1 - l1max
            l2max = torch.max(l2)
            l2 = l2 - l2max
            part1 = l1max + torch.log(torch.sum(torch.exp(l1)))
            part2 = l2max + torch.log(torch.sum(torch.exp(l2)))
            new_r = old_r + torch.sum(part1) - torch.sum(part2)
            rel_diff = torch.abs(new_r-old_r)/torch.abs(old_r)
            rel_diff = rel_diff.cpu().item()
            old_r = new_r
            curr_t += 1

        if curr_t == 1500:
            print("optimal Bridge estimate did not converge, use with caution...")
            return old_r.cpu().item()
        else:
            return old_r.cpu().item()

    def new_bridge_alt(self, q1_sample, q2_sample):  # without doing everything on log scale, should yield same results
        new_n1 = q1_sample.shape[0]
        new_n2 = q2_sample.shape[0]
        if self.dim_diff != 0:
            q1_sample = np.c_[q1_sample, np.random.randn(new_n1, self.dim_diff)]
        q1_sample = torch.tensor(q1_sample, device=device, dtype=torch.float)
        q2_sample = torch.tensor(q2_sample, device=device, dtype=torch.float)

        self.myGenerator.eval()
        with torch.no_grad():
            q2_sample_at_q2_density_all = self.my_target_density(q2_sample)
            q2_sample_at_transformed_q1_density_all = self.myGenerator.log_prob(q2_sample)
            aug_q1_transformed_all, aug_q1_transformed_det_all = self.myGenerator.inverse(q1_sample)
            transformed_q1_sample_at_q2_density_all = self.my_target_density(aug_q1_transformed_all)
            transformed_q1_sample_at_transformed_q1_density_all = \
                self.myGenerator.augmented_base_density(q1_sample) - \
                torch.sum(aug_q1_transformed_det_all, dim=1)

        l1 = torch.exp(transformed_q1_sample_at_transformed_q1_density_all - transformed_q1_sample_at_q2_density_all)
        l2 = torch.exp(q2_sample_at_transformed_q1_density_all - q2_sample_at_q2_density_all)
        s1 = torch.tensor(new_n1/(new_n2+new_n1), dtype=torch.float, device=device)
        old_r = torch.exp(torch.tensor(self.fitted_r, dtype=torch.float, device=device))
        rel_diff = 1.
        curr_t = 0
        while curr_t < 1500 and rel_diff > 1e-7:
            new_r = torch.mean(l2/(s1*l2 + (1-s1)*old_r))/torch.mean(1/(s1*l1 + (1-s1)*old_r))
            rel_diff = torch.abs(new_r-old_r)/torch.abs(old_r)
            rel_diff = rel_diff.cpu().item()
            old_r = new_r
            curr_t += 1
        if curr_t == 1500:
            print("optimal Bridge estimate did not converge, use with caution...")
            return old_r.cpu().item()
        else:
            return old_r.cpu().item()
