import bridge_fGAN
import scipy
import numpy as np
import torch
import torch.nn as nn
import scipy.stats
import scipy.special
import matplotlib.pyplot as plt
import time

np.random.seed(314159)
torch.manual_seed(314159)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# #########################################################################
# ############### densities and samplers of mixture of rings ##############
# #########################################################################

def ring_sampler_2d(n, mu, sigma):
    r = np.sqrt(scipy.stats.truncnorm.rvs(-mu/sigma, np.inf, size=n)*sigma+mu)
    theta = np.random.uniform(0, 2*np.pi, n)
    ans = np.c_[r*np.cos(theta), r*np.sin(theta)]
    return ans


def ring_log_density_2d(x, mu, sigma):
    if len(x.shape) == 1:
        x = x[np.newaxis, :]
    return -(x[:, 0]**2 + x[:, 1]**2 - mu)**2/(2*sigma**2)


def ring_log_nc_2d(mu, sigma):
    return np.log(np.pi*sigma)+0.5*np.log(2*np.pi)+scipy.stats.norm.logcdf(mu/sigma)


# #################mixture of rings###########
def mixture_of_ring_2d_sampler(n, x1, y1, x2, y2, mu, sigma, pi=0.5):
    n1 = scipy.stats.binom.rvs(n=n, p=pi, size=1)[0]
    ans = ring_sampler_2d(n, mu, sigma)
    ans[0:n1] += np.array([[x1, y1]])
    ans[n1:n] += np.array([[x2, y2]])
    return ans


def mixture_of_ring_2d_logdens(x, x1, y1, x2, y2, mu, sigma, pi=0.5):
    if len(x.shape) == 1:
        x = x[np.newaxis, :]
    l1 = ring_log_density_2d(x-np.array([[x1, y1]]), mu, sigma)
    l2 = ring_log_density_2d(x-np.array([[x2, y2]]), mu, sigma)
    lmax = np.maximum(l1, l2)
    return lmax + np.log(pi*np.exp(l1-lmax)+(1-pi)*np.exp(l2-lmax))


def mixture_of_ring_2d_nc(x1, y1, x2, y2, mu, sigma, pi=0.5):
    return ring_log_nc_2d(mu, sigma)


# multi dim mix of two rings sampler, log density and normalizing constant
class MixRingDist:
    def __init__(self, x1, y1, x2, y2, mu, sigma, P, pi=0.5):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.mu = mu
        self.sigma = sigma
        self.pi = pi
        self.P = P
        self.dimm =np.int(P)

    def rvs(self, n):
        ans = np.zeros((n, self.dimm))
        for i in range(0, self.dimm//2):
            # ans[:, (2*i):(2*i+2)] = ring_sampler_2d(n, self.mu[i], self.sigma[i],
            #                                         self.a[i], self.b[i], self.rotation[i])
            ans[:, (2*i):(2*i+2)] = mixture_of_ring_2d_sampler(n, x1=self.x1, y1=self.y1, x2=self.x2,
                                                               y2=self.y2, mu=self.mu, sigma=self.sigma,
                                                               pi=self.pi)
        return ans

    def log_density(self, x):
        if len(x.shape) == 1:
            x = x[np.newaxis, :]
        ans = np.zeros(x.shape[0])
        for i in range(0, self.dimm//2):
            ans += mixture_of_ring_2d_logdens(x=x[:, (2*i):(2*i+2)], x1=self.x1, y1=self.y1, x2=self.x2,
                                              y2=self.y2, mu=self.mu, sigma=self.sigma,
                                              pi=self.pi)
        return ans

    def log_nc(self):
        return ring_log_nc_2d(self.mu, self.sigma)*(self.P/2)


# differentiable nn.Module version of log density
class MixRingDens(nn.Module):
    def __init__(self, x1, y1, x2, y2, mu, sigma, P, pi=0.5):
        super(MixRingDens, self).__init__()
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float))
        self.register_buffer("sigma", torch.tensor(sigma, dtype=torch.float))
        self.register_buffer("P", torch.tensor(P, dtype=torch.long))
        self.register_buffer("center1", torch.tensor(np.c_[x1, y1], dtype=torch.float))
        self.register_buffer("center2", torch.tensor(np.c_[x2, y2], dtype=torch.float))
        self.register_buffer("pi", torch.tensor(pi, dtype=torch.float))

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        dev = self.mu.device
        ans = torch.zeros(x.shape[0], device=dev)
        for i in range(0, self.P//2):
            temp1 = (x[:, (2*i):(2*i+2)]-self.center1)
            temp2 = (x[:, (2*i):(2*i+2)]-self.center2)
            l1 = -(temp1[:, 0]**2 + temp1[:, 1]**2 - self.mu)**2/(2*self.sigma**2)
            l2 = -(temp2[:, 0]**2 + temp2[:, 1]**2 - self.mu)**2/(2*self.sigma**2)
            lmax = torch.maximum(l1, l2)
            ans += lmax + torch.log(self.pi*torch.exp(l1-lmax) + (1-self.pi)*torch.exp(l2-lmax))
        return ans


# ################################################################
# ######################### set parameters #######################
# ################################################################

n1 = 2000
n2 = 2000
P = 36

mu1 = 3
sigma1 = 1
x11 = 2
x12 = -2
y11 = 2
y12 = -2
pi1 = 0.5

mu2 = 6
sigma2 = 2
x21 = -3
x22 = 3
y21 = 3
y22 = -3
pi2 = 0.5

mix_ring1 = MixRingDist(x11, y11, x12, y12, mu1, sigma1, P, pi1)
mix_ring2 = MixRingDist(x21, y21, x22, y22, mu2, sigma2, P, pi2)


# getting training and estimating samples for Algorithm 2

sample_q1 = mix_ring1.rvs(n1//2)
sample_q2 = mix_ring2.rvs(n2//2)
estimate_q1 = mix_ring1.rvs(n1//2)
estimate_q2 = mix_ring2.rvs(n2//2)

# define the log densities of q1,q2
logdensq1 = MixRingDens(x11, y11, x12, y12, mu1, sigma1, P, pi1)
logdensq1 = logdensq1.to(device)

logdensq2 = MixRingDens(x21, y21, x22, y22, mu2, sigma2, P, pi2)
logdensq2 = logdensq2.to(device)

# run Algorithm 2
start = time.time()
mixring_bf = bridge_fGAN.fGANBridge(sample_q1, logdensq1, sample_q2, logdensq2,
                                    estimate_q1, estimate_q2, fdiv="harmonic")

estimated_bf = mixring_bf.fit(12, 40, 8, epoch=25, lr_discriminator=0.2,
                              lr_generator=1e-3, beta_for_kl1=0.05, beta_for_kl2=0.05, batch_norm=False)
end = time.time()

print("estimated log r = {}".format(estimated_bf))
print("took {} s".format(end-start))
print("truth is {}".format(mix_ring1.log_nc()-mix_ring2.log_nc()))

# generate plots, left: classification lkd as a function of r right: the f-GAN objective as a function of r
mixring_bf.generate_plots(mix_ring1.log_nc()-mix_ring2.log_nc(), 3)

# getting the estimated ratio of NCs
print(mixring_bf.fitted_r)

# getting estimate RMSE of r or MSE of log r
print(mixring_bf.estimated_RMSE)

# plot the first two dimensions of the original and transformed samples
new_samples = mixring_bf.myGenerator.inverse(torch.tensor(estimate_q1, device=device, dtype=torch.float))[0]
new_samples = new_samples.detach().cpu().numpy()
plt.scatter(estimate_q1[:, 0], estimate_q1[:, 1], label=r'$q_1$', alpha=0.5, s=8.)
plt.scatter(new_samples[:, 0], new_samples[:, 1], label=r'$q^{(\phi)}_1$', alpha=0.5, s=8.)
plt.scatter(estimate_q2[:, 0], estimate_q2[:, 1], label=r'$q_2$', alpha=0.5, s=8.)
