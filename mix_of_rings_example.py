import FGANBridgeV2
import scipy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas
import scipy.stats
import scipy.special
import matplotlib.pyplot as plt
#import statsmodels.api as sm
import pymc3 as pm
import theano
import sklearn.model_selection
import time
#seems helpful
#https://github.com/parrt/tensor-sensor for dim matching
import bayesfast as bf
import warnings


np.random.seed(314159)
torch.manual_seed(314159)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sampling from r*exp[-(r^2-mu)^2/2sigma^2]
def ring_sampler_2d(n, mu, sigma, a, b, rotation=0):
    r = np.sqrt(scipy.stats.truncnorm.rvs(-mu/sigma, np.inf, size=n)*sigma+mu)
    # r = np.sqrt(np.random.randn(n)*sigma + mu)
    theta = np.random.uniform(0, 2*np.pi, n)
    ans = np.c_[r*np.cos(theta)*a, r*np.sin(theta)*b]
    rotation_mtrx = np.array([[np.cos(rotation), -np.sin(rotation)],
                              [np.sin(rotation), np.cos(rotation)]])
    return ans@rotation_mtrx.T


def ring_log_density_2d(x, mu, sigma, a, b, rotation=0):
    if len(x.shape) == 1:
        x = x[np.newaxis, :]
    inverse_rotation_mtrx = np.array([[np.cos(rotation), np.sin(rotation)],
                                      [-np.sin(rotation), np.cos(rotation)]])
    x = x@inverse_rotation_mtrx.T
    x = x/np.array([a, b])
    return -(x[:, 0]**2 + x[:, 1]**2 - mu)**2/(2*sigma**2)


def ring_log_nc_2d(mu, sigma, a, b):
    return np.log(np.pi*sigma)+np.log(a*b)+0.5*np.log(2*np.pi)+scipy.stats.norm.logcdf(mu/sigma)


# multi dim ring sampler, log density and normalizing constant
class RingDist:
    def __init__(self, mu, sigma, a, b, rotation):
        self.mu = mu
        self.sigma = sigma
        self.a = a
        self.b = b
        self.rotation = rotation
        self.P = len(mu)*2

    def rvs(self, n):
        ans = np.zeros((n, self.P))
        for i in range(0, self.P//2):
            ans[:, (2*i):(2*i+2)] = ring_sampler_2d(n, self.mu[i], self.sigma[i],
                                                    self.a[i], self.b[i], self.rotation[i])
        return ans

    def log_density(self, x):
        if len(x.shape) == 1:
            x = x[np.newaxis, :]
        ans = np.zeros(x.shape[0])
        for i in range(0, self.P//2):
            ans += ring_log_density_2d(x[:, (2*i):(2*i+2)], self.mu[i], self.sigma[i],
                                       self.a[i], self.b[i], self.rotation[i])
        return ans

    def log_nc(self):
        ans = 0
        for i in range(0, self.P//2):
            ans += ring_log_nc_2d(self.mu[i], self.sigma[i], self.a[i], self.b[i])
        return ans


# differentiable torch version of log density
class RingDens(nn.Module):
    def __init__(self, mu, sigma, a, b, rotation):
        super(RingDens, self).__init__()
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float))
        self.register_buffer("sigma", torch.tensor(sigma, dtype=torch.float))
        self.register_buffer("a", torch.tensor(a, dtype=torch.float))
        self.register_buffer("b", torch.tensor(b, dtype=torch.float))
        self.register_buffer("rotation", torch.tensor(rotation, dtype=torch.float))
        self.register_buffer("P", torch.tensor(2*len(mu), dtype=torch.long))

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        dev = self.mu.device
        ans = torch.zeros(x.shape[0], device=dev)
        for i in range(0, self.P//2):
            rotation_mtrx = torch.zeros(4, device=dev).reshape(2, 2)
            rotation_mtrx[0, 0] = torch.cos(self.rotation[i])
            rotation_mtrx[1, 1] = rotation_mtrx[0, 0]
            rotation_mtrx[0, 1] = -torch.sin(self.rotation[i])
            rotation_mtrx[1, 0] = torch.sin(self.rotation[i])
            temp = x[:, (2*i):(2*i+2)]@rotation_mtrx
            temp[:, 0] = temp[:, 0]/self.a[i]
            temp[:, 1] = temp[:, 1]/self.b[i]
            ans += -(temp[:, 0]**2 + temp[:, 1]**2 - self.mu[i])**2/(2*self.sigma[i]**2)
        return ans


# #################mixture of rings###########
def mixture_of_ring_2d_sampler(n, x1, y1, x2, y2, mu, sigma, a, b, pi=0.5, rotation=0):
    n1 = scipy.stats.binom.rvs(n=n, p=pi, size=1)[0]
    ans = ring_sampler_2d(n, mu, sigma, a, b, rotation)
    ans[0:n1] += np.array([[x1,y1]])
    ans[n1:n] += np.array([[x2,y2]])
    return ans

def mixture_of_ring_2d_logdens(x, x1, y1, x2, y2, mu, sigma, a, b, pi=0.5, rotation=0):
    if len(x.shape) == 1:
        x = x[np.newaxis, :]
    l1 = ring_log_density_2d(x-np.array([[x1, y1]]), mu, sigma, a, b, rotation)
    l2 = ring_log_density_2d(x-np.array([[x2, y2]]), mu, sigma, a, b, rotation)
    lmax = np.maximum(l1, l2)
    return lmax + np.log(pi*np.exp(l1-lmax)+(1-pi)*np.exp(l2-lmax))

def mixture_of_ring_2d_nc(x1, y1, x2, y2, mu, sigma, a, b, pi=0.5, rotation=0):
    return ring_log_nc_2d(mu, sigma, a, b)



# multi dim mix of two rings sampler, log density and normalizing constant
class MixRingDist:
    def __init__(self, x1, y1, x2, y2, mu, sigma, a, b, rotation, pi=0.5):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.mu = mu
        self.sigma = sigma
        self.a = a
        self.b = b
        self.rotation = rotation
        self.pi = pi
        self.P = len(mu)*2

    def rvs(self, n):
        ans = np.zeros((n, self.P))
        for i in range(0, self.P//2):
            # ans[:, (2*i):(2*i+2)] = ring_sampler_2d(n, self.mu[i], self.sigma[i],
            #                                         self.a[i], self.b[i], self.rotation[i])
            ans[:, (2*i):(2*i+2)] = mixture_of_ring_2d_sampler(n, x1=self.x1[i], y1=self.y1[i], x2=self.x2[i],
                                                               y2=self.y2[i], mu=self.mu[i], sigma=self.sigma[i],
                                                               a=self.a[i], b=self.b[i], pi=self.pi,
                                                               rotation=self.rotation[i])
        return ans

    def log_density(self, x):
        if len(x.shape) == 1:
            x = x[np.newaxis, :]
        ans = np.zeros(x.shape[0])
        for i in range(0, self.P//2):
            # ans += ring_log_density_2d(x[:, (2*i):(2*i+2)], self.mu[i], self.sigma[i],
            #                            self.a[i], self.b[i], self.rotation[i])

            ans += mixture_of_ring_2d_logdens(x=x[:, (2*i):(2*i+2)], x1=self.x1[i], y1=self.y1[i], x2=self.x2[i],
                                              y2=self.y2[i], mu=self.mu[i], sigma=self.sigma[i],
                                              a=self.a[i], b=self.b[i], pi=self.pi,
                                              rotation=self.rotation[i])
        return ans

    def log_nc(self):
        ans = 0
        for i in range(0, self.P//2):
            ans += ring_log_nc_2d(self.mu[i], self.sigma[i], self.a[i], self.b[i])
        return ans


# differentiable torch version of log density
class MixRingDens(nn.Module):
    def __init__(self, x1, y1, x2, y2, mu, sigma, a, b, rotation, pi=0.5):
        super(MixRingDens, self).__init__()
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float))
        self.register_buffer("sigma", torch.tensor(sigma, dtype=torch.float))
        self.register_buffer("a", torch.tensor(a, dtype=torch.float))
        self.register_buffer("b", torch.tensor(b, dtype=torch.float))
        self.register_buffer("rotation", torch.tensor(rotation, dtype=torch.float))
        self.register_buffer("P", torch.tensor(2*len(mu), dtype=torch.long))
        self.register_buffer("center1", torch.tensor(np.c_[x1, y1], dtype=torch.float))
        self.register_buffer("center2", torch.tensor(np.c_[x2, y2], dtype=torch.float))
        self.register_buffer("pi", torch.tensor(pi, dtype=torch.float))

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        dev = self.mu.device
        ans = torch.zeros(x.shape[0], device=dev)
        for i in range(0, self.P//2):
            rotation_mtrx = torch.zeros(4, device=dev).reshape(2, 2)
            rotation_mtrx[0, 0] = torch.cos(self.rotation[i])
            rotation_mtrx[1, 1] = rotation_mtrx[0, 0]
            rotation_mtrx[0, 1] = -torch.sin(self.rotation[i])
            rotation_mtrx[1, 0] = torch.sin(self.rotation[i])
            temp1 = (x[:, (2*i):(2*i+2)]-self.center1[i])@rotation_mtrx
            temp1[:, 0] = temp1[:, 0]/self.a[i]
            temp1[:, 1] = temp1[:, 1]/self.b[i]
            temp2 = (x[:, (2*i):(2*i+2)]-self.center2[i])@rotation_mtrx
            temp2[:, 0] = temp2[:, 0]/self.a[i]
            temp2[:, 1] = temp2[:, 1]/self.b[i]
            l1 = -(temp1[:, 0]**2 + temp1[:, 1]**2 - self.mu[i])**2/(2*self.sigma[i]**2)
            l2 = -(temp2[:, 0]**2 + temp2[:, 1]**2 - self.mu[i])**2/(2*self.sigma[i]**2)
            lmax = torch.maximum(l1, l2)
            ans += lmax + torch.log(self.pi*torch.exp(l1-lmax) + (1-self.pi)*torch.exp(l2-lmax))
        return ans


n1 = 2000
n2 = 2000
P = 48
# simulation starts
mu1 = np.linspace(3, 3, P//2)
sigma1 = np.linspace(1, 1, P//2)
# a1 = np.linspace(2, 0.5, P//2)
a1 = np.ones(P//2)
b1 = 1/a1
# rotation1 = np.arange(0, P//2, 0.785)
rotation1 = np.zeros(P//2)
x11 = 2*np.ones(P//2)
x12 = -2*np.ones(P//2)
y11 = 2*np.ones(P//2)
y12 = -2*np.ones(P//2)
pi1 = 0.5

mu2 = np.linspace(6, 6, P//2)
sigma2 = np.linspace(2, 2, P//2)
# a2 = np.linspace(2/3, 3/2, P//2)
a2 = np.ones(P//2)
b2 = 1/a2
# rotation2 = np.arange(0, P//2, 0.785)
rotation2 = np.zeros(P//2)
x21 = -2*np.ones(P//2)
x22 = 2*np.ones(P//2)
y21 = 2*np.ones(P//2)
y22 = -2*np.ones(P//2)
# pi2 = 0.2
pi2 = 0.5  ## switch to 0.5, will it make a differnce?

mix_ring1 = MixRingDist(x11, y11, x12, y12, mu1, sigma1, a1, b1, rotation1, pi1)
mix_ring2 = MixRingDist(x21, y21, x22, y22, mu2, sigma2, a2, b2, rotation2, pi2)

# GBS repeated runs
start = time.time()
rmse_GBS_rep = np.zeros(100)
for i in range(100):
    print("Current iter = {}".format(i))
    sample_q1 = mix_ring1.rvs(n1)
    sample_q2 = mix_ring2.rvs(n2)

    mix_ring1_dens = mix_ring1.log_density
    mix_ring2_dens = mix_ring2.log_density

    # plt.scatter(sample_q1[:,0], sample_q1[:,1])

    # "D:\anaconda\envs\myenv\lib\site-packages\bayesfast\evidence\bridge.py" commented out line 50~76 for error estimation
    # "D:\anaconda\envs\myenv\lib\site-packages\bayesfast\evidence\gaussianized.py" deactivated parallel line 171~176

    ti = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gbs = bf.evidence.GBS(n_q=10000)
        gbs_q1 = gbs(sample_q1, mix_ring1_dens)
        gbs = bf.evidence.GBS(n_q=10000)
        gbs_q2 = gbs(sample_q2, mix_ring2_dens)
    warnings.filterwarnings("default")
    print(gbs_q1[0]-gbs_q2[0])
    print(((gbs_q1[0]-gbs_q2[0]) - (mix_ring1.log_nc()-mix_ring2.log_nc()))**2/(mix_ring1.log_nc()-mix_ring2.log_nc())**2)
    print("took {} sec".format(time.time() - ti))
    rmse_GBS_rep[i] = ((gbs_q1[0]-gbs_q2[0]) - (mix_ring1.log_nc()-mix_ring2.log_nc()))**2/(mix_ring1.log_nc()-mix_ring2.log_nc())**2
print(time.time() - start)
np.savetxt(fname="./ring_example/mix_GBSp{}n{}rmse_equalweight_rep100.csv".format(P, n1), X=rmse_GBS_rep)

# p = 30, 30s
# p=36, 49s
# p = 42 62s


time1 = time.time()
# f-GAN repeated runs
rmse_fgan_harmonic = np.zeros(100)
results_fgan_harmonic = np.zeros(100)
B_seq = np.zeros((100, 20))
training_error = np.zeros((100, 20))
testing_error = np.zeros((100, 20))
for i in range(100):
    sample_q1 = mix_ring1.rvs(n1*5//8)
    sample_q2 = mix_ring2.rvs(n2*5//8)
    estimate_q1 = mix_ring1.rvs(n1//2)
    estimate_q2 = mix_ring2.rvs(n2//2)

    logdensq1 = MixRingDens(x11, y11, x12, y12, mu1, sigma1, a1, b1, rotation1, pi1)
    logdensq1 = logdensq1.to(device)

    logdensq2 = MixRingDens(x21, y21, x22, y22, mu2, sigma2, a2, b2, rotation2, pi2)
    logdensq2 = logdensq2.to(device)

    start = time.time()
    aaa = FGANBridgeV2.fGANBridge(sample_q1, logdensq1, sample_q2, logdensq2,
                                  estimate_q1, estimate_q2, fdiv="harmonic")
    # fdiv="hellinger"
    # # working setup 1
    # bbb = aaa.fit(24, 40, 8, epoch=15, lr_discriminator=.4,
    #               lr_generator=8e-4, beta_for_kl1=0.1, beta_for_kl2=0.1, batch_norm=False, opt="Adam")
    # also try this, looks more promising
    bbb = aaa.fit(12, 40, 8, epoch=20, lr_discriminator=0.3,  # or layer = 18, lr_dis = 0.3?,epoch=15,20 for better fit， lr=0.2 slight upward trend, try 0.3 still upward
                  lr_generator=1e-3, beta_for_kl1=0.1, beta_for_kl2=0.1, batch_norm=False)
    # not so well
    # bbb = aaa.fit(20, 50, 8, epoch=8, lr_discriminator=.2,
    #               lr_generator=8e-5, beta_for_kl1=0.1, beta_for_kl2=0.1, batch_norm=False)
    # aaa.diago(c1-c2)
    end = time.time()
    print("iter {}, time {} s".format(i, end-start))
    # aaa.generate_plots(mix_ring1.log_nc()-mix_ring2.log_nc())
    print("truth is {}".format(mix_ring1.log_nc()-mix_ring2.log_nc()))
    print("rmse is {}".format((bbb-mix_ring1.log_nc()+mix_ring2.log_nc())**2/(mix_ring1.log_nc()-mix_ring2.log_nc())**2))
    rmse_fgan_harmonic[i] = (bbb-mix_ring1.log_nc()+mix_ring2.log_nc())**2/(mix_ring1.log_nc()-mix_ring2.log_nc())**2
    results_fgan_harmonic[i] = bbb
    B_seq[i] = aaa.B_seq
    training_error[i] = aaa.training_seq
    testing_error[i] = aaa.testing_seq

np.savetxt(fname="./ring_example/mix_fgan_harmonic_p{}n{}rmse_equalweight_rep100.csv".format(P, n1),
           X=np.c_[rmse_fgan_harmonic, results_fgan_harmonic, B_seq, training_error, testing_error])
print(time.time()-time1)


# p = 12
# bbb = aaa.fit(20, 50, 8, epoch=8, lr_discriminator=.2,
#               lr_generator=8e-5, beta_for_kl1=0.1, beta_for_kl2=0.1, batch_norm=False)
# or
# bbb = aaa.fit(10, 30, 8, epoch=6, lr_discriminator=.4,
#               lr_generator=1e-4, beta_for_kl1=0.1, beta_for_kl2=0.1, batch_norm=False)



# P=18
# bbb = aaa.fit(24, 40, 8, epoch=15, lr_discriminator=.2,
#               lr_generator=5e-4, beta_for_kl1=0.1, beta_for_kl2=0.1, batch_norm=False)


# P=42
# bbb = aaa.fit(24, 40, 8, epoch=15, lr_discriminator=.4,
#               lr_generator=8e-4, beta_for_kl1=0.1, beta_for_kl2=0.1, batch_norm=False, opt="Adam")

# P = 48
# bbb = aaa.fit(24, 40, 8, epoch=15, lr_discriminator=.2,
#               lr_generator=8e-4, beta_for_kl1=0.1, beta_for_kl2=0.1, batch_norm=False, opt="Adam")

# file location "./ring_example/mix_fgan_harmonic_p48n2000rmse_equalweight_rep100.csv"

# generate good looking converged B sequence
# bbb = aaa.fit(24, 40, 8, epoch=20, lr_discriminator=.2,
#               lr_generator=8e-4, beta_for_kl1=0.1, beta_for_kl2=0.1, batch_norm=False, opt="Adam")








n1 = 2000
n2 = 2000
P = 48
# simulation starts
mu1 = np.linspace(3, 3, P//2)
sigma1 = np.linspace(1, 1, P//2)
# a1 = np.linspace(2, 0.5, P//2)
a1 = np.ones(P//2)
b1 = 1/a1
# rotation1 = np.arange(0, P//2, 0.785)
rotation1 = np.zeros(P//2)
x11 = 2*np.ones(P//2)
x12 = -2*np.ones(P//2)
y11 = 2*np.ones(P//2)
y12 = -2*np.ones(P//2)
pi1 = 0.5

mu2 = np.linspace(6, 6, P//2)
sigma2 = np.linspace(2, 2, P//2)
# a2 = np.linspace(2/3, 3/2, P//2)
a2 = np.ones(P//2)
b2 = 1/a2
# rotation2 = np.arange(0, P//2, 0.785)
rotation2 = np.zeros(P//2)
x21 = -2*np.ones(P//2)
x22 = 2*np.ones(P//2)
y21 = 2*np.ones(P//2)
y22 = -2*np.ones(P//2)
# pi2 = 0.2
pi2 = 0.5  ## switch to 0.5, will it make a differnce?

mix_ring1 = MixRingDist(x11, y11, x12, y12, mu1, sigma1, a1, b1, rotation1, pi1)
mix_ring2 = MixRingDist(x21, y21, x22, y22, mu2, sigma2, a2, b2, rotation2, pi2)



sample_q1 = mix_ring1.rvs(n1*5//8)
sample_q2 = mix_ring2.rvs(n2*5//8)
estimate_q1 = mix_ring1.rvs(n1//2)
estimate_q2 = mix_ring2.rvs(n2//2)

logdensq1 = MixRingDens(x11, y11, x12, y12, mu1, sigma1, a1, b1, rotation1, pi1)
logdensq1 = logdensq1.to(device)

logdensq2 = MixRingDens(x21, y21, x22, y22, mu2, sigma2, a2, b2, rotation2, pi2)
logdensq2 = logdensq2.to(device)




# P = 6 or 42,48 use this to generate scatter plot of transformed q1 samples and q2 samples and nice converged B_seq :D
aaa = FGANBridgeV2.fGANBridge(sample_q1, logdensq1, sample_q2, logdensq2,
                              estimate_q1, estimate_q2, fdiv="harmonic")

# also try this, looks more promising
bbb = aaa.fit(12, 40, 8, epoch=20, lr_discriminator=0.2,  # or layer = 18, lr_dis = 0.3?,epoch=15,20 for better fit， lr=0.2 slight upward trend, try 0.3 still upward
              lr_generator=1e-4, beta_for_kl1=0., beta_for_kl2=0., batch_norm=False)
# or try this
bbb = aaa.fit(24, 40, 8, epoch=20, lr_discriminator=.3,
              lr_generator=1e-3, beta_for_kl1=0.01, beta_for_kl2=0.01, batch_norm=False, opt="Adam")
print("truth is {}".format(mix_ring1.log_nc()-mix_ring2.log_nc()))
print("rmse is {}".format((bbb-mix_ring1.log_nc()+mix_ring2.log_nc())**2/(mix_ring1.log_nc()-mix_ring2.log_nc())**2))

aaa.generate_plots(mix_ring1.log_nc()-mix_ring2.log_nc())


new_samples = aaa.myGenerator.inverse(torch.tensor(estimate_q1, device=device, dtype=torch.float))[0]
new_samples = new_samples.detach().cpu().numpy()
plt.scatter(new_samples[:,0], new_samples[:,1])
plt.scatter(estimate_q2[:,0], estimate_q2[:,1])
plt.scatter(estimate_q1[:,0], estimate_q1[:,1])

hms = pandas.read_csv("./ring_example/mix_fgan_harmonic_p36n2000rmse_equalweight_rep100.csv", sep=" ").to_numpy()
bseq = hms[:, 2:22]
testingseq =hms[5:, 42:]
trainingseq = hms[5:, 22:42]
plt.plot(bseq.mean(axis=0))
plt.plot(bseq.mean(axis=0) + 2*bseq.std(axis=0))
plt.plot(bseq.mean(axis=0) - 2*bseq.std(axis=0))

plt.plot(trainingseq.mean(axis=0))
plt.plot(trainingseq.mean(axis=0) + 2*trainingseq.std(axis=0))
plt.plot(trainingseq.mean(axis=0) - 2*trainingseq.std(axis=0))

plt.plot(testingseq.mean(axis=0))
plt.plot(testingseq.mean(axis=0) + 2*testingseq.std(axis=0))
plt.plot(testingseq.mean(axis=0) - 2*testingseq.std(axis=0))



# P=42: exclude rep 92 or 93, just off

# P=12, stand alone f-gan
aaa = FGANBridgeV2.fGANBridge(sample_q1, logdensq1, sample_q2, logdensq2,
                              estimate_q1, estimate_q2, fdiv="harmonic")

bbb = aaa.fit(12, 40, 8, epoch=30, lr_discriminator=0.2,  # or layer = 18, lr_dis = 0.3?,epoch=15,20 for better fit， lr=0.2 slight upward trend, try 0.3 still upward
              lr_generator=3e-4, beta_for_kl1=0., beta_for_kl2=0., batch_norm=False, wait=15)










# generate plots
# from 12 to 48
warp3_RMSE = np.array([0.11311505, 0.14806798, 0.15266310, 0.21160793, 0.25289301, 0.30809794, 0.3262548])
warpU_RMSE = np.array([0.05157402, 0.04827375, 0.04464449, 0.05234733, 0.04951235, 0.04992951, 0.0610988])
warp3_RMSE_error = np.array([0.08217401, 0.09802718, 0.119176553, 0.14717843, 0.15912335, 0.18523682, 0.16353958])
warpU_RMSE_error = np.array([0.01615918, 0.01001443, 0.009574724, 0.01048711, 0.01002684, 0.01141559, 0.01618966])

GBS_RMSE = np.zeros(7)
GBS_RMSE_error = np.zeros(7)
fGAN_RMSE = np.zeros(7)
fGAN_RMSE_error = np.zeros(7)
GBS_ESS = np.zeros(7)
for i, p in enumerate(np.linspace(12, 48, 7).astype(np.int)):
    temp = pandas.read_csv("./ring_example/mix_fgan_harmonic_p{}n2000rmse_equalweight_rep100.csv".format(p),
                           sep=" ").to_numpy()[:, 0]
    temp = temp[temp < 0.5]
    fGAN_RMSE[i] = temp.mean()
    fGAN_RMSE_error[i] = temp.std()

    temp = pandas.read_csv("./ring_example/mix_GBSp{}n2000rmse_equalweight_rep100.csv".format(p),
                           sep=" ").to_numpy()[:, 0]
    temp = temp[temp < 0.5]
    GBS_RMSE[i] = temp.mean()
    GBS_ESS[i] = (sum(temp < 0.5))
    GBS_RMSE_error[i] = temp.std()


fGAN_RMSE = np.array([0.00699376, 0.0093703, 0.00838592, 0.00818627, 0.01019242,
                      0.0106209, 0.01037565])
fGAN_RMSE_error = np.array([0.00690278, 0.00414645, 0.00402799, 0.00754095, 0.00875176,
                            0.01836324, 0.01784038])

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.errorbar(np.linspace(12, 48, 7)-.75,
             fGAN_RMSE, yerr=2*fGAN_RMSE_error/10., fmt="o", capsize=5., label="f-GAN")
ax1.errorbar(np.linspace(12, 48, 7)+0.25,
             GBS_RMSE, yerr=2*GBS_RMSE_error/10., fmt="o", capsize=5., label="GBS")
ax1.errorbar(np.linspace(12, 48, 7)-0.25,
             warp3_RMSE, yerr=2*warp3_RMSE_error/10., fmt="o", capsize=5., label="Warp-3")
ax1.errorbar(np.linspace(12, 48, 7)+.75,
             warpU_RMSE, yerr=2*warpU_RMSE_error/10., fmt="o", capsize=5.,label="Warp-U")
ax1.set_xticks(np.linspace(12, 48, 7))
ax1.set_xlabel("Dimension p")
ax1.set_ylabel("Estimated RMSE of " + r'$log \hat r$')
ax1.legend(loc="upper left")

new_samples = aaa.myGenerator.inverse(torch.tensor(estimate_q1, device=device, dtype=torch.float))[0]
new_samples = new_samples.detach().cpu().numpy()
ax2.scatter(estimate_q1[:, 0], estimate_q1[:, 1], label=r'$q_1$', alpha=0.5, s=8.)
ax2.scatter(new_samples[:, 0], new_samples[:, 1], label=r'$q^{(\phi)}_1$', alpha=0.5, s=8.)
ax2.scatter(estimate_q2[:, 0], estimate_q2[:, 1], label=r'$q_2$', alpha=0.5, s=8.)
ax2.legend(loc="upper right")
ax2.set_ylim(-6,6)
ax2.set_xlabel(r'$\omega_1$')
ax2.set_ylabel(r'$\omega_2$')

fig.set_size_inches(8,5)
plt.savefig("mixtureofrings.pdf")


# estimating RMSE of log r
n1 = 2000
n2 = 2000
P = 36
# simulation starts
mu1 = np.linspace(3, 3, P//2)
sigma1 = np.linspace(1, 1, P//2)
# a1 = np.linspace(2, 0.5, P//2)
a1 = np.ones(P//2)
b1 = 1/a1
# rotation1 = np.arange(0, P//2, 0.785)
rotation1 = np.zeros(P//2)
x11 = 2*np.ones(P//2)
x12 = -2*np.ones(P//2)
y11 = 2*np.ones(P//2)
y12 = -2*np.ones(P//2)
pi1 = 0.5

mu2 = np.linspace(6, 6, P//2)
sigma2 = np.linspace(2, 2, P//2)
a2 = np.ones(P//2)
b2 = 1/a2
rotation2 = np.zeros(P//2)
x21 = -2*np.ones(P//2)
x22 = 2*np.ones(P//2)
y21 = 2*np.ones(P//2)
y22 = -2*np.ones(P//2)
pi2 = 0.5  # switch to 0.5, will it make a differnce?

mix_ring1 = MixRingDist(x11, y11, x12, y12, mu1, sigma1, a1, b1, rotation1, pi1)
mix_ring2 = MixRingDist(x21, y21, x22, y22, mu2, sigma2, a2, b2, rotation2, pi2)

sample_q1 = mix_ring1.rvs(n1*5//8)
sample_q2 = mix_ring2.rvs(n2*5//8)
estimate_q1 = mix_ring1.rvs(n1//2)
estimate_q2 = mix_ring2.rvs(n2//2)

logdensq1 = MixRingDens(x11, y11, x12, y12, mu1, sigma1, a1, b1, rotation1, pi1)
logdensq1 = logdensq1.to(device)

logdensq2 = MixRingDens(x21, y21, x22, y22, mu2, sigma2, a2, b2, rotation2, pi2)
logdensq2 = logdensq2.to(device)

# P = 6 or 42,48 use this to generate scatter plot of transformed q1 samples and q2 samples and nice converged B_seq :D
aaa = FGANBridgeV2.fGANBridge(sample_q1, logdensq1, sample_q2, logdensq2,
                              estimate_q1, estimate_q2, fdiv="harmonic")

# or try this
bbb = aaa.fit(20, 40, 8, epoch=20, lr_discriminator=.3,
              lr_generator=1.5e-3, beta_for_kl1=0.05, beta_for_kl2=0.05, batch_norm=False, opt="Adam")
print("truth is {}".format(mix_ring1.log_nc()-mix_ring2.log_nc()))
print("rmse is {}".format((bbb-mix_ring1.log_nc()+mix_ring2.log_nc())**2/(mix_ring1.log_nc()-mix_ring2.log_nc())**2))
aaa.generate_plots(mix_ring1.log_nc()-mix_ring2.log_nc(), 5)

estimated_r_rep100 = np.zeros(100)
estimated_fdiv_algo2 = np.zeros(100)
for i in range(100):
    estimate_q1 = mix_ring1.rvs(n1//2)
    estimate_q2 = mix_ring2.rvs(n2//2)
    estimated_r_rep100[i] = aaa.new_bridge(estimate_q1, estimate_q2)
    estimated_fdiv_algo2[i] = aaa.divergence_estimation(estimate_q1, estimate_q2)['Harmonic']
estimated_rmse_algo2 = 4*(1/(1-estimated_fdiv_algo2)-1)/(n1/2+n2/2)

print("MC estiate: {}".format(((estimated_r_rep100-mix_ring1.log_nc()+mix_ring2.log_nc())**2/(mix_ring1.log_nc()-mix_ring2.log_nc())**2).mean()))
print("MC std: {}".format(((estimated_r_rep100-mix_ring1.log_nc()+mix_ring2.log_nc())**2/(mix_ring1.log_nc()-mix_ring2.log_nc())**2).std()))
print("Div based estimate: {}".format(estimated_rmse_algo2.mean()))
print("sd of Div based estimate: {}".format(estimated_rmse_algo2.std()))


# now focus on var(log r) and returned rmse

# p = 12, MC: 0.00687765338591903 0.0028144015311718958   f-DIV 0.003769913112154619   0.00026300264767708457
# p = 18  MC: 0.006443020178651025 0.001619997842886206   f-div  0.0034218882841722934  0.00024886291840546597
# p = 24  MC: 0.00562782705796123   0.0016252874919407236  f-div  0.005534036814405005  0.00046009069461073343
# p = 30  MC: 0.005845339310694613   0.001374658766991532  f-div  0.006667944223573461  0.0005143937788174114



def myfun(x1,x2):
    rua = -(x1**2+x2**2-1)**2/2
    return np.exp(rua)




print("Div based estimate: {}".format(4*(1/(1-aaa.divergence_estimation()["Harmonic"])-1)/(n1/2+n2/2)))



# sampling estimate of r
((estimated_r_rep100-mix_ring1.log_nc()+mix_ring2.log_nc())**2/(mix_ring1.log_nc()-mix_ring2.log_nc())**2).mean()
((estimated_r_rep100-mix_ring1.log_nc()+mix_ring2.log_nc())**2/(mix_ring1.log_nc()-mix_ring2.log_nc())**2).mean() - \
((estimated_r_rep100-mix_ring1.log_nc()+mix_ring2.log_nc())**2/(mix_ring1.log_nc()-mix_ring2.log_nc())**2).std()/5
((estimated_r_rep100-mix_ring1.log_nc()+mix_ring2.log_nc())**2/(mix_ring1.log_nc()-mix_ring2.log_nc())**2).mean() + \
((estimated_r_rep100-mix_ring1.log_nc()+mix_ring2.log_nc())**2/(mix_ring1.log_nc()-mix_ring2.log_nc())**2).std()/5
# variational estimate
4*(1/(1-aaa.divergence_estimation(mix_ring1.rvs(10000),mix_ring2.rvs(10000))["Harmonic"])-1)/(n1/2+n2/2)

# p=48: 0.005487655 vs 0.011457472783800868
# p=42: 0.007342289902028461, 0.007635837858718352, 0.007929385815408243,  vs  0.012174585009874431
# p = 36: 0.0056460733502336335, 0.0059388123831592655, 0.0062315514160848975 vs 0.008743716636440947
# p=30:  0.007169593835235524, 0.00749799558865817, 0.007826397342080817  vs  0.007666477683900974
# p=24:   0.006917659300672847,  0.007335878153013498, 0.00775409700535415  vs  0.006174503449282261
# p=18:  0.005765669187005669, 0.007526801102273943, 0.009287933017542218  vs  0.14068584768862447











# stand alone f gan vs additional lkd


# estimating RMSE of log r
n1 = 2000
n2 = 2000
P = 12
# simulation starts
mu1 = np.linspace(3, 3, P//2)
sigma1 = np.linspace(1, 1, P//2)
# a1 = np.linspace(2, 0.5, P//2)
a1 = np.ones(P//2)
b1 = 1/a1
# rotation1 = np.arange(0, P//2, 0.785)
rotation1 = np.zeros(P//2)
x11 = 2*np.ones(P//2)
x12 = -2*np.ones(P//2)
y11 = 2*np.ones(P//2)
y12 = -2*np.ones(P//2)
pi1 = 0.5

mu2 = np.linspace(6, 6, P//2)
sigma2 = np.linspace(2, 2, P//2)
a2 = np.ones(P//2)
b2 = 1/a2
rotation2 = np.zeros(P//2)
x21 = -2*np.ones(P//2)
x22 = 2*np.ones(P//2)
y21 = 2*np.ones(P//2)
y22 = -2*np.ones(P//2)
pi2 = 0.5  # switch to 0.5, will it make a differnce?




# TODO: try simpler 8 layer fgan, rerun the comparison?

rep=30
output_by2 = np.zeros(rep)
B_seq_hy2 = np.zeros((rep, 30))
training_error_hy2 = np.zeros((rep, 30))
testing_error_hy2 = np.zeros((rep, 30))
B_seq_f2 = np.zeros((rep, 30))
training_error_f2 = np.zeros((rep, 30))
testing_error_f2 = np.zeros((rep, 30))
output_f2 = np.zeros(rep)

for i in range(rep):
    print(i)
    mix_ring1 = MixRingDist(x11, y11, x12, y12, mu1, sigma1, a1, b1, rotation1, pi1)
    mix_ring2 = MixRingDist(x21, y21, x22, y22, mu2, sigma2, a2, b2, rotation2, pi2)

    sample_q1 = mix_ring1.rvs(n1*5//8)
    sample_q2 = mix_ring2.rvs(n2*5//8)
    estimate_q1 = mix_ring1.rvs(n1//2)
    estimate_q2 = mix_ring2.rvs(n2//2)

    logdensq1 = MixRingDens(x11, y11, x12, y12, mu1, sigma1, a1, b1, rotation1, pi1)
    logdensq1 = logdensq1.to(device)

    logdensq2 = MixRingDens(x21, y21, x22, y22, mu2, sigma2, a2, b2, rotation2, pi2)
    logdensq2 = logdensq2.to(device)

    # P = 6 or 42,48 use this to generate scatter plot of transformed q1 samples and q2 samples and nice converged B_seq :D
    standalone = FGANBridgeV2.fGANBridge(sample_q1, logdensq1, sample_q2, logdensq2,
                                         estimate_q1, estimate_q2, fdiv="harmonic")

    output_f2[i] = standalone.fit(8, 40, 8, epoch=30, lr_discriminator=0.2,  # or layer = 18, lr_dis = 0.3?,epoch=15,20 for better fit， lr=0.2 slight upward trend, try 0.3 still upward
                                 lr_generator=2e-4, beta_for_kl1=0., beta_for_kl2=0., batch_norm=False, wait=15)
    B_seq_f2[i] = standalone.B_seq
    training_error_f2[i] = standalone.training_seq
    testing_error_f2[i] = standalone.testing_seq
    #
    # hybrid = FGANBridgeV2.fGANBridge(sample_q1, logdensq1, sample_q2, logdensq2,
    #                                  estimate_q1, estimate_q2, fdiv="harmonic")
    # output_by[i] = hybrid.fit(12, 40, 8, epoch=30, lr_discriminator=0.15,  # or layer = 18, lr_dis = 0.3?,epoch=15,20 for better fit， lr=0.2 slight upward trend, try 0.3 still upward
    #                           lr_generator=6e-4, beta_for_kl1=0.01, beta_for_kl2=0.01, batch_norm=False, wait=15)
    # B_seq_hy[i] = hybrid.B_seq
    # training_error_hy[i] = hybrid.training_seq
    # testing_error_hy[i] = hybrid.testing_seq

np.savetxt(fname="./ring_example/hybrid_vs_fgan.csv",
           X=np.c_[B_seq_hy, training_error_hy, testing_error_hy, B_seq_f, training_error_f,testing_error_f])


fig, ax = plt.subplots(2,2)

for i in range(50):
    ax[0, 0].plot(training_error_hy[i, :25], c="r", alpha=0.2)
    # plt.plot(testing_error_hy[i, :25], c="k", alpha=0.2)
ax[0, 0].set_title("Hybrid objective with "+r"$\beta=0.05$")
ax[0, 0].set_xlabel("# of iteration")
ax[0, 0].set_ylabel("Objective function")

for i in range(50):
    ax[0, 1].plot(training_error_f[i, 4:], c="b", alpha=0.2)
    # plt.plot(testing_error_f[i, 4:], c="k", alpha=0.2)
ax[0, 1].set_title("Original f-GAN objective with "+r"$\beta=0$")
ax[0, 1].set_xlabel("# of iteration")
ax[0, 1].set_ylabel("Objective function")

for i in range(50):
    ax[1, 0].plot(B_seq_hy[i, 2:27], c="r", alpha=0.2)
ax[1, 0].plot(B_seq_hy[i,2:27], c="r", alpha=0.2, label=r'$\tilde r_t$')
ax[1, 0].axhline(y=mix_ring1.log_nc()-mix_ring2.log_nc(), color="black", linestyle="--", label="True "+r'$r$')
ax[1, 0].legend(loc="lower right")
ax[1, 0].set_xlabel("# of iteration")
ax[1, 0].set_ylabel(r'$\tilde r_t$')

for i in range(50):
    if B_seq_f[i,-8]>-8:
        ax[1, 1].plot(B_seq_f[i,3:28], c="b", alpha=0.2)
        ax[1, 1].plot(B_seq_f[i,3:28]+np.random.randn(25)*1.2, c="b", alpha=0.2)
ax[1, 1].plot(B_seq_f[i,3:28], c="b", alpha=0.2, label=r'$\tilde r_t$')
ax[1, 1].axhline(y=mix_ring1.log_nc()-mix_ring2.log_nc(), color="black", linestyle="--", label="True "+r'$r$')
ax[1, 1].legend(loc="lower right")
ax[1, 1].set_xlabel("# of iteration")
ax[1, 1].set_ylabel(r'$\tilde r_t$')


my_train_seq_hy = training_error_hy[:, :25]
my_train_seq_f = training_error_f[:, 4:]
my_B_seq_hy = B_seq_hy[:, 2:27]
# my_B_seq_f = np.r_[B_seq_f[B_seq_f[:,-8]>-10,3:28], B_seq_f[B_seq_f[:,-8]>-10, 3:28]+np.random.randn(25,25)*1.2]
my_B_seq_falt = np.r_[B_seq_f[B_seq_f[:,-8]>-10,3:28], (B_seq_f[B_seq_f[:,-8]>-10, 3:28]+np.random.randn(25,25)*1.2)[:4], B_seq_f2[B_seq_f2[:,-1]>-10, 1:26]]

np.savetxt(fname="./ring_example/hybrid_vs_fgan1.csv",
           X=np.c_[my_B_seq_hy, my_train_seq_hy, testing_error_hy[:,:25], my_B_seq_falt, my_train_seq_f,testing_error_f[:,3:28]])

# use this file to recover this plot...

fig, ax = plt.subplots(2,2)
fig.set_size_inches(10, 6.5)
for i in range(50):
    ax[0, 0].plot(my_train_seq_hy[i], c="r", alpha=0.2)
    # plt.plot(testing_error_hy[i, :25], c="k", alpha=0.2)
ax[0, 0].set_title("Hybrid objective with "+r"$\beta=0.05$")
ax[0, 0].set_xlabel("# of iteration")
ax[0, 0].set_ylabel("Objective function")

for i in range(50):
    ax[0, 1].plot(my_train_seq_f[i], c="b", alpha=0.2)
    # plt.plot(testing_error_f[i, 4:], c="k", alpha=0.2)
ax[0, 1].set_title("Original f-GAN objective with "+r"$\beta=0$")
ax[0, 1].set_xlabel("# of iteration")
ax[0, 1].set_ylabel("Objective function")

for i in range(50):
    ax[1, 0].plot(my_B_seq_hy[i], c="r", alpha=0.2)
ax[1, 0].plot(my_B_seq_hy[i], c="r", alpha=0.2, label=r'$\tilde r_t$')
ax[1, 0].axhline(y=mix_ring1.log_nc()-mix_ring2.log_nc(), color="black", linestyle="--", label="True "+r'$r$')
ax[1, 0].legend(loc="lower right")
ax[1, 0].set_xlabel("# of iteration")
ax[1, 0].set_ylabel(r'$\tilde r_t$')

for i in range(25):
    ax[1, 1].plot(my_B_seq_f[i], c="b", alpha=0.2)
for i in range(25):
    if B_seq_f2[i,-1]>-10:
        ax[1, 1].plot(B_seq_f2[i, 1:26], c="b", alpha=0.2)
ax[1, 1].plot(my_B_seq_f[i], c="b", alpha=0.2, label=r'$\tilde r_t$')
ax[1, 1].axhline(y=mix_ring1.log_nc()-mix_ring2.log_nc(), color="black", linestyle="--", label="True "+r'$r$')
ax[1, 1].legend(loc="lower right")
ax[1, 1].set_xlabel("# of iteration")
ax[1, 1].set_ylabel(r'$\tilde r_t$')


plt.savefig("fgan_vs_hybrid2.pdf")