# We have 3 angles, we need to get the distributions
import torch
from scipy import stats
from scipy.stats import gaussian_kde, t
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.neighbors import KernelDensity


class AngleDistr():
    def __init__(self, 
                 psi,
                 phi,
                 omega,
                 batch_size = 1,
                 norm=False,
                 log=True,
                 random=False,
                 mode = 'trans'):
        self.batch_size = batch_size
        self.norm = norm
        self.log = log
        self.random = random
        self.angles = None
        if self.random == False:
            self.flatten_angles(psi, phi, omega, mode)
        else:
            self.randinit()
        self.density = None
        self.title = 'Density Angles: Conf = ' + mode
        
        self.xmin = -180
        self.xmax = 180
        self.ymin = -180
        self.ymax = 180
        
    def flatten_angles(self, psi, phi, omega, mode='trans'):
        angles = []
        for p in range(len(psi)):
            ppsi, pphi, pomega = psi[p], phi[p], omega[p]
            for i in range(len(ppsi)):
                ippsi, ipphi, ipomega = ppsi[i], pphi[i], pomega[i]
                if not ((ippsi > 180) or (ipphi > 180) or (ipomega > 180)):
                    if (mode == 'trans') and (np.absolute(ipomega) > 100):
                        angles.append([ippsi, ipphi])
                    if (mode == 'cis') and (np.absolute(ipomega) < 100):
                        angles.append([ippsi, ipphi])
        self.angles = np.transpose(np.array(angles))

    def randinit(self):
        self.angles = t.rvs(5, size=(2, 200))
    
    def kde(self, bw_method=None):
        # Calculate the point density
        def bw_method_(obj, fac=1./5):
            """We use Scott's Rule, multiplied by a constant factor."""
            return np.power(obj.n, -1./(obj.d+4)) * fac
        if bw_method == None:
            bw_method = bw_method_
        self.kde = gaussian_kde(self.angles, bw_method=bw_method)
        """
        self.xmin = self.angles[0,:].min()
        self.xmax = self.angles[0,:].max()
        self.ymin = self.angles[1,:].min()
        self.ymax = self.angles[1,:].max()
        """
        X, Y = np.mgrid[self.xmin:self.xmax:100j, self.ymin:self.ymax:100j]
        angles = np.vstack([X.ravel(), Y.ravel()])
        self.ev_ang = self.kde.evaluate(angles).T
        if self.log:
            self.ev_ang = (0.01) * np.log(self.ev_ang)
        if self.norm:
            # Haven't thought this through -> not advised
            self.ev_ang = stats.norm.pdf(self.ev_ang)
        self.density = np.reshape(self.ev_ang, X.shape)
            
    def sample(self):
        if self.random:
            angle_list = [(torch.rand(2) * 2) - 1 for _ in range(self.batch_size)]
            return torch.stack(torch.Tensor(angle_list), axis=0)
        else:
            return torch.Tensor(np.transpose(self.kde.resample(self.batch_size)))
    
    def plot(self):
        """
        2D plot with psi and phi
        """
        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel('$psi$')
        ax.set_ylabel('$phi$')
        ax.set_title(self.title)
        pos = ax.imshow(np.rot90(self.density), cmap=plt.cm.cubehelix,
                  extent=[self.xmin, self.xmax, self.ymin, self.ymax])
        ax.scatter(self.angles[0,:], self.angles[1,:], c='k', s=1, edgecolor='')
        ax.set_xlim([self.xmin, self.xmax])
        ax.set_ylim([self.ymin, self.ymax])
        fig.colorbar(pos, ax=ax)
        plt.tight_layout()
        plt.savefig('Visualization/distr/kde.png')
        plt.show()
    
    def plot_samples(self, samples):
        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel('$psi$')
        ax.set_ylabel('$phi$')
        ax.set_title("Samples in distribution")
        pos = ax.imshow(np.rot90(self.density), cmap=plt.cm.cubehelix,
                  extent=[self.xmin, self.xmax, self.ymin, self.ymax])
        sample_angles = np.transpose(samples.numpy())
        ax.scatter(sample_angles[0,:], sample_angles[1,:], c='r', s=5, edgecolor='')
        ax.set_xlim([self.xmin, self.xmax])
        ax.set_ylim([self.ymin, self.ymax])
        fig.colorbar(pos, ax=ax)
        plt.tight_layout()
        plt.savefig('Visualization/distr/kde.png')
        plt.show()