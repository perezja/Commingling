#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy.stats import norm
import glob
import argparse
import os

# Implemented from {https://www.ics.uci.edu/~smyth/courses/cs274/notes/EMnotes.pdf}
class Link:
    def __init__(self, simdata, true_params, tol, max_iter, maf):

        # tolerance level
        self.tol = tol
        # max iterations
        self.max_iter = max_iter

        # minor allele frequency
        self.maf = maf
 
        # arrange simulation data
        raw_df = pd.read_csv(simdata, sep='\t')

        "founder QT phenotypes"
        qtp_df = pd.melt(raw_df, value_vars=['pf','pm'], var_name='founder')   
        self.qtp = np.array(qtp_df['value'])

        with open(true_params) as f:
            params = f.read().strip().split(' ')[:-1] 
        
        # set parameter vector (label, value)
        self.true_params = self.set_true_params(params) 

        # EM result containers
        self.est_params = None
        self.sq_dist = None
        self.mse = None

       
    def true_param_string(self):
        return(', '.join([lab+"={0:5.3f}".format(val) for lab, val in self.true_params]))


    def est_param_string(self):
        labels = ['dq','muii', 'muij', 'mujj', 'de']
        return(', '.join([lab+"={0:5.3f}".format(val) for lab, val in self.est_params]))

    def sq_dist_string(self):

        self.calculate_square_distance()

        return(', '.join([lab+'={0:5.2e}'.format(val) for lab, val in self.sq_dist]))

    def set_true_params(self, params):

        theta = np.array([float(p) for p in params])

        dq = ('dq', theta[0])
        de = ('de', theta[-1])
        mu = [(l, v) for l,v in zip(['muii','muij','mujj'], theta[1:4])]
        mu.sort(key=lambda x:x[1])

        theta = [dq, de]+mu
        
        return([theta[i] for i in [0,2,3,4,1]])

    def set_est_params(self):

        self.Alpha_new.sort() 
        self.Mu_new.sort()

        dq = ('dq', np.sqrt(self.Alpha_new[0]))
        de = ('de', np.mean(self.Sigma_new))
        mu = list(map(lambda x: (x[1][0], x[0]), zip(self.Mu_new, self.true_params[1:4]))) 

        theta = [dq, de] + mu 
        self.est_params = [theta[i] for i in [0,2,3,4,1]]

    def display_true_params(self):
        block = "###############"
        print(block+'\nTrue Parameters\n'+block+'\n')
        print(self.true_param_string())
        print('\n')

    def display_config(self):
        block = "###############"
        print(block+'\nConfiguration\n'+block+'\n')
        print("tol: {}".format(self.tol))
        print("max iter: {}".format(self.max_iter))
        print('\n')

    def display_iteration(self, itr):

        head_str = "\nIteration: {}".format(itr)
        mu_str = " * mu_ii={0:5.3f}, mu_ij={1:5.3f}, mu_jj={2:5.3f}".format(self.Mu_new[0], self.Mu_new[1], self.Mu_new[2])
        mix_str = " * a_ii={0:5.3f}, a_ij={1:5.3f}, a_jj={2:5.3f}".format(self.Alpha_new[0], self.Alpha_new[1], self.Alpha_new[2])
        sig_str = " * sig_e={0:5.3f}".format(np.mean(self.Sigma_new))

        itr_str = '\n'.join([head_str, mu_str, mix_str, sig_str])

        print('\n'+itr_str, end='', flush=True), 

    def display_em_results(self):

        block = "#############################"
        print(block+'\nParameter Esimation Results\n'+block+'\n\n')

        print('true params: '+self.true_param_string())
        print('est. params: '+self.est_param_string())
        print('sq. dist: '+self.sq_dist_string())
        print('mse: {0:5.3f}'.format(self.mse))

        print('\n')

    def calculate_square_distance(self):

        sq_dist = []
        for true, est in zip(self.true_params, self.est_params):
            sqd = (true[1]-est[1])**2 
            sq_dist.append((true[0], sqd))

        self.sq_dist = sq_dist 
        self.mse = np.mean([sq[1] for sq in sq_dist])

    def log_likelihood(self, mixture_pdfs, mixture_wts):

        logl_xis = []

        # Incomplete log likelihood [Slide 9]
        for xi in self.qtp:
            logl_xis.append(np.log(np.sum([a*rv.pdf(xi) for rv, a in zip(mixture_pdfs, mixture_wts)])))
        return(np.sum(logl_xis))


    def hardyw_gf(self, q):
        """
        input: minor allele frequency
        returns: mixture component weights {a1,a2,...,ak} (i.e., genotype frequencies)
          * ak = p(zk) 
            where z = {z1, z2, ... , zk} is a vector of indicator variables for 'k' mixture components 
        """

        a1 = (1-q)**2
        a2 = 2*q*(1-q)
        a3 = q**2

        return(np.array((a1, a2, a3)))

    def EM(self, mean_init=None, sigma_init=None):

        # d-dimensional vector measurement
        # D = {x1, x2, ... , xN}
        D = self.qtp 

       # number of mixture components 
        K = 3 

        # initialize mixture weights vector A : (1 x k)"""
        Alpha = self.hardyw_gf(self.maf) 
        
        # initialize standard deviations
        if sigma_init is not None:
            if not isinstance(sigma_init, float):
                raise ValueError("Initial variance 'sigma_init' must be a point scalar.")
            Sigma = np.full(3, sigma_init)
        else:
            Sigma = np.full(3, np.var(D)) 

       # initialize mean vector
        if mean_init is not None:
            if not len(mean_init)==3:
                raise ValueError("Initial mean array 'mean_init' must be of length 3.")
            else:
                Mu = np.array(mean_init)
        else:
            Mu = np.random.normal(loc=np.mean(D), scale=np.var(D), size=K) 

        self.display_true_params()

        self.display_config()

        print('Initialization:')
        print(' * mixture weights: {0:5.3f}, {1:5.3f}, {2:5.3f}'.format(Alpha[0], Alpha[1], Alpha[2]))
        print(' * means: {0:5.3f}, {1:5.3f}, {2:5.3f}'.format(Mu[0], Mu[1], Mu[2]))
        print(' * stdv: {0:5.3f}'.format(Sigma[0]))


        for itr in range(self.max_iter):

            """
            E-step: calculate "membership weight" matrix W : (n x k) 
            """
    
            # 1. get 'frozen' pdf of 'k' mixture components
            rv = [] 
            for u, sig in zip(Mu, Sigma):
               rv.append(norm(loc=u, scale=np.sqrt(sig))) 
                
            # [EQN 1: Membership weight matrix]

            W = np.zeros((len(D),K))
            for i, xi in enumerate(D):

                denom = np.sum([rv[k].pdf(xi)*ak for k, ak in enumerate(Alpha)])
                for j in range(K):
                    W[i,j] = (rv[j].pdf(xi) * Alpha[j]) / denom
    
            # proves all weights form a partition (sum to 1)
            assert(all(np.around(np.apply_along_axis(np.sum, 1, W), 2)==1.0))
    
            """ 
            M-step: update values for parameters given current distribution
            """
    
            Alpha_new = np.zeros(len(Alpha)) 
            Mu_new = np.zeros(len(Mu))

            for j in range(K):

                # [EQN 2: MLE of mixture weights]
                Nk = np.sum(W[:,j]) 
                Alpha_new[j] = Nk/W.shape[0]    

                # [EQN 3: MLE of means for each mixture] 
                Mu_new[j] = np.sum([W[i,j]*xi for i, xi in enumerate(D)]) / Nk 
           
            # [EQN 4: MLE of variance]
            Sigma_new = np.zeros(K) 
            for j in range(K):
                Sigma_new[j] = np.sum([W[i,j]*((xi-Mu_new[j])**2) for i, xi in enumerate(D)]) / Nk

            # Update class accessible estimates
            self.Alpha_new = Alpha_new
            self.Mu_new = Mu_new
            self.Sigma_new = Sigma_new

            # Update mixture pdfs
            rv_new = [] 
            for u, sig in zip(self.Mu_new, self.Sigma_new):
               rv_new.append(norm(loc=u, scale=np.sqrt(sig))) 

            """
            Convergence: Check using Log-likelihood 
            """

            # [EQN 4: log-likelihood]
            logl_old = self.log_likelihood(rv, Alpha)
            logl_new = self.log_likelihood(rv_new, self.Alpha_new)
            if np.abs(logl_new - logl_old) < self.tol:
                print("\nConverged at {} iterations.\n".format(itr))
                self.set_est_params()
                self.display_em_results()
                return(0)

                break

            Alpha = self.Alpha_new
            Mu = self.Mu_new
            Sigma = self.Sigma_new

            self.display_iteration(itr)

        print("\nFailed to Converge.")
        return(1)

def main():

    parser = argparse.ArgumentParser(description="Project 1: Expectation Maximization for Commingling Analysis.")
    parser.add_argument('simdata_dir',help="Directory to simulated linkage data from 'link_data.sas'.\nDirectory must contain 'link_data<#>.txt' and 'params<#>.txt")
    parser.add_argument('-t', '--tol', type=float, default=0.001, help="Tolerance level for parameter estimation")
    parser.add_argument('-m', '--max_iter', type=int, default=100, help="Maximum number of iterations for EM routine.")
    parser.add_argument('-f', '--maf', type=float, default=0.2, help="Minor allele frequency (determines initial mixing component probabilities.)")
 
 
    args = parser.parse_args()
    
    simdata_files = glob.glob(os.path.join(args.simdata_dir, 'link_data*.txt'))
    param_files = glob.glob(os.path.join(args.simdata_dir, 'params*.txt'))

    simdata_files.sort()
    param_files.sort()

    for simdata, params in zip(simdata_files, param_files):
        l = Link(simdata, params, tol=args.tol, max_iter=args.max_iter, maf=args.maf) 
        l.EM()
        quit()
main()    
