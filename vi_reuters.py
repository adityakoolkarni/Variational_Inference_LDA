import numpy as np
import pandas as pd
import nltk
from nltk.corpus import reuters
from scipy.special import digamma
from scipy.special import gamma as Gamma
from tqdm import tqdm
import matplotlib.pyplot as plt

def variational_inference(**kwds):
    '''
    '''
    

    ## Get total words and total relations
    N = len(reuters.words(reuters.fileids()[:kwds['num_doc']])) #TODO: Make read all files
    words = reuters.words(reuters.fileids()[:kwds['num_doc']])
    #print("Total words ",N)
    unique_words = {word:ind for ind,word in enumerate(set(words))}
    #print("Unique words ",len(unique_words))
    
    K = len(reuters.words(reuters.fileids()[:]))
    ## Number of Topics
    K = 5

    ## Initialize alpha and beta
    alpha = np.ones(K)/100
    N_unique = len(unique_words) #number of unique words
    beta = np.random.dirichlet(np.ones(N_unique),(K))#topics(K) x words(N)
    assert np.sum(np.sum(beta,axis=1)) >= K-1e-3,'Beta initialization failed %s'%(np.sum(np.sum(beta,axis=1)))

    ## Initialize phi and gamma
    phi = np.ones((N,K))/K 
    phi_next = np.ones((N,K))/K
    gamma = alpha + N/K #size = K
    gamma_next = alpha + N/K
    variational_free_enery = np.zeros(kwds['iterations'])

    ## Start VI inference 
    for iteration in tqdm(range(kwds['iterations'])):
        w_ind = []
        for n in range(N):
            ind = unique_words[words[n]]
            w_ind.append(int(ind))
            for i in range(K):
                phi[n,i] = beta[i,ind] * np.exp(digamma(gamma[i]))
            phi[n] = normalize(phi[n])
            assert np.sum(phi[n]) >= 1.00 - 1e-3, 'Normalization Failed'
        #print("Gamma is {} and Alpha is {} {} ".format(gamma,alpha,iteration))
        gamma = alpha + np.sum(phi,axis=0)
        #print("Phi summation",np.sum(phi,axis=0),phi[0])
        assert int(gamma.shape[0]) == K,'Gamma shape changed to %s!'%gamma.shape
        variational_free_enery[iteration] = get_variational_free_energy(alpha,beta,gamma,phi,w_ind,N_unique)
        if iteration % 100 == 0:
            print('Final VI',variational_free_enery[iteration])
    return variational_free_enery
    #plt.plot(range(kwds['iterations']),variational_free_enery)
    #plt.label('Variational Free Energy')
    #plt.xlabel('iterations')
    #plt.ylabel('Free Energy')
            


def normalize(phi):
    '''
    returns the normalized phi along the second axes
    params: phi is 1D np array
    '''
    assert isinstance(phi,np.ndarray)
    assert phi.ndim == 1
    return phi / np.sum(phi)

def get_variational_free_energy(alpha,beta,gamma,phi,w_ind,V):
    '''
    This computes variational free energy

    '''
    digamma_diff = digamma(gamma) - digamma(np.sum(gamma))
    w_one_hot = np.zeros((len(w_ind),V))
    w_one_hot[np.arange(len(w_ind)),np.array(w_ind)] = 1
    l1 = np.log(digamma(np.sum(alpha))) - np.sum(np.log(Gamma(alpha))) + np.sum((alpha - 1)*digamma_diff)
    l2 = np.sum(phi * (digamma_diff[np.newaxis,:] + np.zeros((phi.shape))))
    l3 = np.sum(phi * (w_one_hot @ np.log(beta).T)) 
    l4 = -np.log(Gamma(np.sum(gamma))) + np.sum(np.log(Gamma(gamma))) - np.sum((gamma-1) * (digamma_diff))
    ##print(np.sum((gamma-1) * (digamma_diff)))
    #print(Gamma(gamma))
    #print("gama again", gamma)
    #print('alpha again',alpha)
    #print( np.sum(np.log(Gamma(alpha))))
    #print(np.sum(np.log(Gamma(gamma))))
    #print(np.log(Gamma(np.sum(gamma))))


    l5 = -np.sum(phi * np.log(phi))
    #print("L1 ", l1)
    #print("L2 ", l2)
    #print("L3 ", l3)
    #print("L4 ", l4)
    #print("L5 ", l5)
    return l1 + l2 + l3 + l5# + l4



























