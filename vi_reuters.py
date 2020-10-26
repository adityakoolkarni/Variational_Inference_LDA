import numpy as np
import pandas as pd
import nltk
from nltk.corpus import reuters
from scipy.special import digamma
from scipy.special import gamma as Gamma
from scipy.special import loggamma as LGamma
from tqdm import tqdm
import matplotlib.pyplot as plt

def variational_inference(num_doc=5,iterations=1000):
    '''
    The following contains the implementation of the Variationa Inference Algorithm Based on the 
    paper: https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf

    Here we apply the algorithm to Reuters Corpus Dataset to estimate the parameters gamma,phi
    of the surrogate function q(theta,z|gamma,phi), assuming we have learnt the parameters of the conditional 
    distribution p(theta,z|w,alpha,beta). Here gamma,alpha is a k-dimensional(number of topics i.e. latent variables) vector.
    Alpha determines the distribution of the topics within a corpus, gamma is the surrogate equivalent of alpha. Phi is a
    Nxk matrix, where N is the number of words in the document. And each row of phi are the parameters of a multinomial distribution
    over the topics.

    parameters: num_doc -> integer indicating number of documents to be read.
                iterations -> indicating number of iterations to be performed 
                                for the convergence of the algorithm

    returns: gamma (k-dimensional), phi (Nxk) 
    '''
    assert isinstance(num_doc,int), "Number of doc should be int and not %s"%type(num_doc)
    assert isinstance(iterations,int),"Number of iterations should be int and not %s"%type(iterations)

    

    ## Get total words and total relations
    N = len(reuters.words(reuters.fileids()[num_doc])) #TODO: Make read all files
    words = reuters.words(reuters.fileids()[num_doc])
    #print("Total words ",N)
    unique_words = {word:ind for ind,word in enumerate(set(words))}
    #print("Unique words ",len(unique_words))
    
    K = len(reuters.words(reuters.fileids()[:]))
    ## Number of Topics
    K = 5

    ## Initialize alpha and beta
    alpha = np.ones(K)
    N_unique = len(unique_words) #number of unique words
    beta = np.random.dirichlet(np.ones(N_unique),(K))#topics(K) x words(N)
    assert np.sum(np.sum(beta,axis=1)) >= K-1e-3,'Beta initialization failed %s'%(np.sum(np.sum(beta,axis=1)))

    ## Initialize phi and gamma
    phi = np.ones((N,K))/K 
    phi_next = np.ones((N,K))/K
    gamma = alpha + N/K #size = K
    gamma_next = alpha + N/K
    variational_free_enery = np.zeros(iterations)

    ## Start VI inference 
    for iteration in tqdm(range(iterations)):
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
    plt.figure(dpi=200)
    plt.plot(range(iterations),variational_free_enery)
    plt.title('Variational Free Energy')
    plt.xlabel('iterations')
    plt.ylabel('Free Energy')
            


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
    This computes variational free energy or ELBO a tractable form from the LDA paper
    has 5 components to it. Each of it are encoded in li term below

    '''
    digamma_diff = digamma(gamma) - digamma(np.sum(gamma))
    w_one_hot = np.zeros((len(w_ind),V))
    w_one_hot[np.arange(len(w_ind)),np.array(w_ind)] = 1
    l1 = np.log(digamma(np.sum(alpha))) - np.sum(LGamma(alpha)) + np.sum((alpha - 1)*digamma_diff)
    l2 = np.sum(phi * (digamma_diff[np.newaxis,:] + np.zeros((phi.shape))))
    l3 = np.sum(phi * (w_one_hot @ np.log(beta).T)) 
    ## This is unstable 
    #l4 = -np.log(Gamma(np.sum(gamma))) + np.sum(np.log(Gamma(gamma))) - np.sum((gamma-1) * (digamma_diff))
    ## Stable Equivalent
    l4 = -LGamma(np.sum(gamma)) + np.sum(LGamma(gamma)) - np.sum((gamma-1) * (digamma_diff))
    l5 = -np.sum(phi * np.log(phi))

    #print("L1 ", l1)
    #print("L2 ", l2)
    #print("L3 ", l3)
    #print("L4 ", l4)
    #print("L5 ", l5)
    return l1 + l2 + l3 + l5 + l4



























