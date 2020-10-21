def variational_inference(**kwds):
    '''
    '''
    import numpy as np
    import pandas as pd
    import nltk
    from nltk.corpus import reuters
    from scipy.special import digamma
    

    ## Get total words and total relations
    N = len(reuters.words(reuters.fileids()[:10])) #TODO: Make read all files
    words = reuters.words(reuters.fileids()[:10])
    unique_words = {word:ind for ind,word in enumerate(set(words))}
    
    K = len(reuters.words(reuters.fileids()[:]))

    ## Initialize alpha and beta
    alpha = np.ones(K)
    N_unique = len(unique_words) #number of unique words
    beta = np.random.dirichlet(np.ones(N_unique),(K))#topics(K) x words(N)

    ## Initialize phi and gamma
    phi = np.ones(K,N) 
    phi_next = np.ones(K,N) 
    gamma = alpha + N/K #size = K
    gamma_next = alpha + N/K

    ## Start VI inference 
    for iteration in range(kwds['iterations']):
        for n in range(N):
            for i in range(K):
                ind = unique[words[n]]
                phi_next[n,i] = beta[i,ind] * np.exp(digamma(gamma[i]))
            phi[n] = normalize(phi_next[n])
            assert np.sum(phi[n]) >= 1.00 - 1e-3
        gamma_next = alpha + np.sum(phi,axis=0)
        variational_free_enery[iteration] = None
    plt.plot(range(kwds['iterations'],variational_free_enery))
    plt.label('Variational Free Energy')
    plt.xlabel('iterations')
    plt.ylabel('Free Energy')
            


def normalize(phi):
    '''
    returns the normalized phi along the second axes
    params: phi is 1D np array
    '''
    import numpy as np
    assert isinstance(phi,np.ndarray)
    assert phi.ndim == 1
    return phi / np.sum(phi)



