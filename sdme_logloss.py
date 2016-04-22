# Routines for SDME
import numpy as np
import scipy as sp


def get_states(N):
    nstates = 2**N
    stvec = range(nstates)
    stmat = np.tile(stvec, (N, 1))
    for rw in range(N):
        stmat[rw, :] = np.mod(stmat[rw, :]/(2**rw), 2)
    return stmat

def data_to_empirical(spikes, stmat):
    # spikes: N x stimlen x nrepeats
    N, stimlen, nreps = np.shape(spikes)
    spikes2 = spikes
    stmat2= stmat
    stmat2[stmat2 == 0] = -1
    spikes2[spikes2 == 0] = -1
    data_state = np.tensordot(np.transpose(stmat2), spikes2, 1) / (1.0*(2**N)) # Find which state the network is in
    data_state[data_state < 1.0] = 0
    empirical_distrib = np.sum(data_state, 2) / (1.0*nreps) # compute histogram over each rep. 
    return empirical_distrib
    
def data_to_empirical2(spikes):
    # spikes: N x stimlen x nrepeats
    N, stimlen, nreps = np.shape(spikes)
    spikes2 = 1.0*spikes
    stmat= get_states(N)
    stmat[stmat == 0] = -1
    spikes2[spikes2 == 0] = -1
    data_state = np.tensordot(np.transpose(stmat), spikes2, 1) / (1.0*(N)) # Find which state the network is in
    data_state[data_state < 1.0] = 0
    empirical_distrib = np.sum(data_state, 2) / (1.0*nreps) # compute histogram over each rep. 
    return empirical_distrib
    
def data_to_sta(spikes, stim):
    # sptimes: N x stimlen x nrepreats
    # stim: stimdim x stimlen
    N, stimlen, nreps = np.shape(spikes)
    spikes2 = np.sum(spikes, 2)
    sta = np.tensordot(spikes2, np.transpose(stim), 1)
    sta = sta / (1.0*nreps*stimlen)
    return sta

def data_to_stc(spikes, stim):
    # spikes: N x stimlen x nrepeats
    # stim: stimdim x stimlen
    # gonna need to break the stim/spikes into chunks, do stc individually, then average together?
    # sta: N x stimdim
    N, stimlen, nreps = np.shape(spikes)
    
    #stim_outer_diag = np.diagonal(np.einsum('ij, kl', stim, stim), axis1=1, axis2=3)
    #stc = np.einsum('ij,kjl->ikl', spikes, stim_outer_diag) / (1.0*nreps*stimlen)
    spikes2 = 1.0*np.squeeze(np.sum(spikes, 2))
    stim_outer_diag = np.einsum('ij,kj,lj->ikl', spikes2,stim,stim)
    stc = stim_outer_diag / (1.0*nreps*stimlen)
    return stc

#def data_to_cov(spikes):
    # Estimate the neuron-neuron covariance
    # spikes: N x stimlen x nrepeats
    #N, stimlen, nreps = np.shape(spikes)
    
    #spikes = 1.0*np.squeeze(np.sum(spikes,2))
    # cov = np.tensordot(spikes, np.transpose(spikes), 1) / (1.0*nreps*stimlen)
    return cov

def data_to_cov(spikes):
    # Estimate the neuron-neuron covariance
    # spikes: N x stimlen x nrepeats
    N, stimlen, nreps = np.shape(spikes)
    
    spikes2 = 1.0*np.squeeze(np.sum(spikes,2))
    spikes2 -= np.transpose(np.tile(np.mean(spikes2, 1), (stimlen, 1)))
 
    cov = np.tensordot(spikes2, np.transpose(spikes2), 1) / (1.0*nreps*stimlen)
    cov -= np.diagflat(np.diag(cov))
    return cov

def data_to_cov2(spikes):
    # Estimate the neuron-neuron covariance
    # spikes: N x stimlen x nrepeats
    N, stimlen, nreps = np.shape(spikes)
    
    spikes = 1.0*np.squeeze(np.sum(spikes,2))
    spikes -= np.transpose(np.tile(np.mean(spikes, 1), (stimlen, 1)))

    cov = np.tensordot(spikes, np.transpose(spikes), 1) / (1.0*nreps*stimlen)
    cov -= np.diagflat(np.diag(cov))
    return cov
    
def sdme_logloss(stim, N, stimlen, A, B, C):
    
    # A: Current STA model estimate N x StimDim
    # B: Current STC model estimate N x StimDim x StimDim
    # C: Current Neuron-Neuron covarariance estimate N x N
    # stmat: matrix of possible states
    # stim: stim matrix StimDim x StimLen
    # resp: response matrix N x SimLen   or... P(x|s) from data: 2^N x Stimlen because this is fixed for the fitting
    
    
    stimdim, stimlen = np.shape(stim)
    
    #Generate samples 
    p_params = [stim, 1, A, B, C]
    N_samples = 500
    N_burnin = 50
    N_skip = 100
    init = 1.0*(np.random.uniform(size=(N, stimlen)) >0.5)
    dx=0
    samples = parallel_gibbs_sampler2(sdme_p, N, stimlen, N_samples, N_burnin, N_skip, 
                                      init, dx, *p_params)

    lgp_samp = np.zeros(stimlen)
    for samp in range(N_samples):
        lgp_samp = lgp_samp + -1.0*np.log(sdme_p(samples[:, :, samp], stim, 1, A, B, C))
    
    logloss = np.mean(lgp_samp)
    return logloss

def sdme_logloss0(stim, resp, stmat, A, B, C):
    
    # A: Current STA model estimate N x StimDim
    # B: Current STC model estimate N x StimDim x StimDim
    # C: Current Neuron-Neuron covarariance estimate N x N
    # stmat: matrix of possible states
    # stim: stim matrix StimDim x StimLen
    # resp: response matrix N x SimLen   or... P(x|s) from data: 2^N x Stimlen because this is fixed for the fitting
    
    N, stimlen = np.shape(resp)
    stimdim, stimlen = np.shape(stim)
    
    # Compute the probability over states
    corr_states = np.diag(np.dot(np.dot(np.transpose(stmat), C), stmat))
    # **** Need to add second order RF
    E = np.dot(np.dot(np.transpose(stmat), A),stim) + np.transpose(np.tile(corr_states, (stim_len, 1)))
    probs = np.exp(E) / np.sum(np.exp(E),0 )  # 2^N x StimLen 
    
    logloss = -1.0*np.trace(np.dot(np.transpose(resp), probs)) / 1.0*stimlen
    return logloss
    
    
def sdme_logloss2(stim, resp, stmat, A, B, C):
    
    # A: Current STA model estimate N x StimDim
    # B: Current STC model estimate N x StimDim x StimDim
    # C: Current Neuron-Neuron covarariance estimate 2^N x 2^N
    # stmat: matrix of possible states
    # stim: stim matrix StimDim x StimLen
    # resp: response matrix N x SimLen   or... P(x|s) from data: 2^N x Stimlen because this is fixed for the fitting
    
    N, stimlen = np.shape(resp)
    stimdim, stimlen = np.shape(stim)
    
    # Compute the probability over states
    corr_states = np.diag(np.dot(np.dot(np.transpose(stmat), C), stmat))
    outcorr = np.einsum('ijk,jt,kt->it', B, stim, stim)
    
    E = np.dot(np.dot(np.transpose(stmat), A),stim) + np.transpose(np.tile(corr_states, (stimlen, 1))) + np.dot(np.transpose(stmat), outcorr)
  
    probs = np.exp(E) / np.sum(np.exp(E),0 )  # 2^N x StimLen 
    probs[np.isnan(probs)] = 1.0
   
    logprobs = np.log(probs)
    logprobs[np.isnan(logprobs)] = 0.0
    
    logloss = np.diag(np.dot(np.transpose(resp), logprobs)) 
    loglosszeros = np.isnan(logloss)
    #logloss[loglosszeros] = 0.0
    logloss = -1.0*np.sum(logloss) / 1.0*stimlen
    return logloss

def sdme_logloss3(stim, resp, stmat, A, B, C):
    
    # A: Current STA model estimate N x StimDim
    # B: Current STC model estimate N x StimDim x StimDim
    # C: Current Neuron-Neuron covarariance estimate 2^N x 2^N
    # stmat: matrix of possible states
    # stim: stim matrix StimDim x StimLen
    # resp: response matrix N x SimLen   or... P(x|s) from data: 2^N x Stimlen because this is fixed for the fitting
    
    N, stimlen = np.shape(resp)
    stimdim, stimlen = np.shape(stim)
    
    # Compute the probability over states
    corr_states = np.diag(np.dot(np.dot(np.transpose(stmat), C), stmat))
    outcorr = np.einsum('ijk,jt,kt->it', B, stim, stim)
    
    E = np.dot(np.dot(np.transpose(stmat), A),stim) + np.transpose(np.tile(corr_states, (stimlen, 1))) + np.dot(np.transpose(stmat), outcorr)
  
    probs = np.exp(E) / np.sum(np.exp(E),0 )  # 2^N x StimLen 
    probs[np.isnan(probs)] = 1.0
   
    logprobs = np.log(probs)
    logprobs[np.isnan(logprobs)] = 0.0
    
    #logloss = np.diag(np.dot(np.transpose(resp), logprobs)) 
    
    #loglosszeros = np.isnan(logloss)
    #logloss[loglosszeros] = 0.0
    #logloss = -1.0*np.sum(logloss) / 1.0*stimlen
    logloss = -1.0*np.einsum('ij,ij',logprobs, resp)  / (1.0*stimlen)
    return logloss
    
def sdme_dlogloss(stim, N, dat_STA, dat_STC, dat_COV, A, B, C):
    # A: Current STA model estimate N x StimDim
    # B: Current STC model estimate N x StimDim x StimDim
    # C: Current Neuron-Neuron covarariance estimate 2^N x 2^N
    # stmat: matrix of possible states
    # stim: stim matrix StimDim x StimLen
    # dat_ : parameters estimated from data. 
    
        
    # N, stimlen = np.shape(resp)
    stimdim, stimlen = np.shape(stim)
    
    # #Generate samples 
    # p_params = [stim, 1, A, B, C]
    # N_samples = 100
    # N_burnin = 50
    # N_skip = 10
    # init = 1.0*(np.random.uniform(size=(N, stimlen)) >0.5)
    # dx=0
    # samples = parallel_gibbs_sampler2(sdme.sdme_p, N, stimlen, N_samples, N_burnin, N_skip, 
    #                                   init, dx, *p_params)
    
    #Generate samples 
    p_params = [stim, 1, A, B, C]
    N_samples = 500
    N_burnin = 50
    N_skip = 100
    init = 1.0*(np.random.uniform(size=(N, stimlen)) >0.5)
    dx=0
    samples = parallel_gibbs_sampler2(sdme_p, N, stimlen, N_samples, N_burnin, N_skip, 
                                      init, dx, *p_params)

    mod_STA = data_to_sta(samples, stim)
    #mod_STC = data_to_stc(samples, stim)
    mod_COV = data_to_cov(samples)
    
    df1 = dat_STA - mod_STA
    #df2 = dat_STC - mod_STC
    df3 = dat_COV - mod_COV
    df2 = 0
    return [df1, df2, df3]
    

def sdme_dlogloss0(stim, dat_STA, dat_STC, dat_COV, stmat, A, B, C):
    # A: Current STA model estimate N x StimDim
    # B: Current STC model estimate N x StimDim x StimDim
    # C: Current Neuron-Neuron covarariance estimate 2^N x 2^N
    # stmat: matrix of possible states
    # stim: stim matrix StimDim x StimLen
    # dat_ : parameters estimated from data. 
    
        
    N, stimlen = np.shape(resp)
    stimdim, stimlen = np.shape(stim)
    
    # Compute the probability over states
    corr_states = np.diag(np.dot(np.dot(np.transpose(stmat), C), stmat))
    outcorr = np.einsum('ijk,jt,kt->it', B, stim, stim)
    
    E = np.dot(np.dot(np.transpose(stmat), A),stim) + np.transpose(np.tile(corr_states, (stimlen, 1))) + np.dot(np.transpose(stmat), outcorr)
    #print(E)
    probs = np.exp(E) / np.sum(np.exp(E),0 )  # 2^N x StimLen 
    
    probs = np.expand_dims(probs, 2)
    mod_STA = data_to_sta(probs)
    mod_STC = data_to_stc(probs)
    mod_COV = data_to_cov(probs)
    
    df1 = dat_STA - mod_STA
    df2 = dat_STC - mod_STC
    df3 = dat_COV - mod_COV
    
    return [df1, df2, df3]
    
    
def sdme_dlogloss2(stim, stmat, dat_STA, dat_STC, dat_COV, A, B, C):
    # A: Current STA model estimate N x StimDim
    # B: Current STC model estimate N x StimDim x StimDim
    # C: Current Neuron-Neuron covarariance estimate 2^N x 2^N
    # stmat: matrix of possible states
    # stim: stim matrix StimDim x StimLen
    # dat_ : parameters estimated from data. 
    
        
    N, N2 = np.shape(stmat)
    stimdim, stimlen = np.shape(stim)
    
    # Compute the probability over states
    corr_states = np.diag(np.dot(np.dot(np.transpose(stmat), C), stmat))
    outcorr = np.einsum('ijk,jt,kt->it', B, stim, stim)
    
    E = np.dot(np.dot(np.transpose(stmat), A),stim) + np.transpose(np.tile(corr_states, (stimlen, 1))) + np.dot(np.transpose(stmat), outcorr)
    #print(E)
    probs = np.exp(E) / np.sum(np.exp(E),0 )  # 2^N x StimLen 
    
    #probs = np.expand_dims(probs, 2)
    #mod_STA = data_to_sta(probs, stim)
    #mod_STC = data_to_stc(probs, stim)
    #mod_COV = data_to_cov(probs)
    
    unit_probs = np.dot(stmat, probs)
    unit_probs_expand = np.expand_dims(unit_probs, 2)
    mod_STA = data_to_sta(unit_probs_expand, stim)
    mod_STC = data_to_stc(unit_probs_expand, stim)
    mod_COV = data_to_cov(unit_probs_expand)
    
    df1 = dat_STA - mod_STA
    df2 = dat_STC - mod_STC
    df3 = dat_COV - mod_COV
    
    return [df1, df2, df3]
    
def sdme_dlogloss3(stim, stmat, dat_STA, dat_STC, dat_COV, A, B, C):
    # A: Current STA model estimate N x StimDim
    # B: Current STC model estimate N x StimDim x StimDim
    # C: Current Neuron-Neuron covarariance estimate 2^N x 2^N
    # stmat: matrix of possible states
    # stim: stim matrix StimDim x StimLen
    # dat_ : parameters estimated from data. 
    
        
    N, N2 = np.shape(stmat)
    stimdim, stimlen = np.shape(stim)
    
    # Compute the probability over states
    corr_states = np.diag(np.dot(np.dot(np.transpose(stmat), C), stmat))
    outcorr = np.einsum('ijk,jt,kt->it', B, stim, stim)
    
    E = np.dot(np.dot(np.transpose(stmat), A),stim) + np.transpose(np.tile(corr_states, (stimlen, 1))) + np.dot(np.transpose(stmat), outcorr)
    #print(E)
    probs = np.exp(E) / np.sum(np.exp(E),0 )  # 2^N x StimLen 
    
    probs[np.isnan(probs)] = 1.0

    # Generate Responses
    # cumulative probs
    probs_c = np.concatenate((np.zeros((1, stimlen)), np.cumsum(probs, 0)), 0)

    # generate random vec  (COULD BE MADE MORE EFFICIENT)
    nreps = 20
    pop_response_dlogloss = np.zeros((N, stimlen, nreps))
    for rep in range(nreps):
        prb = np.random.rand(1, stimlen)
        outcomes = 1*np.logical_and(probs_c[0:-1, :] < prb, probs_c[1:, :] > prb)
        #np.concatenate((outcomesav, outcomes))
        pop_response_this_rep = np.dot(stmat, outcomes)
        pop_response_dlogloss[:, :, rep] = pop_response_this_rep

    
    
    #probs = np.expand_dims(probs, 2)
    #mod_STA = data_to_sta(probs, stim)
    #mod_STC = data_to_stc(probs, stim)
    #mod_COV = data_to_cov(probs)
    
    #unit_probs = np.dot(stmat, probs)
    #unit_probs_expand = np.expand_dims(unit_probs, 2)
    mod_STA = data_to_sta(pop_response_dlogloss, stim)
    mod_STC = data_to_stc(pop_response_dlogloss, stim)
    mod_COV = data_to_cov(pop_response_dlogloss)
    
    df1 = dat_STA - mod_STA
    df2 = dat_STC - mod_STC
    df3 = dat_COV - mod_COV
    
    return [df1, df2, df3]
    

def sdme_p(resp_vec, stim, order, beta_1, beta_2, beta_3):
    '''
    This generates the probability of a response vector based on model parameters 

    Parameters
    ------
    resp_vec : numpy array
        Array of binary values representing the response of each neuron
        N_neuron x StimLen
    stim :
        stimulus matrix: StimDim x StimLen
    beta_1 : Current STA model estimate NxStimDim
    beta_2 :
        Current STC model estimate N x StimDim x StimDim 
    beta_3 :
        current model covariance estimate NxN

    Returns
    ------
    prob : float 
        probability.  (not really a probability. But we don't need normalization for MC per se )
    '''
    if order == 1:
        E = np.diag(np.dot(np.dot(np.transpose(resp_vec),beta_1),stim)) + np.diag(np.dot(np.dot(np.transpose(resp_vec), beta_3), resp_vec))
    elif order == 2:
        E = np.diag(np.dot(np.dot(np.transpose(resp_vec),beta_1),stim)) + np.diag(np.dot(np.dot(np.transpose(resp_vec), beta_3), resp_vec))
    probs = np.exp(-E)
    return probs

def parallel_gibbs_sampler(p, N_dim, N_t, N_samples, N_burnin, N_skip, init, dx, *args, **kwargs):
    '''
    What a year it's been. 
    This will return a number of samples from the probability function p
    obtained by gibbs sampling

    Parameters 
    ------
    p : function 
        A probability density over Ndim response variables that are {0, 1}
    N_dim : int 
        Number of dimensions of response
    N_t : int 
        Number of time points in response  
    N_samples : int 
        Number of samples you want 
    N_burnin : int 
        Number of iterations to throw away at beginning 
    N_skip : int 
        Number of iterations to skip before taking a sample 
    init : array 
        starting point.  Ndim x N_t

    Returns
    ------
    samples : array 
        Samples generated from p
        Ndim x N_t x N_samples
    '''

    resp = init 
    samples = np.zeros((N_dim, N_t, N_samples))
    n_accept = np.zeros((N_dim, N_t))
    # Burn in
    for step in range(N_burnin):
        for x in range(N_dim):
            proposal = 2.0*(np.random.uniform(size=(1, N_t)) >= 0.5) - 1
            current = resp[x, :] 
            prob_minus = p(resp, *args, **kwargs)
            resp[x, :] = proposal.flatten()
            prob_plus = p(resp, *args, **kwargs)
            prob_rat = np.divide(prob_plus,prob_minus)
            acc = np.minimum(np.ones(prob_rat.shape), prob_rat)
            keeps = np.random.uniform(size=(1, N_t))
            acc_vec = np.greater(keeps, acc)
            resp[x, acc_vec.flatten()] = current[acc_vec.flatten()]
    iters=0
    samps=0
    while samps < N_samples:
        iters = iters+1
        for x in range(N_dim):
            proposal = 2.0*(np.random.uniform(size=(1, N_t)) >= 0.5) - 1
            current = resp[x, :] 
            prob_minus = p(resp, *args, **kwargs)
            resp[x, :] = proposal.flatten()
            prob_plus = p(resp, *args, **kwargs)
            prob_rat = np.divide(prob_plus,prob_minus)
            acc = np.minimum(np.ones(prob_rat.shape), prob_rat)
            keeps = np.random.uniform(size=(1, N_t))
            acc_vec = np.greater(keeps, acc)
            resp[x, acc_vec.flatten()] = current[acc_vec.flatten()]
            n_accept[x, :] = n_accept[x, :] + 1.0*acc_vec
        if np.mod(iters, N_skip) == 0:
            samples[:, :, samps] = resp
            samps = samps+1
    return (samples, n_accept/iters) 

def parallel_gibbs_sampler2(p, N_dim, N_t, N_samples, N_burnin, N_skip, init, dx, *args, **kwargs):
    '''
    What a year it's been. 
    This will return a number of samples from the probability function p
    obtained by gibbs sampling

    Parameters 
    ------
    p : function 
        A probability density over Ndim response variables that are {0, 1}
    N_dim : int 
        Number of dimensions of response
    N_t : int 
        Number of time points in response  
    N_samples : int 
        Number of samples you want 
    N_burnin : int 
        Number of iterations to throw away at beginning 
    N_skip : int 
        Number of iterations to skip before taking a sample 
    init : array 
        starting point.  Ndim x N_t

    Returns
    ------
    samples : array 
        Samples generated from p
        Ndim x N_t x N_samples
    '''

    resp = init 
    samples = np.zeros((N_dim, N_t, N_samples))
    n_accept = np.zeros((N_dim, N_t))
    # Burn in
    for step in range(N_burnin):
        for x in range(N_dim):
            resp[x, :] = 1
            raw_prob1 = p(resp, *args, **kwargs)
            resp[x, :] = -1
            raw_prob2 = p(resp, *args, **kwargs)
            prob1 = np.divide(raw_prob1,(raw_prob1+raw_prob2))
            to_1 = np.less(np.random.uniform(size=(1, N_t)), prob1)
            resp[x, to_1.flatten()] = 1
    iters=0
    samps=0
    while samps < N_samples:
        iters = iters+1
        for x in range(N_dim):
            resp[x, :] = 1
            raw_prob1 = p(resp, *args, **kwargs)
            resp[x, :] = -1
            raw_prob2 = p(resp, *args, **kwargs)
            prob1 = np.divide(raw_prob1,(raw_prob1+raw_prob2))
            to_1 = np.less(np.random.uniform(size=(1, N_t)), prob1)
            resp[x, to_1.flatten()] = 1
        if np.mod(iters, N_skip) == 0:
            samples[:, :, samps] = resp
            samps = samps+1
    return samples 

def gibbs_sampler(p, N_dim, N_samples, N_burnin, N_skip, init, dx, *args, **kwargs):
    '''
    What a year it's been. 
    This will return a number of samples from the probability function p
    obtained by gibbs sampling

    Parameters 
    ------
    p : function 
        A probability density over Ndim response variables that are {0, 1}
    N_dim : int 
        Number of dimensions of response 
    N_samples : int 
        Number of samples you want 
    N_burnin : int 
        Number of iterations to throw away at beginning 
    N_skip : int 
        Number of iterations to skip before taking a sample 

    Returns
    ------
    samples : array 
        Samples generated from p
    '''

    resp = init 
    samples = np.zeros((N_dim, N_samples))
    # Burn in
    for step in range(N_burnin):
        for x in range(N_dim):
            proposal = 2.0*(np.random.uniform() >= 0.5) - 1
            current = resp[x] 
            prob_minus = p(resp, *args, **kwargs)
            resp[x] = proposal
            prob_plus = p(resp, *args, **kwargs)
            acc = np.min([1.0, prob_plus/prob_minus])
            if np.random.uniform() >= acc:
                resp[x] = current
    iters=0
    samps=0
    while samps < N_samples:
        iters = iters+1
        for x in range(N_dim):
            proposal = 2.0*(np.random.uniform() >= 0.5) - 1
            current = resp[x]
            prob_minus = p(resp, *args, **kwargs)
            resp[x] = proposal
            prob_plus = p(resp, *args, **kwargs)
            acc = np.min([1.0, prob_plus/prob_minus])
            if np.random.uniform() >= acc:
                resp[x] = current
        if np.mod(iters, N_skip) == 0:
            samples[:, samps] = resp
            samps = samps+1
    return samples 

def mh_sampler(p, N_dim, N_samples, N_burnin, N_skip, init, dx, *args, **kwargs):
    '''
    What a year it's been. 
    This will return a number of samples from the probability function p
    obtained by gibbs sampling

    Parameters 
    ------
    p : function 
        A probability density over Ndim response variables that are {0, 1}
    N_dim : int 
        Number of dimensions of response 
    N_samples : int 
        Number of samples you want 
    N_burnin : int 
        Number of iterations to throw away at beginning 
    N_skip : int 
        Number of iterations to skip before taking a sample 

    Returns
    ------
    samples : array 
        Samples generated from p
    '''

    resp = init 
    samples = np.zeros((N_dim, N_samples))
    # Burn in
    for step in range(N_burnin):
        for x in range(N_dim):
            
            current = resp[x] 
            proposal = dx*np.random.randn() + current
            prob_minus = p(resp, *args, **kwargs)
            resp[x] = proposal
            prob_plus = p(resp, *args, **kwargs)
            acc = np.min([1.0, prob_plus/prob_minus])
            if np.random.uniform() >= acc:
                resp[x] = current
    iters=0
    samps=0
    while samps < N_samples:
        iters = iters+1
        for x in range(N_dim):
            
            current = resp[x]
            proposal = dx*np.random.randn() + current
            prob_minus = p(resp, *args, **kwargs)
            resp[x] = proposal
            prob_plus = p(resp, *args, **kwargs)
            acc = np.min([1.0, prob_plus/prob_minus])
            if np.random.uniform() >= acc:
                resp[x] = current
        if np.mod(iters, N_skip) == 0:
            samples[:, samps] = resp
            samps = samps+1
    return samples

def sdme_p_binary(resp_vec, stim, neural_order, stim_order, ):
    return 0