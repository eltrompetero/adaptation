# ===================================================================================== #
# Module for pipelining calculations needed for paper.
# Author : Eddie Lee, edlee@santafe.edu
# ===================================================================================== #
import numpy as np
from pyutils.learners import *
from multiprocess import cpu_count
from workspace.utils import save_pickle
from itertools import product

SEED = 1  # fixed seed for generating trajectories, reduces variance, increases bias



def tau_range(run_passive=True):
    """Agent unfitness landscape for various environments. From "agents in binary
    environment.ipynb".

    Parameters
    ----------
    run_passive : bool, True
    """

    seed = SEED
    
    # shared properties
    betaRange = linspace_beta(1e-1, 1e3, 100)
    tauRange = np.logspace(0, 5, 13)

    # passive
    if run_passive:
        kwargs = {'noise':{'type':'binary', 'scale':.2},
                  'T':1_000_000,
                  'nBatch':1_000}
        learners = []
        dkl = {}

        for tau in tauRange:
            kwargs['rng'] = np.random.RandomState(seed)
            kwargs['noise']['tau'] = tau
            learner = Vision(**kwargs)
            dkl[tau] = learner.learn(betaRange, n_cpus=cpu_count()-1, save=False)
            learners.append(learner)
            print("Done with tau=%1.1E."%tau)

        save_pickle(['learners', 'betaRange', 'tauRange', 'seed', 'kwargs', 'dkl'],
                    'cache/vision_agent_sim_tau_range.p', True)

    # stabilizer
    kwargs = {'noise':{'type':'binary', 'scale':.2, 'weight':.95, 'v':.01},
              'T':10_000_000,
              'nBatch':1_000}
    learners = []
    dkl = {}

    for tau in tauRange:
        kwargs['rng'] = np.random.RandomState(seed)
        kwargs['noise']['tau'] = tau
        learner = Stigmergy(**kwargs)
        dkl[tau] = learner.learn(betaRange, n_cpus=cpu_count()-1, save=False)
        learners.append(learner)
        print("Done with tau=%1.1E."%tau)

    save_pickle(['learners', 'betaRange', 'tauRange', 'seed', 'kwargs', 'dkl'],
                'cache/stabilizer_agent_sim_tau_range.p', True)

    # dissipator
    kwargs['noise']['v'] = -.01 
    learners = []
    dkl = {}

    for tau in tauRange:
        kwargs['rng'] = np.random.RandomState(seed)
        kwargs['noise']['tau'] = tau
        learner = Stigmergy(**kwargs)
        dkl[tau] = learner.learn(betaRange, n_cpus=cpu_count()-1, save=False)
        learners.append(learner)
        print("Done with tau=%1.1E."%tau)

    save_pickle(['learners', 'betaRange', 'tauRange', 'seed', 'kwargs', 'dkl'],
                'cache/dissipator_agent_sim_tau_range.p', True)

def info_gain():
    """Landscapes for optimal memory and memory/forgetting tradeoff. From "info
    gain.ipynb."
    """

    seed = SEED

    scaleRange = np.logspace(-2, 0, 20)
    nBatchRange = np.logspace(1, 4, 20).astype(int)
    # agent properties
    betaRange = linspace_beta(1e-1, 100, 100)
    tau = 10
    T = 1_000_000

    # passive
    learners = {}
    dkl = {}
    for scale, nBatch in product(scaleRange, nBatchRange):
        # environment properties
        kwargs = {'noise':{'type':'binary', 'scale':scale},
                  'T':T,
                  'nBatch':nBatch}

        kwargs['rng'] = np.random.RandomState(seed)
        kwargs['noise']['tau'] = tau
        learner = Vision(**kwargs)
        dkl[(scale, nBatch)] = learner.learn(betaRange, n_cpus=cpu_count()-1, save=False)
        learners[(scale, nBatch)] = learner
        print("Done with (scale, nBatch)=(%1.2f, %d)."%(scale, nBatch))

    save_pickle(['learners', 'scaleRange', 'nBatchRange', 'betaRange', 'tau', 'seed', 'T', 'dkl'],
                'cache/vision_agent_landscape_tau100.p', True)
    
    # stabilizer
    learners = {}
    dkl = {}
    for scale, nBatch in product(scaleRange, nBatchRange):
        # environment properties
        kwargs = {'noise':{'type':'binary', 'scale':scale, 'weight':.95, 'v':.01},
                  'T':T,
                  'nBatch':nBatch}

        kwargs['rng'] = np.random.RandomState(seed)
        kwargs['noise']['tau'] = tau
        learner = Stigmergy(**kwargs)
        dkl[(scale, nBatch)] = learner.learn(betaRange, n_cpus=cpu_count()-1, save=False)
        learners[(scale, nBatch)] = learner
        print("Done with (scale, nBatch)=(%1.2f, %d)."%(scale, nBatch))

    save_pickle(['learners', 'scaleRange', 'nBatchRange', 'betaRange', 'tau', 'seed', 'T', 'dkl'],
                'cache/stabilizer_agent_landscape.p', True)

    # dissipator
    learners = {}
    dkl = {}
    for scale, nBatch in product(scaleRange, nBatchRange):
        # environment properties
        kwargs = {'noise':{'type':'binary', 'scale':scale, 'weight':.95, 'v':-.01},
                  'T':T,
                  'nBatch':nBatch}

        kwargs['rng'] = np.random.RandomState(seed)
        kwargs['noise']['tau'] = tau
        learner = Stigmergy(**kwargs)
        dkl[(scale, nBatch)] = learner.learn(betaRange, n_cpus=cpu_count()-1, save=False)
        learners[(scale, nBatch)] = learner
        print("Done with (scale, nBatch)=(%1.2f, %d)."%(scale, nBatch))

    save_pickle(['learners', 'scaleRange', 'nBatchRange', 'betaRange', 'tau', 'seed', 'T', 'dkl'],
                'cache/dissipator_agent_landscape.p', True)
