# ====================================================================================== #
# Module for pipelining calculations needed for paper.
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
import numpy as np
from . import agent, eigen
from .utils import *
from multiprocess import cpu_count
from workspace.utils import save_pickle
from itertools import product

SEED = 1  # fixed seed for generating trajectories, reduces variance, increases bias



def tau_range(run_passive=True, run_stabilizer=True, run_dissipator=True):
    """Agent unfitness landscape for various environments. From "agents in binary
    environment.ipynb".

    Parameters
    ----------
    run_passive : bool, True
    run_stabilizer : bool, True
    run_dissipator : bool, True
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
            learner = agent.Vision(**kwargs)
            dkl[tau] = learner.learn(betaRange, n_cpus=cpu_count()-1, save=False)
            learners.append(learner)
            print("Done with tau=%1.1E."%tau)

        save_pickle(['learners', 'betaRange', 'tauRange', 'seed', 'kwargs', 'dkl'],
                    'cache/vision_agent_sim_tau_range.p', True)

    # dissipator
    if run_dissipator:
        kwargs = {'noise':{'type':'binary', 'scale':.2, 'weight':.95, 'v':-.01},
                  'T':10_000_000,
                  'nBatch':1_000}
        learners = []
        dkl = {}

        for tau in tauRange:
            kwargs['rng'] = np.random.RandomState(seed)
            kwargs['noise']['tau'] = tau
            learner = agent.Stigmergy(**kwargs)
            dkl[tau] = learner.learn(betaRange, n_cpus=cpu_count()-1, save=False)
            learners.append(learner)
            print("Done with tau=%1.1E."%tau)

        save_pickle(['learners', 'betaRange', 'tauRange', 'seed', 'kwargs', 'dkl'],
                    'cache/dissipator_agent_sim_tau_range.p', True)

    # stabilizer
    if run_stabilizer:
        kwargs = {'noise':{'type':'binary', 'scale':.2, 'weight':.95, 'v':.01},
                  'T':100_000_000,
                  'nBatch':1_000}
        learners = []
        dkl = {}

        for tau in tauRange:
            kwargs['rng'] = np.random.RandomState(seed)
            kwargs['noise']['tau'] = tau
            learner = agent.Stigmergy(**kwargs)
            dkl[tau] = learner.learn(betaRange, n_cpus=cpu_count()-1, save=False)
            learners.append(learner)
            print("Done with tau=%1.1E."%tau)

        save_pickle(['learners', 'betaRange', 'tauRange', 'seed', 'kwargs', 'dkl'],
                    'cache/stabilizer_agent_sim_tau_range.p', True)

def info_gain(run_passive=True, run_stabilizer=True, run_dissipator=True):
    """Landscapes for optimal memory and memory/forgetting tradeoff. From "info
    gain.ipynb."

    Parameters
    ----------
    run_passive : bool, True
    run_stabilizer : bool, True
    run_dissipator : bool, True
    """

    seed = SEED

    scaleRange = np.linspace(.01, 1, 20)
    nBatchRange = np.logspace(2, 5, 20).astype(int)
    # agent properties
    betaRange = linspace_beta(1e-1, 1e3, 126)
    tau = 10
    T = 100_000

    # passive
    if run_passive:
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

        print("Saving passive simulation.")
        print()
        save_pickle(['learners', 'scaleRange', 'nBatchRange', 'betaRange', 'tau', 'seed', 'T', 'dkl'],
                    'cache/vision_agent_landscape.p', True)
    
    # stabilizer
    if run_stabilizer:
        learners = {}
        dkl = {}
        for scale, nBatch in product(scaleRange, nBatchRange):
            # environment properties
            kwargs = {'noise':{'type':'binary', 'scale':scale, 'weight':.95, 'v':.01},
                      'T':T,
                      'nBatch':nBatch}

            kwargs['rng'] = np.random.RandomState(seed)
            kwargs['noise']['tau'] = tau
            learner = agent.Stigmergy(**kwargs)
            dkl[(scale, nBatch)] = learner.learn(betaRange, n_cpus=cpu_count()-1, save=False)
            learners[(scale, nBatch)] = learner

        print("Saving stabilizer simulation.")
        print()
        save_pickle(['learners', 'scaleRange', 'nBatchRange', 'betaRange', 'tau', 'seed', 'T', 'dkl'],
                    'cache/stabilizer_agent_landscape.p', True)

    # dissipator
    if run_dissipator:
        learners = {}
        dkl = {}
        for scale, nBatch in product(scaleRange, nBatchRange):
            # environment properties
            kwargs = {'noise':{'type':'binary', 'scale':scale, 'weight':.95, 'v':-.01},
                      'T':T,
                      'nBatch':nBatch}

            kwargs['rng'] = np.random.RandomState(seed)
            kwargs['noise']['tau'] = tau
            learner = agent.Stigmergy(**kwargs)
            dkl[(scale, nBatch)] = learner.learn(betaRange, n_cpus=cpu_count()-1, save=False)
            learners[(scale, nBatch)] = learner

        print("Saving dissipator simulation.")
        print()
        save_pickle(['learners', 'scaleRange', 'nBatchRange', 'betaRange', 'tau', 'seed', 'T', 'dkl'],
                    'cache/dissipator_agent_landscape.p', True)

def tau_range_eigen(run_passive=True, run_dissipator=True, run_stabilizer=True):
    tauRange = np.logspace(0, 5, 13)  # time scale for flipping external field
    h0 = .2  # magnitude of external field
    nBatch = 1_000
    degfit = 30
    betaRange = lobatto_beta(degfit)
    weight = .95
    
    if run_passive:
        solvers = {}
        edkl = {}
        errs = {}

        for tau in tauRange:
            # recursive solution
            solvers[tau] = eigen.Vision(tau, h0, 0, nBatch, L=.5, dx=2.5e-4)
            edkl[tau], errs[tau] = solvers[tau].dkl(betaRange)
            print("Done with tau=%E."%tau)
            
        save_pickle(['edkl', 'errs', 'betaRange', 'h0', 'tauRange'],
                    'cache/eigen_passive_tau_range.p', True)
    
    if run_dissipator:
        v = -.01
        solvers = {}
        edkl = {}
        errs = {}
        cost = {}

        for tau in tauRange:
            solvers[tau] = eigen.Stigmergy(tau, h0, 0, nBatch, L=.5, v=v, weight=weight, dx=2.5e-4)
            edkl[tau], errs[tau], cost[tau] = solvers[tau].dkl(betaRange)
            print("Done with tau=%E."%tau)
            
        varlist = ['edkl', 'errs', 'cost', 'betaRange', 'h0', 'nBatch', 'tauRange']
        save_pickle(varlist, 'cache/eigen_dissipator_tau_range.p', True)

    if run_stabilizer:
        v = .01
        solvers = {}
        edkl = {}
        errs = {}
        cost = {}

        for tau in tauRange:
            solvers[tau] = eigen.Stigmergy(tau, h0, 0, nBatch, L=.5, v=v, weight=weight, dx=2.5e-4)
            edkl[tau], errs[tau], cost[tau] = solvers[tau].dkl(betaRange)
            print("Done with tau=%E."%tau)
            
        varlist = ['edkl', 'errs', 'cost', 'betaRange', 'h0', 'nBatch', 'tauRange']
        save_pickle(varlist, 'cache/eigen_stabilizer_tau_range.p', True)
