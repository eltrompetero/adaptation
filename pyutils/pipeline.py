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



def tau_range(run_passive=True, run_stabilizer=True, run_destabilizer=True):
    """Agent unfitness landscape for environments changing with timescale. From "agents in
    binary environment.ipynb", using ABS.

    Parameters
    ----------
    run_passive : bool, True
    run_stabilizer : bool, True
    run_destabilizer : bool, True
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

    # destabilizer
    if run_destabilizer:
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
                    'cache/destabilizer_agent_sim_tau_range.p', True)

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

def info_gain(run_passive=True, run_stabilizer=True, run_destabilizer=True):
    """Landscapes for optimal memory and memory/forgetting tradeoff. From "info
    gain.ipynb."

    Parameters
    ----------
    run_passive : bool, True
    run_stabilizer : bool, True
    run_destabilizer : bool, True
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

    # destabilizer
    if run_destabilizer:
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

        print("Saving destabilizer simulation.")
        print()
        save_pickle(['learners', 'scaleRange', 'nBatchRange', 'betaRange', 'tau', 'seed', 'T', 'dkl'],
                    'cache/destabilizer_agent_landscape.p', True)

def tau_range_eigen(run_passive=True, run_destabilizer=True, run_stabilizer=True):
    """Agent unfitness as a function of environmental timescale using eigenfunction solution
    method.

    Parameters
    ----------
    run_passive : bool, True
    run_destabilizer : bool, True
    run_stabilizer : bool, True
    """

    tauRange = np.logspace(0, 5, 13)  # time scale for flipping external field
    h0 = .2  # magnitude of external field
    nBatch = 1_000
    degfit = 40
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
    
    if run_destabilizer:
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
        save_pickle(varlist, 'cache/eigen_destabilizer_tau_range.p', True)

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

def effective_timescales_stabilizer():
    """Comparing divergence profiles for stabilizer with passive agent at 
    effective timescales.
    """
    
    # set agent/env properties
    h0 = .2
    nBatch = 1_000

    # specify range to solve for
    betaRange = lobatto_beta(55)
    
    def loop_wrapper(tau):
        """Inner function to run eigenfunction solution procedure on 
        stabilizer and passive agents."""
        
        # solve
        solver = eigen.Stigmergy(tau, h0, 0, nBatch, weight=.95, v=.01)
        dkl, errs, cost = solver.dkl(betaRange,
                                     iprint=False,
                                     dx_res_factor=8)

        # average decay rate once having accounted for stabilization
        meanTau = np.zeros_like(betaRange)
        for i, beta in enumerate(betaRange):
            p, _, scost, x = solver.cache_phatpos[beta]
            # this is just like weighting each observation directly by probability
            # assuming that we have a discrete space
            # seems like it's missing a jacobian, but it works really well
            # meanTau[i] = 1/(1 - trapz(solver.binary_env_stay_p(x-h0) * p, x))
            meanTau[i] = 1/(1 - solver.binary_env_stay_p(x-h0).dot(p/p.sum()))
        
        # passive agent shifted to new effective timescale
        vdkl = np.zeros_like(betaRange)
        verrs = np.zeros_like(betaRange)

        for i in range(betaRange.size):
            vsolver = eigen.Passive(meanTau[i], h0, 0, nBatch)
            vdkl[i], verrs[i] = vsolver.dkl(np.array([betaRange[i]]),
                                            iprint=False,
                                            dx_res_factor=2)

        # what passive looks like if not accounting for new averaged time scale
        ovsolver = eigen.Vision(tau, h0, 0, nBatch)
        ovdkl, overrs = ovsolver.dkl(betaRange,
                                     dx_res_factor=2,
                                     iprint=False)
        
        return (dkl, errs), (vdkl, verrs), (ovdkl, overrs)
    
    tauRange = [10, 50, 250, 1250]
    dkl = {}
    vdkl = {}
    ovdkl = {}
    for tau in tauRange:
        dkl[tau], vdkl[tau], ovdkl[tau] = loop_wrapper(tau)        
        print(f"Done with {tau}.")
        
    save_pickle(['dkl','ovdkl','vdkl','betaRange','h0','nBatch'],
                'cache/effective_timescales_stabilizer.p', True)

def effective_timescales_destabilizer():
    """Comparing divergence profiles for stabilizer with passive agent at 
    effective timescales.
    """
    
    # set agent/env properties
    h0 = .2
    nBatch = 1_000

    # specify range to solve for
    betaRange = lobatto_beta(45)
    
    def loop_wrapper(tau):
        """Inner function to run eigenfunction solution procedure on 
        stabilizer and passive agents."""
        
        # solve
        solver = eigen.Stigmergy(tau, h0, 0, nBatch, weight=.95, v=-.01)
        dkl, errs, cost = solver.dkl(betaRange,
                                     iprint=False,
                                     dx_res_factor=4)

        # average decay rate once having accounted for stabilization
        meanTau = np.zeros_like(betaRange)
        for i, beta in enumerate(betaRange):
            p, _, scost, x = solver.cache_phatpos[beta]
            # this is just like weighting each observation directly by probability
            # assuming that we have a discrete space
            # seems like it's missing a jacobian, but it works really well
            # meanTau[i] = 1/(1 - trapz(solver.binary_env_stay_p(x-h0) * p, x))
            meanTau[i] = 1/(1 - solver.binary_env_stay_p(x-h0).dot(p/p.sum()))
        
        # passive agent shifted to new effective timescale
        vdkl = np.zeros_like(betaRange)
        verrs = np.zeros_like(betaRange)

        for i in range(betaRange.size):
            vsolver = eigen.Passive(meanTau[i], h0, 0, nBatch)
            vdkl[i], verrs[i] = vsolver.dkl(np.array([betaRange[i]]),
                                            iprint=False,
                                            dx_res_factor=2)

        # what passive looks like if not accounting for new averaged time scale
        ovsolver = eigen.Vision(tau, h0, 0, nBatch)
        ovdkl, overrs = ovsolver.dkl(betaRange,
                                     dx_res_factor=2,
                                     iprint=False)
        
        return (dkl, errs), (vdkl, verrs), (ovdkl, overrs)
    
    tauRange = [10, 50, 250, 1250, 6250]
    dkl = {}
    vdkl = {}
    ovdkl = {}
    for tau in tauRange:
        dkl[tau], vdkl[tau], ovdkl[tau] = loop_wrapper(tau)        
        print(f"Done with {tau}.")
        
    save_pickle(['dkl','ovdkl','vdkl','betaRange','h0','nBatch'],
                'cache/effective_timescales_destabilizer.p', True)

def costs_example():
    """Example of divergence landscape with algorithmic costs.
    """

    degfit = 35
    betaRange = lobatto_beta(degfit)

    learner = eigen.Stigmergy(100, .2, 0, 1_000, L=.5, dx=2.5e-4, weight=.95, v=.01)
    dkl, errs, cost = learner.dkl(betaRange)

    betaPlot = linspace_beta(1e-2, 1e3, 200)
    save_pickle(['dkl', 'betaRange', 'betaPlot', 'errs', 'cost'],
                 'plotting/cost_tradeoff_example.p', True)

