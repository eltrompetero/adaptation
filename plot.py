# ====================================================================================== #
# Module for plotting graphs. 
# 
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from .utils import *
import matplotlib.pyplot as plt
from misc.plot import colorcycle
import pickle



def tau_range(agent_type, show_legend=True):
    """Show example comparison of unfitness landscape for various tau for eigenvalue formulation and
    agent-based simulation.

    Parameters
    ----------
    agent_type : str
        'passive', 'dissipator', 'stabilizer'
    show_legend : bool, True

    Returns
    -------
    matplotlib.pyplot.Figure
    """

    assert agent_type in ['passive','dissipator','stabilizer']

    loaddata = pickle.load(open('cache/%s_agent_sim_tau_range.p'%agent_type, 'rb'))
    betaRange = loaddata['betaRange']
    dkl = loaddata['dkl']

    fig, ax = plt.subplots()

    ccycle = colorcycle(len(dkl), cmap=plt.cm.viridis)
    for i, mdkl in enumerate(dkl.values()):
         ax.loglog(-1/np.log(betaRange), mdkl, '-',
                   c=next(ccycle))

    loaddata = pickle.load(open('cache/eigen_%s_tau_range.p'%agent_type,'rb'))
    betaRange = loaddata['betaRange']
    edkl = loaddata['edkl']
    errs = loaddata['errs']
    tauRange = loaddata['tauRange']

    ccycle = colorcycle(len(edkl)//2+1, cmap=plt.cm.viridis)
    ix = (-1/np.log(betaRange))<10
    for edkl_,errs_ in list(zip(edkl.values(),errs.values()))[::2]:
        c = next(ccycle)
        spline = interpolate(betaRange, edkl_, .2, errs_.ravel())
        
        x = linspace_beta(.1, 1000, 100)
        ax.plot(-1/np.log(x), spline(x), '--', c=c)

    ax.set(xlabel=r'agent memory $\tau$',
           ylabel=r'unfitness $D$',
           xlim=-1/np.log(betaRange[[0,-1]]), ylim=(1e-6, .1))

    if show_legend:
        leg = ax.legend([r'$%1.1f$'%(np.log10(i)) for i in tauRange],
                  fontsize='xx-small',
                  loc=4,
                  ncol=3,
                  handlelength=.6,
                  columnspacing=.7,
                  handletextpad=.2,)
        leg.set_title('env. persistence\n'+r'        $\log_{10}\tau_{\rm e}$', prop={'size':'xx-small'})

    return fig
