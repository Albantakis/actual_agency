from analysis import *

from matplotlib import pyplot as plt
import matplotlib as mpl

# Setting colors used in plotting functions
blue, red, green, grey, white = '#77b3f9', '#f98e81', '#8abf69', '#adadad', '#ffffff'
purple, yellow = '#d279fc', '#f4db81'

### PLOTTING FUNCTIONS

def plot_ACanalysis(animat, world, trial, t, causal_chain=None, plot_state_history=False,
                    causal_history=None,causal_history_motor=None, plot_causal_account=True,
                    plot_state_transitions=True, n_backsteps=None):

    '''
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    '''

    n_sensors = animat.n_sensors
    n_nodes = animat.n_nodes
    # PLOT SCREEN
    _, block = world._get_initial_condition(trial)
    fullgame_history = world.get_fullgame_history()
    wins = world.wins

    n_cols = 4
    if plot_state_history:
        n_cols +=1
    if causal_history is not None:
        n_cols +=1
    if causal_history_motor is not None:
        n_cols +=1
    if causal_chain is not None:
        n_cols +=1

    n_rows = 2
    col=0

    plt.subplot2grid((2,n_cols),(0,col),rowspan=2, colspan=2)
    col+=2
    plt.imshow(fullgame_history[trial,t,:,:],cmap=plt.cm.binary);
    plt.xlabel('x'); plt.ylabel('y')
    win = 'WON' if wins[trial] else 'LOST'
    direction = '━▶' if block.direction=='right' else '◀━'
    plt.title('Game - Trial: {}, {}, {}, {}'.format(block.size,block.type,direction,win))

    # PLOT CAUSAL HISTORIES
    labels = ['S1', 'S2', 'M1', 'M2', 'A', 'B', 'C', 'D'] if n_sensors==2 else ['S1', 'M1', 'M2', 'A', 'B', 'C', 'D']

    if plot_state_history:
        plt.subplot2grid((2,n_cols),(0,col),rowspan=2)
        col+=1
        # plt.colorbar(shrink=0.2)
        plt.xticks(range(n_nodes),labels)
        plt.xlabel('Node'); plt.ylabel('Time')
        plt.title('Brain states history')
        plt.imshow(animat.brain_activity[trial],cmap=plt.cm.binary)
        plt.colorbar(shrink=0.1)
        plt.axhline(t-0.5,color=red)
        plt.axhline(t+0.5,color=red)

    if causal_history is not None:
        plt.subplot2grid((2,n_cols),(0,col),rowspan=2)
        col+=1
        plt.imshow(causal_history, cmap=plt.cm.magma_r)
        plt.colorbar(shrink=0.2)
        plt.xticks(range(n_nodes),labels)
        plt.xlabel('Node'); plt.ylabel('Time')
        plt.title('Causal history')
    if causal_history_motor is not None:
        plt.subplot2grid((2,n_cols),(0,col),rowspan=2)
        col+=1
        plt.imshow(causal_history_motor, vmax=6, cmap=plt.cm.magma_r)
        plt.colorbar(shrink=0.2)
        plt.xticks(range(n_nodes),labels)
        plt.xlabel('Node')
        plt.title('Causal history of M1,M2')

    # PLOT BACKTRACKING
    if causal_chain is not None:
        BT = get_backtrack_array(causal_chain,n_nodes=n_nodes)
        if n_backsteps is None:
            n_backsteps = len(causal_chain)
        BT = BT[-n_backsteps:]
        S = np.zeros((world.height,n_nodes))
        ixs = [0,3,4,5,6] if n_sensors==1 else [0,1,4,5,6,7]
        S[t-n_backsteps:t,ixs] = BT

        plt.subplot2grid((2,n_cols),(0,col),rowspan=2)
        col+=1
        plt.imshow(S,vmin=0,vmax=np.max(BT),cmap=plt.cm.magma_r)
        plt.colorbar(shrink=0.2)

        labels = ['S1', 'S2', 'M1', 'M2', 'A', 'B', 'C', 'D'] if n_sensors==2 else ['S1', 'M1', 'M2', 'A', 'B', 'C', 'D']
        plt.xticks(range(n_nodes),labels)
        plt.xlabel('Node')
        plt.title('Backtracking of M1,M2')

    # PLOT ANIMAT BRAIN
    if plot_state_transitions:
        transition = animat.get_transition(trial,t,False)
        if animat.n_nodes==8:
            cause_ixs = [0,1,4,5,6,7]
            effect_ixs = [2,3]
        else:
            cause_ixs = [0,3,4,5,6]
            effect_ixs = [1,2]
        T = pyphi.actual.Transition(animat.brain, transition[0], transition[1], cause_ixs, effect_ixs)
        account = pyphi.actual.account(T, direction=pyphi.Direction.CAUSE)
        causes = account.irreducible_causes

        plt.subplot2grid((2,n_cols),(0,col),colspan=2)
        animat.plot_brain(transition[0])
        plt.title('Brain\n\nT={}'.format(t-1),y=0.85)

        if plot_causal_account:
            plt.text(0,0,causes,fontsize=12)


        plt.subplot2grid((2,n_cols),(1,col),colspan=2)
        animat.plot_brain(transition[1])
        plt.title('T={}'.format(t),y=0.85)

        plt.text(-2,6,transition_str(transition),fontsize=12)

    else:
        state = animat.get_state(trial,t)
        plt.subplot2grid((18,n_cols),(0,col),colspan=2,rowspan=9)
        animat.plot_brain(state)
        plt.subplot2grid((18,n_cols),(9,col),colspan=2)
        plt.imshow(np.array(state)[np.newaxis,:],cmap=plt.cm.binary)
        plt.yticks([])
        plt.xticks(range(animat.n_nodes),labels)
    # plt.tight_layout()

def ac_plot_brain(cm, graph=None, state=None, ax=None):
    '''
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    '''

    n_nodes = cm.shape[0]
    if n_nodes==7:
        labels = ['S1','M1','M2','A','B','C','D']
        pos = {'S1': (5,40), #'S2': (20, 40),
           'A': (0, 30), 'B': (20, 30),
           'C': (0, 20), 'D': (20, 20),
          'M1': (5,10), 'M2': (15,10)}
        nodetype = (0,1,1,2,2,2,2)

        ini_hidden = 3

    elif n_nodes==8:
        labels = ['S1','S2','M1','M2','A','B','C','D']
        pos = {'S1': (5,40), 'S2': (15, 40),
           'A': (0, 30), 'B': (20, 30),
           'C': (0, 20), 'D': (20, 20),
          'M1': (5,10), 'M2': (15,10)}
        nodetype = (0,0,1,1,2,2,2,2)
        ini_hidden = 4

    if graph is None:
        graph = nx.from_numpy_matrix(cm, create_using=nx.DiGraph())
        mapping = {key:x for key,x in zip(range(n_nodes),labels)}
        graph = nx.relabel_nodes(graph, mapping)

    state = [1]*n_nodes if state==None else state

    blue, red, green, grey, white = '#6badf9', '#f77b6c', '#8abf69', '#adadad', '#ffffff'
    blue_off, red_off, green_off, grey_off = '#e8f0ff','#ffe9e8', '#e8f2e3', '#f2f2f2'

    colors = np.array([red, blue, green, grey, white])
    colors = np.array([[red_off,blue_off,green_off, grey_off, white],
                       [red,blue,green, grey, white]])

    node_colors = [colors[state[i],nodetype[i]] for i in range(n_nodes)]
    # Grey Uneffective or unaffected nodes
    cm_temp = copy.copy(cm)
    cm_temp[range(n_nodes),range(n_nodes)]=0
    unaffected = np.where(np.sum(cm_temp,axis=0)==0)[0]
    uneffective = np.where(np.sum(cm_temp,axis=1)==0)[0]
    noeffect = list(set(unaffected).union(set(uneffective)))
    noeffect = [ix for ix in noeffect if ix in range(ini_hidden,ini_hidden+4)]
    node_colors = [node_colors[i] if i not in noeffect else colors[state[i],3] for i in range(n_nodes)]

    #   White isolate nodes
    isolates = [x for x in nx.isolates(graph)]
    node_colors = [node_colors[i] if labels[i] not in isolates else colors[0,4] for i in range(n_nodes)]

    self_nodes = [labels[i] for i in range(n_nodes) if cm[i,i]==1]
    linewidths = [2.5 if labels[i] in self_nodes else 1 for i in range(n_nodes)]

#     fig, ax = plt.subplots(1,1, figsize=(4,6))
    nx.draw(graph, with_labels=True, node_size=800, node_color=node_colors,
    edgecolors='#000000', linewidths=linewidths, pos=pos, ax=ax)


def plot_mean_with_errors(x, y, yerr, color, label=None, linestyle=None):
    '''
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    '''

    if len(yerr)==2: # top and bottom percentiles
        plt.fill_between(x, yerr[0], yerr[1], color=color, alpha=0.1)
    else:
        plt.fill_between(x, y-yerr, y+yerr, color=color, alpha=0.1)
    plt.plot(x, y, label=label, color=color, linestyle=linestyle)


def plot_LODdata_and_Bootstrap(x,LODdata,label='data',color='b',linestyle='-',figsize=[20,10]):
    '''
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    '''

    fit = Bootstrap_mean(LODdata,500)
    m_fit = np.mean(fit,0)
    s_fit = np.std(fit,0)
    fig = plt.figure(figsize=figsize)
    for LOD in LODdata:
        plt.plot(x,LOD,'r',alpha=0.1)
    plt.fill_between(x, m_fit-s_fit, m_fit+s_fit, color=color, alpha=0.2)
    plt.plot(x, m_fit, label=label, color=color, linestyle=linestyle)

    return fig



def plot_2LODdata_and_Bootstrap(x,LODdata1,LODdata2,label=['data1','data2'],color=['k','y'],linestyle='-',figsize=[20,10],fig=None,subplot=111):
    '''
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    '''

    fit1 = Bootstrap_mean(LODdata1,500)
    m_fit1 = np.mean(fit1,0)
    s_fit1 = np.std(fit1,0)
    fit2 = Bootstrap_mean(LODdata2,500)
    m_fit2 = np.mean(fit2,0)
    s_fit2 = np.std(fit2,0)
    if fig==None:
        fig = plt.figure(figsize=figsize)
    plt.subplot(subplot)
    for LOD1,LOD2 in zip(LODdata1,LODdata2):
        plt.plot(x,LOD1,color[0],alpha=0.1)
        plt.plot(x,LOD2,color[1],alpha=0.1)
    plt.fill_between(x, m_fit1-s_fit1, m_fit1+s_fit1, color=color[0], alpha=0.2)
    plt.plot(x, m_fit1, label=label[0], color=color[0], linestyle=linestyle)
    plt.fill_between(x, m_fit2-s_fit2, m_fit2+s_fit2, color=color[1], alpha=0.2)
    plt.plot(x, m_fit2, label=label[1], color=color[1], linestyle=linestyle)

    return fig

def plot_multiple(x,data=[],title=None,label=None,colormap=None,linestyle='-',figsize=[20,10],fig=None,subplot=111):
    '''
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    '''
    n_datasets = len(data)
    # creating colormaps
    if colormap==None:
        color = mpl.cm.get_cmap('rainbow',n_datasets)
    else:
        color = mpl.cm.get_cmap(colormap,n_datasets)

    # doing bootstrapping and PLOTTING
    if fig==None:
        fig = plt.figure(figsize=figsize)
    plt.subplot(subplot)

    for n in range(n_datasets):
        d = data[n]
        fit = Bootstrap_mean(d,500)
        m_fit = np.mean(fit,0)
        s_fit = np.std(fit,0)

        for raw in d:
            plt.plot(x,raw,color=color(n),alpha=0.1)

        plt.fill_between(x, m_fit-s_fit*1.96, m_fit+s_fit*1.96, color=color(n), alpha=0.2)
        if not label==None:
            plt.plot(x, m_fit, label=label[n],color=color(n), linestyle=linestyle)
            plt.legend()
        else:
            plt.plot(x, m_fit, color=color(n), linestyle=linestyle)
        if not title==None:
            plt.title(title)

    return fig


def hist2d_2LODdata(LODdata1x,LODdata1y,LODdata2x,LODdata2y, nbins=20):
    '''
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    '''

    xmin = np.min((np.min(LODdata1x),np.min(LODdata2x)))
    xmax = np.max((np.max(LODdata1x),np.max(LODdata2x)))
    ymin = np.min((np.min(LODdata1y),np.min(LODdata2y)))
    ymax = np.max((np.max(LODdata1y),np.max(LODdata2y)))

    xbins = np.linspace(xmin,xmax,nbins)
    ybins = np.linspace(ymin,ymax,nbins)
    plt.figure()
    plt.subplot(121)
    plt.hist2d(np.ravel(LODdata1x),np.ravel(LODdata1y),[xbins,ybins],norm=mpl.colors.LogNorm())
    plt.subplot(122)
    plt.hist2d(np.ravel(LODdata2x),np.ravel(LODdata2y),[xbins,ybins],norm=mpl.colors.LogNorm())

def plot_mean_with_errors(x, y, yerr, color, label=None, linestyle=None):
    '''
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    '''

    plt.fill_between(x, y-yerr, y+yerr, color=color, alpha=0.1)
    plt.plot(x, y, label=label, color=color, linestyle=linestyle)

def plot_2Ddensity(x,y, plot_samples=True, cmap=plt.cm.Blues, color=None, markersize=0.7):
    '''
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    '''

    data = np.c_[x,y]
    k = kde.gaussian_kde(data.T)
    nbins = 20
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=cmap)

    plt.plot(x,y,'.', color=color, markersize=markersize)
