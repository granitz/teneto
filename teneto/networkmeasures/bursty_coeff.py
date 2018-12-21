"""
networkmeasures.bursty_coeff
"""

import numpy as np
from .intercontacttimes import intercontacttimes

def bursty_coeff(data, calc='edge', nodes='all', communities=None):
    """
    Calculates the bursty coefficient.[1][2]

    Parameters
    ----------

    data : array, dict
        This is either (1) temporal network input (graphlet or contact) with nettype: 'bu', 'bd'. (2) dictionary of ICTs (output of *intercontacttimes*).

    calc : str
        Caclulate the bursty coeff over what. Options include 'edge': calculate B on all ICTs between node i and j. (Default); 'node': caclulate B on all ICTs connected to node i.;
        'communities': calculate B for each communities (argument communities then required);
        'meanEdgePerNode': first calculate the ICTs between node i and j, then take the mean over all j.

    nodes: list or str
        Options: 'all': do for all nodes (default) or list of node indexes to calculate.

    communities : array, optional
        None (default) or Nx1 vector of communities assignment. This returns a "centrality" per communities instead of per node.


    Returns
    -------
    B : array
        Bursty coefficienct per (edge or node measure). 


    Notes 
    ------ 

    The burstiness coefficent, B, is defined in refs [1,2] as:  

    .. math:: B = {{\sigma_{ICT} - \mu_{ICT}} \over {\sigma_{ICT} + \mu_{ICT}}}

    Where :math:`\sigma_{ICT}` and :math:`\mu_{ICT}` are the standard deviation and mean of the inter-contact times respectively (see teneto.networkmeasures.intercontacttimes)

    When B > 0, indicates bursty intercontact times. When B < 0, indicates periodic/tonic intercontact times. When B = 0, indicates random.


    Examples
    ---------

    First import all necessary packages 

    >>> import teneto  
    >>> import numpy as np 

    Now create 2 temporal network of 2 nodes and 60 time points. The first has periodict edges, repeating every other time-point:

    >>> G_periodic = np.zeros([2, 2, 60])
    >>> ts_periodic = np.arange(0, 60, 2)
    >>> G_periodic[:,:,ts_periodic] = 1

    The second has a more bursty pattern of edges: 

    >>> ts_bursty = [1, 8, 9, 32, 33, 34, 39, 40, 50, 51, 52, 55]
    >>> G_bursty = np.zeros([2, 2, 60])
    >>> G_bursty[:,:,ts_bursty] = 1

    Now we call bursty_coeff. 

    >>> B_periodic = teneto.networkmeasures.bursty_coeff(G_periodic)
    >>> B_periodic
    array([[nan, -1.],
        [-1., nan]])

    Above we can see that between node 0 and 1, B=-1 (the diagonal is nan). 
    Doing the same for the second example: 

    >>> B_bursty = teneto.networkmeasures.bursty_coeff(G_bursty)
    >>> B_bursty
    array([[nan, 0.13311003],
        [0.13311003, nan]])

    gives a positive value, indicating the inter-contact times between node 0 and 1 is bursty.

    References 
    ----------

    .. [1] Goh, KI & Barabasi, AL (2008) Burstiness and Memory in Complex Systems. EPL (Europhysics Letters), 81: 4 [`Link <https://arxiv.org/pdf/physics/0610233.pdf>`_]
    .. [2] Holme, P & Saramäki J (2012) Temporal networks. Physics Reports. 519: 3. [`Link <https://arxiv.org/pdf/1108.1780.pdf>`_] (Discrete formulation used here) 

    """

    if calc == 'communities' and not communities:
        raise ValueError("Specified calc='communities' but no communities argument provided (list of clusters/modules)")

    ict = 0  # are ict present
    if isinstance(data, dict):
        # This could be done better
        if [k for k in list(data.keys()) if k == 'intercontacttimes'] == ['intercontacttimes']:
            ict = 1
    # if shortest paths are not calculated, calculate them
    if ict == 0:
        data = intercontacttimes(data)

    ict_shape = data['intercontacttimes'].shape

    if len(ict_shape) == 2:
        node_len = ict_shape[0] * ict_shape[1]
    elif len(ict_shape) == 1:
        node_len = 1
    else:
        raise ValueError('more than two dimensions of intercontacttimes')

    if isinstance(nodes, list) and len(ict_shape) > 1:
        node_combinations = [[list(set(nodes))[t], list(set(nodes))[tt]] for t in range(
            0, len(nodes)) for tt in range(0, len(nodes)) if t != tt]
        do_nodes = [np.ravel_multi_index(n, ict_shape) for n in node_combinations]
    else:
        do_nodes = np.arange(0, node_len)

    # Reshae ICTs
    if calc == 'node':
        ict = np.concatenate(data['intercontacttimes'][do_nodes, do_nodes], axis=1)
    elif calc == 'communities':
        unique_communities = np.unique(communities)
        ict_shape = (len(unique_communities),len(unique_communities))
        ict = np.array([[None] * ict_shape[0]] * ict_shape[1])
        for i, s1 in enumerate(unique_communities):
            for j, s2 in enumerate(unique_communities):
                if s1 == s2:
                    ind = np.triu_indices(sum(communities==s1),k=1)
                    ict[i,j] = np.concatenate(data['intercontacttimes'][ind[0],ind[1]])
                else:
                    ict[i,j] = np.concatenate(np.concatenate(data['intercontacttimes'][communities==s1,:][:,communities==s2]))
        # Quick fix, but could be better
        data['intercontacttimes'] = ict
        do_nodes = np.arange(0,ict_shape[0]*ict_shape[1])

    if len(ict_shape) > 1:
        ict = data['intercontacttimes'].reshape(ict_shape[0] * ict_shape[1])
        b_coeff = np.zeros(len(ict)) * np.nan
    else:
        b_coeff = np.zeros(1) * np.nan
        ict = [data['intercontacttimes']]

    for i in do_nodes:
        if isinstance(ict[i],np.ndarray):
            mu_ict = np.mean(ict[i])
            sigma_ict = np.std(ict[i])
            b_coeff[i] = (sigma_ict - mu_ict) / (sigma_ict + mu_ict)
        else:
            b_coeff[i] = np.nan

    if len(ict_shape) > 1:
        b_coeff = b_coeff.reshape(ict_shape)
    return b_coeff
