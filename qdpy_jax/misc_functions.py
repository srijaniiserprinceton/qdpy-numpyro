from collections import namedtuple

def create_namedtuple(tname, keys, values):
    NT = namedtuple(tname, keys)
    nt = NT(*values)
    return nt

def tree_map_CNM_AND_NBS(CNM_AND_NBS):
    # converting to tuples and nestes tuples for easy of passing                              
    nl_nbs = tuple(map(tuple, CNM_AND_NBS.nl_nbs))
    nl_nbs_idx = tuple(CNM_AND_NBS.nl_nbs_idx)
    omega_nbs = tuple(CNM_AND_NBS.omega_nbs)

    CENMULT_AND_NBS = create_namedtuple('CENMULT_AND_NBS',
                                        ['nl_nbs',
                                         'nl_nbs_idx',
                                         'omega_nbs'],
                                        (nl_nbs,
                                         nl_nbs_idx,
                                         omega_nbs))

    return CENMULT_AND_NBS
