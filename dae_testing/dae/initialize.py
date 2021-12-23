from pyomo.core.base.component import ComponentData
from pyomo.dae.flatten import flatten_dae_components

def is_discretization(con):
    # TODO: discretization of a particular variable?
    # with respect to a particular set?
    # Note that this function does not properly identify a discretization
    # equation if it is contained in a reference constraint.
    if isinstance(con, ComponentData):
        name = con.parent_component().local_name
    else:
        name = con.local_name
    return "_disc_eq" in name


def identify_constraints_at_time(m, time, t0, dae_cons=None):
    if dae_cons is None:
        scalar_cons, dae_cons = flatten_dae_components(m, time)
    return [con[t0] for t in dae_cons]
