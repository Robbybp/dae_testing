from pyomo.core.base.component import ComponentData
from pyomo.dae.flatten import flatten_dae_components

EXPLICIT_SCHEMES = {
    "FORWARD Difference",
}
IMPLICIT_SCHEMES = {
    "BACKWARD Difference",
    "LAGRANGE-LEGENDRE",
    "LAGRANGE-RADAU",
}
CENTRAL_SCHEMES = {
    "CENTRAL Difference",
}


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


def get_non_collocation_finite_element_points(contset):
    fe_points = contset.get_finite_elements()
    n_fep = len(fe_points)
    disc_info = contset.get_discretization_info()
    scheme = disc_info["scheme"]
    if "tau_points" in disc_info:
        # Normalized collocation points in each finite element
        #
        # This list seems to always include zero, for some reason.
        # But we don't consider zero to be a collocation point (except
        # maybe in an explicit discretization).
        colloc_in_fe = disc_info["tau_points"][1:]
    else:
        if scheme in EXPLICIT_SCHEMES:
            colloc_in_fe = [0.0]
        elif scheme in IMPLICIT_SCHEMES:
            colloc_in_fe = [1.0]
        elif scheme in CENTRAL_SCHEMES:
            raise ValueError(
                "Central discretization schemes are not supported."
            )
        else:
            raise ValueError(
                "Discretization scheme %s not recognized" % scheme
            )
    colloc_in_fe_set = set(colloc_in_fe)
    include_first = (0.0 in colloc_in_fe_set)
    include_last = (1.0 in colloc_in_fe_set)
    include_interior_fe_point = (include_first or include_last)
    colloc_fe_points = [ 
        # FE points that are also collocation points
        p for i, p in enumerate(fe_points)
        if (
            (i == 0 and include_first)
            or (i == n_fep - 1 and include_last)
            or (i != 0 and i != n_fep - 1 and include_interior_fe_point)
        )
    ]   
    colloc_fe_point_set = set(colloc_fe_points)
    non_colloc_fe_points = [ 
        p for p in fe_points if p not in colloc_fe_point_set
    ]   
    return non_colloc_fe_points
