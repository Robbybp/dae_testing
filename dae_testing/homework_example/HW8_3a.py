# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 17:03:25 2021

@author: ssnaik
"""

from pyomo.environ import *
from pyomo.dae import *
import numpy as np
import matplotlib.pyplot as plt

def make_model():
    model = m = ConcreteModel()

    m.tf = Param(initialize = 1)
    m.t = ContinuousSet(bounds=(0, m.tf))
    m.u = Var(m.t)

    m.x1 = Var(m.t)
    m.x2 = Var(m.t)
    m.x3 = Var(m.t)

    m.dx1dt = DerivativeVar(m.x1, wrt = m.t)
    m.dx2dt = DerivativeVar(m.x2, wrt = m.t)
    m.dx3dt = DerivativeVar(m.x3, wrt = m.t)

    m.obj = Objective(expr=m.x3[m.tf])

    def _x1dot(m, t):
        return m.dx1dt[t] == m.x2[t]
    m.x1dot = Constraint(m.t, rule=_x1dot)

    def _x2dot(m, t):
        return m.dx2dt[t] == -m.x2[t] + m.u[t]
    m.x2dot = Constraint(m.t, rule=_x2dot)

    def _x3dot(m, t):
        return m.dx3dt[t] == m.x1[t]**2 + m.x2[t]**2 + 0.005*m.u[t]**2
    m.x3dot = Constraint(m.t, rule=_x3dot)

    def _init(m):
        yield m.x1[0] == 0
        yield m.x2[0] == -1
        yield m.x3[0] == 0
    m.init_conditions = ConstraintList(rule=_init)

    return m


def discretize_model(
        m,
        method="dae.collocation",
        scheme="LAGRANGE-RADAU",
        nfe=2,
        ncp=5,
        ):
    discretizer = TransformationFactory(method)
    kwds = {"nfe": nfe, "scheme": scheme}
    if method == "dae.collocation":
        kwds["ncp"] = ncp
    discretizer.apply_to(m, **kwds)


def discretization_points(m):
    return m.t.get_finite_elements(), m.t._fe


def solve_model(m, tee=True):
    ipopt = SolverFactory("ipopt")
    ipopt.solve(m, tee=tee)


def display_values_and_plot(m, file_prefix=None):
    print("final time = %6.2f" %(value(m.x3[m.tf])))
    x1_list = []
    x3_list = []
    x2_list = []

    # u might not be defined for all time
    u_t_list = []
    u_list = []
    for t in m.t: 
        x1 = value(m.x1[t])
        x1_list.append(x1)

        x2 = value(m.x2[t])
        x2_list.append(x2)

        x3 = value(m.x3[t])
        x3_list.append(x3)

        if m.u[t].value is not None:
            u = value(m.u[t])
            u_list.append(u)
            u_t_list.append(t)

    if m.u[0].value is not None:
        print(value(m.u[0]))
    else:
        print("value of %s is None" % m.u[0].name)

    plt.figure(1)
    plt.plot(m.t,x1_list,'-v',label = 'x1')
    plt.plot(m.t,x2_list,'-.',label = 'x2')
    #plt.plot(m.t,x3_list,'-o',label = 'x3')
    plt.xlabel('t')
    plt.ylabel('x (t)')
    plt.title('State profile')
    plt.legend()
    state_fname = "state_profile.png"
    if file_prefix is not None:
        state_fname = file_prefix + state_fname
    plt.savefig(state_fname)

    plt.figure(2)
    #plt.plot(m.t,u_list,'-*')
    plt.plot(u_t_list, u_list, '-*')
    plt.xlabel('t')
    plt.ylabel('u (t)')
    plt.title('Control Profile')
    control_fname = "control_profile.png"
    if file_prefix is not None:
        control_fname = file_prefix + control_fname
    plt.savefig(control_fname)


def get_non_collocation_finite_element_points(contset):
    fe_points = contset.get_finite_elements()
    n_fep = len(fe_points)
    disc_info = contset.get_discretization_info()
    if "tau_points" in disc_info:
        # Normalized collocation points in each finite element
        #
        # This list seems to always include zero, for some reason.
        # But we don't consider zero to be a collocation point (except
        # maybe in an explicit discretization).
        colloc_in_fe = disc_info["tau_points"][1:]
    else:
        # TODO: what case are we covering here?
        # This depends on the discretization...
        # BACKWARD: [1.0], FORWARD: [0.0]
        colloc_in_fe = [1.0]
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


def solve_and_plot_results(
        method="dae.collocation",
        scheme="LAGRANGE-RADAU",
        ):
    if scheme == "LAGRANGE-RADAU":
        file_prefix = "radau_"
    elif scheme == "LAGRANGE-LEGENDRE":
        file_prefix = "legendre_"
    elif scheme == "BACKWARD":
        file_prefix = "backward_"
    else:
        raise ValueError()
    m = make_model()
    from pyomo.contrib.incidence_analysis.interface import (
        get_incidence_graph,
        _generate_variables_in_constraints,
        IncidenceGraphInterface,
    )
    discretize_model(m, method=method, scheme=scheme, nfe=5, ncp=1)
    m.disc_pts, m.dics_coll_pts = discretization_points(m)
    print(m.disc_pts)
    print(m.dics_coll_pts)
    for key, val in m.t.get_discretization_info().items():
        print(key, val)
    print(get_non_collocation_finite_element_points(m.t))
    '''constraints = list(m.component_data_objects(Constraint, active=True))
    variables = list(_generate_variables_in_constraints(constraints))
    graph = get_incidence_graph(variables, constraints)
    igraph = IncidenceGraphInterface(m)
    var_dmp, con_dmp = igraph.dulmage_mendelsohn()
    # TODO:
    # - Dulmage-Mendelsohn
    # - remove square subsystem
    # - separate connected components
    solve_model(m)
    display_values_and_plot(m, file_prefix=file_prefix)'''


if __name__ == "__main__":
    solve_and_plot_results(
        method="dae.collocation",
        scheme="LAGRANGE-LEGENDRE",
    )
