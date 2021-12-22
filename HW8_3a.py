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
        nfe=10,
        ncp=5,
        ):
    discretizer = TransformationFactory(method)
    discretizer.apply_to(m, nfe=nfe, ncp=ncp, scheme=scheme)


def solve_model(m, tee=True):
    ipopt = SolverFactory("ipopt")
    ipopt.solve(m, tee=tee)


def display_values_and_plot(m, file_prefix=None):
    print("final time = %6.2f" %(value(m.x3[m.tf])))
    x1_list = []
    x3_list = []
    x2_list = []
    u_list = []
    for t in m.t: 
        x1 = value(m.x1[t])
        x1_list.append(x1)

        x2 = value(m.x2[t])
        x2_list.append(x2)

        x3 = value(m.x3[t])
        x3_list.append(x3)

        u = value(m.u[t])
        u_list.append(u)

    print(value(m.u[0]))

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
    plt.plot(m.t,u_list,'-*')
    plt.xlabel('t')
    plt.ylabel('u (t)')
    plt.title('Control Profile')
    control_fname = "control_profile.png"
    if file_prefix is not None:
        control_fname = file_prefix + control_fname
    plt.savefig(control_fname)


def solve_and_plot_results(
        method="dae.collocation",
        scheme="LAGRANGE-RADAU",
        ):
    if scheme == "LAGRANGE-RADAU":
        file_prefix = "radau_"
    elif scheme == "LAGRANGE-LEGENDRE":
        file_prefix = "legendre_"
    else:
        raise ValueError()
    m = make_model()
    from pyomo.contrib.incidence_analysis.interface import (
        get_incidence_graph,
        _generate_variables_in_constraints,
        IncidenceGraphInterface,
    )
    discretize_model(m, scheme=scheme)
    constraints = list(m.component_data_objects(Constraint, active=True))
    variables = list(_generate_variables_in_constraints(constraints))
    graph = get_incidence_graph(variables, constraints)
    igraph = IncidenceGraphInterface(m)
    var_dmp, con_dmp = igraph.dulmage_mendelsohn()
    # TODO:
    # - Dulmage-Mendelsohn
    # - remove square subsystem
    # - separate connected components
    import pdb; pdb.set_trace()
    solve_model(m)
    display_values_and_plot(m, file_prefix=file_prefix)


if __name__ == "__main__":
    solve_and_plot_results(scheme="LAGRANGE-LEGENDRE")
