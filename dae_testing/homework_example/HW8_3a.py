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

    def _init(m):
        yield m.x1[0] == 0
        yield m.x2[0] == -1
        yield m.x3[0] == 0
    m.init_conditions = ConstraintList(rule=_init)
    return m

def Constraint_definitions(m,t):
    def _x1dot(m, t):
        if t in m.non_coll_disc_pts: 
            return Constraint.Skip
        else:
            return m.dx1dt[t] == m.x2[t]
    m.x1dot = Constraint(m.t, rule=_x1dot)

    def _x2dot(m, t):
        if t in m.non_coll_disc_pts: 
            return Constraint.Skip
        else:
            return m.dx2dt[t] == -m.x2[t] + m.u[t]
    m.x2dot = Constraint(m.t, rule=_x2dot)

    def _x3dot(m, t):
        if t in m.non_coll_disc_pts: 
            return Constraint.Skip
        else:
            return m.dx3dt[t] == m.x1[t]**2 + m.x2[t]**2 + 0.005*m.u[t]**2
    m.x3dot = Constraint(m.t, rule=_x3dot)

#Constrain_definitions ignores these constraints at non-collocation, 
#discretization points

def Constraint_definitions_old(m,t):
    def _x1dot(m, t):
        return m.dx1dt[t] == m.x2[t]
    m.x1dot = Constraint(m.t, rule=_x1dot)

    def _x2dot(m, t):
        return m.dx2dt[t] == -m.x2[t] + m.u[t]
    m.x2dot = Constraint(m.t, rule=_x2dot)

    def _x3dot(m, t):
        return m.dx3dt[t] == m.x1[t]**2 + m.x2[t]**2 + 0.005*m.u[t]**2
    m.x3dot = Constraint(m.t, rule=_x3dot)

#Constraint_definitions_old doesn't ignore constraints at 
#non-collocation discretization points


def discretize_model(
        m,
        method="dae.collocation",
        scheme="LAGRANGE-RADAU",
        nfe= 2,
        ncp=3,
        ):
    discretizer = TransformationFactory(method)
    discretizer.apply_to(m, nfe=nfe, ncp=ncp, scheme=scheme)
    

def discretization_points(m):
    return m.t.get_finite_elements(), \
        m.t.get_discretization_info()['tau_points']

#m.t.get_finite_elements = These are all discretization points 
#m.t.tau_pts = These are roots of polynomials including 0 
#as an additional element


def convert_taupts_to_cp(m, nfe):
    h = m.tf/nfe
    cp = []
    for i in m.disc_pts[:-1] :
        for j in m.col_roots[1:]:
            cp.append(i + h*j)
    return cp

#These convert the polynomial roots to collocation points between 
#each of the finite element boundaries.

def non_coll_dicretization_pts(m):
    round_disc_pts = []
    round_cp = []
    for i in m.disc_pts: 
        round_disc_pts.append(round(i, 4))
    for i in m.cp:
        round_cp.append(round(i,4))

    non_coll_disc_pts = list(set(round_disc_pts) - set(round_cp))
    return non_coll_disc_pts

#This function returns the dicretization points which are not collocation points
#If we don't round the items in the list we have problems with list comparison. 
#0.3333 is not same as 0.3333333


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


def solve_and_plot_results(
        method="dae.collocation",
        scheme="LAGRANGE-RADAU",
        nfe= 2,
        ncp=3
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
    discretize_model(m, scheme=scheme, nfe = nfe, ncp = ncp)
    m.disc_pts, m.col_roots = discretization_points(m)

    m.cp = convert_taupts_to_cp(m,nfe = nfe)

    m.non_coll_disc_pts = non_coll_dicretization_pts(m)

    print(m.non_coll_disc_pts)
    Constraint_definitions(m, m.t)

    constraints = list(m.component_data_objects(Constraint, active=True))
    variables = list(_generate_variables_in_constraints(constraints))
    graph = get_incidence_graph(variables, constraints)
    igraph = IncidenceGraphInterface(m)
    var_dmp, con_dmp = igraph.dulmage_mendelsohn()
    # TODO:
    # - Dulmage-Mendelsohn
    # - remove square subsystem
    # - separate connected components
    solve_model(m)
    display_values_and_plot(m, file_prefix=file_prefix)


if __name__ == "__main__":
    solve_and_plot_results(scheme="LAGRANGE-LEGENDRE", nfe =10 , ncp = 4)
