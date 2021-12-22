This directory holds code for testing Pyomo DAE's implementation of
Radau and Legendre collocation. Our immediate goals are to:

(a) Test whether certain equations and variables are "auxilliary,"
i.e. not necessary to solve or optimize the rest of the model
(we suspect that algebraic equations at finite element boundaries,
including `t0`, are auxilliary)
(b) Write functions to remove auxilliary variables and equations
from a model.

A stretch goal is to identify a situation in which the assumptions
made by Pyomo DAE, e.g. that the last collocation point of the previous
element is the same as the current element's finite element point,
actually cause an error or make it demonstrably more difficult to do
some sort of control problem

The concrete goal of this work is some small contributions to Pyomo
DAE to address the problems that we suspect are occuring.
