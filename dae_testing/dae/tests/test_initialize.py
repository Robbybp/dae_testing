import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
import pyomo.environ as pyo
import pyomo.dae as dae

'''from dae_testing.dae.initialize import (
    get_non_collocation_finite_element_points,
    continuity_constraints,
    get_continuity_constraint_names,
    not_cont_constraints_nc_fep,
)'''
from initialize import  (
    get_non_collocation_finite_element_points,
    get_continuity_constraint_names,
    not_cont_constraints_nc_fep,
)

class TestNonCollocationFiniteElementPoints(unittest.TestCase):
    # Cases to test:
    # - One test for each discretization
    #   (Legendre, Radau, forward, backward, central?)

    def test_radau(self):
        m = pyo.ConcreteModel()
        m.cset = dae.ContinuousSet(initialize=[0, 10])
        disc = pyo.TransformationFactory("dae.collocation")
        disc.apply_to(m, wrt=m.cset, nfe=4, ncp=3, scheme="LAGRANGE-RADAU")
        non_colloc_feps = get_non_collocation_finite_element_points(
            m.cset
        )
        expected_non_colloc_feps = [0.0]
        self.assertEqual(non_colloc_feps, expected_non_colloc_feps)

    def test_radau_1_cp(self):
        m = pyo.ConcreteModel()
        m.cset = dae.ContinuousSet(initialize=[0, 10])
        disc = pyo.TransformationFactory("dae.collocation")
        disc.apply_to(m, wrt=m.cset, nfe=4, ncp=1, scheme="LAGRANGE-RADAU")
        non_colloc_feps = get_non_collocation_finite_element_points(
            m.cset
        )
        expected_non_colloc_feps = [0.0]
        self.assertEqual(non_colloc_feps, expected_non_colloc_feps)

    def test_backward(self):
        m = pyo.ConcreteModel()
        m.cset = dae.ContinuousSet(initialize=[0, 10])
        disc = pyo.TransformationFactory("dae.finite_difference")
        disc.apply_to(m, wrt=m.cset, nfe=4, scheme="BACKWARD")
        non_colloc_feps = get_non_collocation_finite_element_points(
            m.cset
        )
        expected_non_colloc_feps = [0.0]
        self.assertEqual(non_colloc_feps, expected_non_colloc_feps)

    def test_legendre(self):
        m = pyo.ConcreteModel()
        m.cset = dae.ContinuousSet(initialize=[0, 10])
        disc = pyo.TransformationFactory("dae.collocation")
        disc.apply_to(m, wrt=m.cset, nfe=4, ncp=3, scheme="LAGRANGE-LEGENDRE")
        non_colloc_feps = get_non_collocation_finite_element_points(
            m.cset
        )
        expected_non_colloc_feps = [0.0, 2.5, 5.0, 7.5, 10.0]
        self.assertEqual(non_colloc_feps, expected_non_colloc_feps)

    def test_forward(self):
        m = pyo.ConcreteModel()
        m.cset = dae.ContinuousSet(initialize=[0, 10])
        disc = pyo.TransformationFactory("dae.finite_difference")
        disc.apply_to(m, wrt=m.cset, nfe=4, scheme="FORWARD")
        non_colloc_feps = get_non_collocation_finite_element_points(
            m.cset
        )
        expected_non_colloc_feps = [10.0]
        self.assertEqual(non_colloc_feps, expected_non_colloc_feps)

    def test_central(self):
        m = pyo.ConcreteModel()
        m.cset = dae.ContinuousSet(initialize=[0, 10])
        disc = pyo.TransformationFactory("dae.finite_difference")
        disc.apply_to(m, wrt=m.cset, nfe=4, scheme="CENTRAL")

        with self.assertRaisesRegex(ValueError, "Central discretization"):
            non_colloc_feps = get_non_collocation_finite_element_points(
                m.cset
            )
            
class TestContinuityConstraints(unittest.TestCase):
    #Cases to test:
    # - Test for 1 continuous set indexing
    # - Test for more than 1 continuous set indexing
    # - Test for no Derivativevar
    # - Test for second derivatives
    
    def test_one_cont_set_index(self):
        m = pyo.ConcreteModel()
        m.cset = dae.ContinuousSet(initialize=[0, 10])
        m.x = pyo.Var(m.cset)
        m.dx = dae.DerivativeVar(m.x, wrt=m.cset)
        cont_constraints = get_continuity_constraint_names(m, [m.cset])
        expected_cont_constraints = {'x_cset_cont_eq'}
        self.assertEqual(cont_constraints, expected_cont_constraints)
        
    def test_many_cont_set_index(self):
        m = pyo.ConcreteModel()
        m.cset1 = dae.ContinuousSet(initialize=[0, 10])
        m.cset2 = dae.ContinuousSet(initialize=[5, 15])
        m.x = pyo.Var(m.cset1, m.cset2)
        m.dx1 = dae.DerivativeVar(m.x, wrt=m.cset1)
        m.dx2 = dae.DerivativeVar(m.x, wrt=m.cset2)
        cont_constraints = get_continuity_constraint_names(m, [m.cset1])
        expected_cont_constraints = {'x_cset1_cont_eq'}
        self.assertEqual(cont_constraints, expected_cont_constraints)
        
    def test_no_derivative_var(self):
        m = pyo.ConcreteModel()
        m.cset = dae.ContinuousSet(initialize=[0, 10])
        m.x = pyo.Var(m.cset)
        cont_constraints = get_continuity_constraint_names(m, [m.cset])
        expected_cont_constraints = set()
        self.assertEqual(cont_constraints, expected_cont_constraints)
        
    def test_second_derivative(self):
        m = pyo.ConcreteModel()
        m.cset = dae.ContinuousSet(initialize=[0, 10])
        m.x = pyo.Var(m.cset)
        m.dx = dae.DerivativeVar(m.x, wrt=m.cset)
        m.dx2 = dae.DerivativeVar(m.dx, wrt=m.cset)
        cont_constraints = get_continuity_constraint_names(m, [m.cset])
        expected_cont_constraints = {'x_cset_cont_eq', 'dx_cset_cont_eq'}
        self.assertEqual(cont_constraints, expected_cont_constraints)

class TestNotContinuityConsNCFEP(unittest.TestCase):
    #Cases to test:
    # - Test radau and 1 continuous set indexing
    def test_one_cont_set_index_radau(self):
        m = pyo.ConcreteModel()
        m.cset = dae.ContinuousSet(initialize=[0, 10])
        m.x = pyo.Var(m.cset)
        m.dx = dae.DerivativeVar(m.x, wrt=m.cset)
        
        def _xdot(m, t):
            return m.dx[t] == m.x[t]
        m.xdot = pyo.Constraint(m.cset, rule=_xdot)
        
        cont_constraints = get_continuity_constraint_names(m, [m.cset])
        disc = pyo.TransformationFactory("dae.collocation")
        disc.apply_to(m, wrt=m.cset, nfe=4, ncp=3, scheme="LAGRANGE-RADAU")
        non_coll_fe_pts = get_non_collocation_finite_element_points(m.cset)
        d_con = not_cont_constraints_nc_fep(m,[m.cset],non_coll_fe_pts,
                                            cont_constraints)
        
        expected_d_con = ['xdot[0]']
        self.assertEqual([c.name for c in d_con], [elem for elem in 
                                                   expected_d_con])
        
    def test_one_cont_set_index_legendre(self):
        m = pyo.ConcreteModel()
        m.cset = dae.ContinuousSet(initialize=[0, 10])
        m.x = pyo.Var(m.cset)
        m.dx = dae.DerivativeVar(m.x, wrt=m.cset)
        
        def _xdot(m, t):
            return m.dx[t] == m.x[t]
        m.xdot = pyo.Constraint(m.cset, rule=_xdot)
        
        cont_constraints = get_continuity_constraint_names(m, [m.cset])
        disc = pyo.TransformationFactory("dae.collocation")
        disc.apply_to(m, wrt=m.cset, nfe=4, ncp=3, scheme="LAGRANGE-LEGENDRE")
        non_coll_fe_pts = get_non_collocation_finite_element_points(m.cset)
        d_con = not_cont_constraints_nc_fep(m,[m.cset],non_coll_fe_pts,
                                            cont_constraints)
        
        expected_d_con = ['xdot[0]', 'xdot[2.5]','xdot[5.0]','xdot[7.5]',
                          'xdot[10]']
        self.assertEqual([c.name for c in d_con], [elem for elem in 
                                                   expected_d_con])
        
    def test_one_cont_set_index_forward(self):
        m = pyo.ConcreteModel()
        m.cset = dae.ContinuousSet(initialize=[0, 10])
        m.x = pyo.Var(m.cset)
        m.dx = dae.DerivativeVar(m.x, wrt=m.cset)
        
        def _xdot(m, t):
            return m.dx[t] == m.x[t]
        m.xdot = pyo.Constraint(m.cset, rule=_xdot)
        
        cont_constraints = get_continuity_constraint_names(m, [m.cset])
        disc = pyo.TransformationFactory("dae.finite_difference")
        disc.apply_to(m, wrt=m.cset, nfe=4, scheme="FORWARD")
        non_coll_fe_pts = get_non_collocation_finite_element_points(m.cset)
        d_con = not_cont_constraints_nc_fep(m,[m.cset],non_coll_fe_pts,
                                            cont_constraints)
        
        expected_d_con = ['xdot[10]']
        self.assertEqual([c.name for c in d_con], [elem for elem in 
                                                   expected_d_con])
        
        
    def test_one_cont_set_index_backward(self):
        m = pyo.ConcreteModel()
        m.cset = dae.ContinuousSet(initialize=[0, 10])
        m.x = pyo.Var(m.cset)
        m.dx = dae.DerivativeVar(m.x, wrt=m.cset)
        
        def _xdot(m, t):
            return m.dx[t] == m.x[t]
        m.xdot = pyo.Constraint(m.cset, rule=_xdot)
        
        cont_constraints = get_continuity_constraint_names(m, [m.cset])
        disc = pyo.TransformationFactory("dae.finite_difference")
        disc.apply_to(m, wrt=m.cset, nfe=4, scheme="BACKWARD")
        non_coll_fe_pts = get_non_collocation_finite_element_points(m.cset)
        d_con = not_cont_constraints_nc_fep(m,[m.cset],non_coll_fe_pts,
                                            cont_constraints)
        
        expected_d_con = ['xdot[0]']
        self.assertEqual([c.name for c in d_con], [elem for elem in 
                                                   expected_d_con])
        
    #Confused with this test
        
    def _test_two_cont_set_index_radau(self):
        m = pyo.ConcreteModel()
        m.cset1 = dae.ContinuousSet(initialize=[0, 10])
        m.cset2 = dae.ContinuousSet(initialize=[5, 15])
        m.x = pyo.Var(m.cset1, m.cset2)
        m.dx1 = dae.DerivativeVar(m.x, wrt=m.cset1)
        m.dx2 = dae.DerivativeVar(m.x, wrt=m.cset2)
        
        def _xdot1(m, t1, t2):
            return m.dx1[t1, t2] == m.x[t1, t2]
        m.xdot1 = pyo.Constraint(m.cset1, m.cset2, rule=_xdot1)
        
        def _xdot2(m, t1, t2):
            return m.dx2[t1, t2] == m.x[t1, t2]**2
        m.xdot2 = pyo.Constraint(m.cset1, m.cset2, rule=_xdot2)
        
        
        cont_constraints = get_continuity_constraint_names(m, [m.cset])
        disc = pyo.TransformationFactory("dae.collocation")
        disc.apply_to(m, wrt=m.cset1, nfe=4, ncp=3, scheme="LAGRANGE-RADAU")
        non_coll_fe_pts = get_non_collocation_finite_element_points(m.cset1)
        d_con = not_cont_constraints_nc_fep(m,[m.cset1, m.cset2],
                                            non_coll_fe_pts, cont_constraints)
        
        expected_d_con = ['xdot1[0,?]','xdot2[0,?]']
        self.assertEqual([c.name for c in d_con], [elem for elem in 
                                                   expected_d_con])

    def test_two_cont_set_discretized(self):
        m = pyo.ConcreteModel()
        m.cset1 = dae.ContinuousSet(initialize=[0, 10])
        m.cset2 = dae.ContinuousSet(initialize=[5, 15])
        m.x = pyo.Var(m.cset1, m.cset2)
        m.dx1 = dae.DerivativeVar(m.x, wrt=m.cset1)
        m.dx2 = dae.DerivativeVar(m.x, wrt=m.cset2)
        
        def _xdot1(m, t1, t2):
            return m.dx1[t1, t2] == m.x[t1, t2]
        m.xdot1 = pyo.Constraint(m.cset1, m.cset2, rule=_xdot1)
        
        def _xdot2(m, t1, t2):
            return m.dx2[t1, t2] == m.x[t1, t2]**2
        m.xdot2 = pyo.Constraint(m.cset1, m.cset2, rule=_xdot2)
        
        
        cont_constraints = get_continuity_constraint_names(m, [m.cset1])
        disc = pyo.TransformationFactory("dae.collocation")
        disc.apply_to(m, wrt=m.cset1, nfe=4, ncp=3, scheme="LAGRANGE-RADAU")
        disc.apply_to(m, wrt=m.cset2, nfe=4, ncp=3, scheme="LAGRANGE-LEGENDRE")
        non_coll_fe_pts = get_non_collocation_finite_element_points(m.cset1)
        d_con = not_cont_constraints_nc_fep(m,[m.cset1, m.cset2],
                                            non_coll_fe_pts, cont_constraints)

        expected_d_con = []
        expected_d_con.extend(m.dx2_disc_eq[0, :])
        expected_d_con.extend(m.x_cset2_cont_eq[0, :])
        expected_d_con.extend(m.xdot1[0, :])
        expected_d_con.extend(m.xdot2[0, :])

        self.assertEqual(len(d_con), len(expected_d_con))
        expected_d_con_set = ComponentSet(expected_d_con)
        for con in d_con:
            self.assertIn(con, expected_d_con_set)
        
        
if __name__ == "__main__":
    unittest.main()
