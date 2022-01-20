import pyomo.common.unittest as unittest
import pyomo.environ as pyo
import pyomo.dae as dae

from dae_testing.dae.initialize import (
    get_non_collocation_finite_element_points,
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


if __name__ == "__main__":
    unittest.main()
