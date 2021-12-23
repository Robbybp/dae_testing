from pyomo.util.subsystems import TemporarySubsystemManager

from HW8_3a import (
    make_model,
    discretize_model,
    solve_model,
    display_values_and_plot,
)
"""
Here we solve the optimization problem, removing the constraints that
we happen to know are "problematic"
"""

def main():
    m = make_model()
    scheme = "LAGRANGE-RADAU"
    discretize_model(m, scheme=scheme)
    to_deactivate = [
        m.x1dot[0], m.x2dot[0], m.x3dot[0]
    ]
    # Here: some function to identify non-continuity constraints at
    # non-collocation finite element points
    with TemporarySubsystemManager(
            to_deactivate=to_deactivate,
            ):
        # We now have 50 dof rather than 51. This makes sense.
        solve_model(m)
    file_prefix = "radau_manual_"
    display_values_and_plot(m, file_prefix=file_prefix)

# Some things to address:
# - How do we properly plot inputs when some inputs aren't defined for
#   all points in time.
# - Do we need to separate "differential variables" from other variables?

if __name__ == "__main__":
    main()
