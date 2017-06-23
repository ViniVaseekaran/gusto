from __future__ import absolute_import, print_function, division
from firedrake import dx, assemble, LinearSolver
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace
from firedrake.parloops import par_loop, READ, RW, INC
from firedrake.ufl_expr import TrialFunction, TestFunction
from firedrake.slope_limiter.limiter import Limiter
from firedrake.slope_limiter.vertex_based_limiter import VertexBasedLimiter
__all__ = ("RhoLimiter",)

_copy_to_vertex_space_loop = """
theta_hat[0][0] = theta[0][0];
theta_hat[1][0] = theta[1][0];
theta_hat[2][0] = theta[3][0];
theta_hat[3][0] = theta[4][0];
"""
_copy_from_vertex_space_loop = """
theta[0][0] = theta_hat[0][0];
theta[1][0] = theta_hat[1][0];
theta[3][0] = theta_hat[2][0];
theta[4][0] = theta_hat[3][0];
"""

_weight_kernel = """
for (int i=0; i<weight.dofs; ++i) {
    weight[i][0] += 1.0;
    }"""

_average_kernel = """
for (int i=0; i<vrec.dofs; ++i) {
        vrec[i][0] += v_b[i][0]/weight[i][0];
        }"""


class RhoLimiter(Limiter):
    """
    A vertex based limiter for P1DG fields.

    This limiter implements the vertex-based limiting scheme described in
    Dmitri Kuzmin, "A vertex-based hierarchical slope limiter for p-adaptive
    discontinuous Galerkin methods". J. Comp. Appl. Maths (2010)
    http://dx.doi.org/10.1016/j.cam.2009.05.028
    """

    def __init__(self, space):
        """
        Initialise limiter

        :param space : FunctionSpace instance
        """

        self.Vr = space
        self.Q1DG = FunctionSpace(self.Vr.mesh(), 'DG', 1) # space with only vertex DOF
        self.vertex_limiter = VertexBasedLimiter(self.Q1DG)
        self.theta_hat = Function(self.Q1DG) # theta function with only vertex DOF
        self.w = Function(self.Vr)
        self.result = Function(self.Vr)
        par_loop(_weight_kernel, dx, {"weight": (self.w, INC)})
        
    def copy_vertex_values(self, field):
        """
        Copies the vertex values from temperature space to
        Q1DG space which only has vertices.
        """
        par_loop(_copy_to_vertex_space_loop, dx,
                 {"theta": (field, READ),
                  "theta_hat": (self.theta_hat, RW)})

    def copy_vertex_values_back(self, field):
        """
        Copies the vertex values back from the Q1DG space to
        the original temperature space.
        """
        par_loop(_copy_from_vertex_space_loop, dx,
                 {"theta": (field, RW),
                  "theta_hat": (self.theta_hat, READ)})

    def remap_to_embedded_space(self, field):
        """
        Not entirely sure yet. Remap to embedded space?
        """

        self.result.assign(0.)
        par_loop(_average_kernel, dx, {"vrec": (self.result, INC),
                                            "v_b": (field, READ),
                                            "weight": (self.w, READ)})
        field.assign(self.result)
        
    def compute_bounds(self, field):
        """
        Blank
        """

    def apply_limiter(self, field):
        """
        Blank
        """

    def apply(self, field):
        """
        Re-computes centroids and applies limiter to given field
        """
        assert field.function_space() == self.Vr, \
            'Given field does not belong to this objects function space'

        #self.copy_vertex_values(field)
        self.vertex_limiter.apply(field)
        #self.copy_vertex_values_back(field)
        #self.remap_to_embedded_space(field)
