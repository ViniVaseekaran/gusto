from firedrake import split, LinearVariationalProblem, \
    LinearVariationalSolver, TestFunctions, TrialFunctions, \
    TestFunction, TrialFunction, lhs, rhs, DirichletBC, FacetNormal, \
    div, dx, jump, avg, dS_v, dS_h, ds_v, ds_t, ds_b, inner, dot, grad, \
    MixedFunctionSpace, FunctionSpace, Function, \
    MixedVectorSpaceBasis, VectorSpaceBasis, BrokenElement, \
    Projector, assemble, LinearSolver, Tensor, AssembledVector
from firedrake.solving_utils import flatten_parameters

from gusto.configuration import DEBUG
from gusto import thermodynamics
from abc import ABCMeta, abstractmethod, abstractproperty


__all__ = ["CompressibleSolver", "IncompressibleSolver", "ShallowWaterSolver",
           "HybridisedCompressibleSolver"]


class TimesteppingSolver(object, metaclass=ABCMeta):
    """
    Base class for timestepping linear solvers for Gusto.

    This is a dummy base class.

    :arg state: :class:`.State` object.
    :arg solver_parameters (optional): solver parameters
    :arg overwrite_solver_parameters: boolean, if True use only the
    solver_parameters that have been passed in, if False then update
    the default solver parameters with the solver_parameters passed in.
    """

    def __init__(self, state, solver_parameters=None,
                 overwrite_solver_parameters=False):

        self.state = state

        if solver_parameters is not None:
            if not overwrite_solver_parameters:
                p = flatten_parameters(self.solver_parameters)
                p.update(flatten_parameters(solver_parameters))
                solver_parameters = p
            self.solver_parameters = solver_parameters

        if state.output.log_level == DEBUG:
            self.solver_parameters["ksp_monitor_true_residual"] = True

        # setup the solver
        self._setup_solver()

    @abstractproperty
    def solver_parameters(self):
        """Solver parameters for this solver"""
        pass

    @abstractmethod
    def solve(self):
        pass


class CompressibleSolver(TimesteppingSolver):
    """
    Timestepping linear solver object for the compressible equations
    in theta-pi formulation with prognostic variables u,rho,theta.

    This solver follows the following strategy:
    (1) Analytically eliminate theta (introduces error near topography)
    (2) Solve resulting system for (u,rho) using a Schur preconditioner
    (3) Reconstruct theta

    :arg state: a :class:`.State` object containing everything else.
    :arg quadrature degree: tuple (q_h, q_v) where q_h is the required
    quadrature degree in the horizontal direction and q_v is that in
    the vertical direction
    :arg solver_parameters (optional): solver parameters
    :arg overwrite_solver_parameters: boolean, if True use only the
    solver_parameters that have been passed in, if False then update
    the default solver parameters with the solver_parameters passed in.
    :arg moisture (optional): list of names of moisture fields.
    """

    solver_parameters = {
        'pc_type': 'fieldsplit',
        'pc_fieldsplit_type': 'schur',
        'ksp_type': 'gmres',
        'ksp_max_it': 100,
        'ksp_gmres_restart': 50,
        'pc_fieldsplit_schur_fact_type': 'FULL',
        'pc_fieldsplit_schur_precondition': 'selfp',
        'fieldsplit_0': {'ksp_type': 'preonly',
                         'pc_type': 'bjacobi',
                         'sub_pc_type': 'ilu'},
        'fieldsplit_1': {'ksp_type': 'preonly',
                         'pc_type': 'gamg',
                         'mg_levels': {'ksp_type': 'chebyshev',
                                       'ksp_chebyshev_esteig': True,
                                       'ksp_max_it': 1,
                                       'pc_type': 'bjacobi',
                                       'sub_pc_type': 'ilu'}}
    }

    def __init__(self, state, quadrature_degree=None, solver_parameters=None,
                 overwrite_solver_parameters=False, moisture=None):

        self.moisture = moisture

        if quadrature_degree is not None:
            self.quadrature_degree = quadrature_degree
        else:
            dgspace = state.spaces("DG")
            if any(deg > 2 for deg in dgspace.ufl_element().degree()):
                state.logger.warning("default quadrature degree most likely not sufficient for this degree element")
            self.quadrature_degree = (5, 5)

        super().__init__(state, solver_parameters, overwrite_solver_parameters)

    def _setup_solver(self):
        state = self.state      # just cutting down line length a bit
        dt = state.timestepping.dt
        beta = dt*state.timestepping.alpha
        cp = state.parameters.cp
        mu = state.mu
        Vu = state.spaces("HDiv")
        Vtheta = state.spaces("HDiv_v")
        Vrho = state.spaces("DG")

        # Split up the rhs vector (symbolically)
        u_in, rho_in, theta_in = split(state.xrhs)

        # Build the reduced function space for u,rho
        M = MixedFunctionSpace((Vu, Vrho))
        w, phi = TestFunctions(M)
        u, rho = TrialFunctions(M)

        n = FacetNormal(state.mesh)

        # Get background fields
        thetabar = state.fields("thetabar")
        rhobar = state.fields("rhobar")
        pibar = thermodynamics.pi(state.parameters, rhobar, thetabar)
        pibar_rho = thermodynamics.pi_rho(state.parameters, rhobar, thetabar)
        pibar_theta = thermodynamics.pi_theta(state.parameters, rhobar, thetabar)

        # Analytical (approximate) elimination of theta
        k = state.k             # Upward pointing unit vector
        theta = -dot(k, u)*dot(k, grad(thetabar))*beta + theta_in

        # Only include theta' (rather than pi') in the vertical
        # component of the gradient

        # the pi prime term (here, bars are for mean and no bars are
        # for linear perturbations)

        pi = pibar_theta*theta + pibar_rho*rho

        # vertical projection
        def V(u):
            return k*inner(u, k)

        # specify degree for some terms as estimated degree is too large
        dxp = dx(degree=(self.quadrature_degree))
        dS_vp = dS_v(degree=(self.quadrature_degree))

        # add effect of density of water upon theta
        if self.moisture is not None:
            water_t = Function(Vtheta).assign(0.0)
            for water in self.moisture:
                water_t += self.state.fields(water)
            theta = theta / (1 + water_t)
            thetabar = thetabar / (1 + water_t)

        eqn = (
            inner(w, (state.h_project(u) - u_in))*dx
            - beta*cp*div(theta*V(w))*pibar*dxp
            # following does nothing but is preserved in the comments
            # to remind us why (because V(w) is purely vertical.
            # + beta*cp*jump(theta*V(w),n)*avg(pibar)*dS_v
            - beta*cp*div(thetabar*w)*pi*dxp
            + beta*cp*jump(thetabar*w, n)*avg(pi)*dS_vp
            + (phi*(rho - rho_in) - beta*inner(grad(phi), u)*rhobar)*dx
            + beta*jump(phi*u, n)*avg(rhobar)*(dS_v + dS_h)
        )

        if mu is not None:
            eqn += dt*mu*inner(w, k)*inner(u, k)*dx
        aeqn = lhs(eqn)
        Leqn = rhs(eqn)

        # Place to put result of u rho solver
        self.urho = Function(M)

        # Boundary conditions (assumes extruded mesh)
        bcs = [DirichletBC(M.sub(0), 0.0, "bottom"),
               DirichletBC(M.sub(0), 0.0, "top")]

        # Solver for u, rho
        urho_problem = LinearVariationalProblem(
            aeqn, Leqn, self.urho, bcs=bcs)

        self.urho_solver = LinearVariationalSolver(urho_problem,
                                                   solver_parameters=self.solver_parameters,
                                                   options_prefix='ImplicitSolver')

        # Reconstruction of theta
        theta = TrialFunction(Vtheta)
        gamma = TestFunction(Vtheta)

        u, rho = self.urho.split()
        self.theta = Function(Vtheta)

        theta_eqn = gamma*(theta - theta_in +
                           dot(k, u)*dot(k, grad(thetabar))*beta)*dx

        theta_problem = LinearVariationalProblem(lhs(theta_eqn),
                                                 rhs(theta_eqn),
                                                 self.theta)
        self.theta_solver = LinearVariationalSolver(theta_problem,
                                                    options_prefix='thetabacksubstitution')

    def solve(self):
        """
        Apply the solver with rhs state.xrhs and result state.dy.
        """

        self.urho_solver.solve()

        u1, rho1 = self.urho.split()
        u, rho, theta = self.state.dy.split()
        u.assign(u1)
        rho.assign(rho1)

        self.theta_solver.solve()
        theta.assign(self.theta)


class HybridisedCompressibleSolver(TimesteppingSolver):
    """
    Timestepping linear solver object for the compressible equations
    in theta-pi formulation with prognostic variables u,rho,theta.

    This solver follows the following strategy:
    (1) Analytically eliminate theta (introduces error near topography)
    (2a) Solve resulting system for (u[broken],rho,lambda) using hybridised
    solver
    (2b) reconstruct unbroken u
    (3) Reconstruct theta

    :arg state: a :class:`.State` object containing everything else.
    :arg quadrature degree: tuple (q_h, q_v) where q_h is the required
    quadrature degree in the horizontal direction and q_v is that in
    the vertical direction
    :arg params (optional): solver parameters
    """
    solver_parameters = {'ksp_type': 'gmres',
                         'pc_type': 'lu',
                         'ksp_monitor_true_residual': True}

    def __init__(self, state, quadrature_degree=None, solver_parameters=None,
                 overwrite_solver_parameters=False, moisture=None):
        self.moisture = moisture

        self.state = state

        if quadrature_degree is not None:
            self.quadrature_degree = quadrature_degree
        else:
            dgspace = state.spaces("DG")
            if any(deg > 2 for deg in dgspace.ufl_element().degree()):
                state.logger.warning("default quadrature degree most likely not sufficient for this degree element")
            self.quadrature_degree = (5, 5)

        super().__init__(state, solver_parameters, overwrite_solver_parameters)

    def _setup_solver(self):
        from firedrake.assemble import create_assembly_callable

        state = self.state      # just cutting down line length a bit
        dt = state.timestepping.dt
        beta = dt*state.timestepping.alpha
        cp = state.parameters.cp
        mu = state.mu
        Vu = state.spaces("HDiv")
        Vu_broken = FunctionSpace(state.mesh, BrokenElement(Vu.ufl_element()))
        Vtheta = state.spaces("HDiv_v")
        Vrho = state.spaces("DG")

        h_deg = state.horizontal_degree
        v_deg = state.vertical_degree
        Vtrace = FunctionSpace(state.mesh, "HDiv Trace", degree=(h_deg, v_deg))

        # Split up the rhs vector (symbolically)
        u_in, rho_in, theta_in = split(state.xrhs)

        # Build the reduced function space for u,rho
        M = MixedFunctionSpace((Vu_broken, Vrho))
        w, phi = TestFunctions(M)
        u, rho = TrialFunctions(M)

        l0 = TrialFunction(Vtrace)
        dl = TestFunction(Vtrace)

        n = FacetNormal(state.mesh)

        # Get background fields
        thetabar = state.fields("thetabar")
        rhobar = state.fields("rhobar")
        pibar = exner(thetabar, rhobar, state)
        pibar_rho = exner_rho(thetabar, rhobar, state)
        pibar_theta = exner_theta(thetabar, rhobar, state)

        # Analytical (approximate) elimination of theta
        k = state.k             # Upward pointing unit vector
        theta = -dot(k, u)*dot(k, grad(thetabar))*beta + theta_in

        # Only include theta' (rather than pi') in the vertical
        # component of the gradient

        # the pi prime term (here, bars are for mean and no bars are
        # for linear perturbations)

        pi = pibar_theta*theta + pibar_rho*rho

        # vertical projection
        def V(u):
            return k*inner(u, k)

        # specify degree for some terms as estimated degree is too large
        dxp = dx(degree=(self.quadrature_degree))
        dS_vp = dS_v(degree=(self.quadrature_degree))
        dS_hp = dS_h(degree=(self.quadrature_degree))
        ds_vp = ds_v(degree=(self.quadrature_degree))
        ds_tbp = ds_t(degree=(self.quadrature_degree)) + ds_b(degree=(self.quadrature_degree))

        rhobar_tr = Function(Vtrace)
        rbareqn = (l0('+') - avg(rhobar))*dl('+')*(dS_vp + dS_hp) + \
                  (l0 - rhobar)*dl*ds_vp + \
                  (l0 - rhobar)*dl*ds_tbp
        rhobar_prob = LinearVariationalProblem(lhs(rbareqn), rhs(rbareqn), rhobar_tr)
        self.rhobar_solver = LinearVariationalSolver(rhobar_prob,
                                                     solver_parameters={'ksp_type': 'preonly',
                                                                        'pc_type': 'bjacobi',
                                                                        'pc_sub_type': 'lu',
                                                                        'ksp_monitor_true_residual': True},
                                                     options_prefix='rhobar_solver')

        Aeqn = (inner(w, (u - u_in))*dx
                - beta*cp*div(theta*V(w))*pibar*dxp
                - beta*cp*div(thetabar*w)*pi*dxp
                + (phi*(rho - rho_in) - beta*inner(grad(phi), u)*rhobar)*dx
                + beta*jump(phi*u, n=n)*rhobar_tr('+')*(dS_v + dS_h))

        if mu is not None:
            Aeqn += dt*mu*inner(w, k)*inner(u, k)*dx

        # (A K)(U) = (U_r)
        # (L 0)(l)   (0  )
        Aop = Tensor(lhs(Aeqn))
        Arhs = rhs(Aeqn)

        dl = dl('+')
        l0 = l0('+')
        K = Tensor(beta*cp*jump(thetabar*w, n=n)*l0*(dS_vp + dS_hp)
                   + beta*cp*inner(thetabar*w, n)*l0*ds_vp
                   + beta*cp*inner(thetabar*w, n)*l0*ds_tbp)
        L = Tensor(dl*jump(u, n=n)*(dS_vp + dS_hp)
                   + dl*inner(u, n)*ds_vp
                   + dl*inner(u, n)*ds_tbp)

        # U = A^{-1}(-Kl + U_r), 0=LU=-(LA^{-1}K)l + LA^{-1}U_r, so (LA^{-1}K)l = LA^{-1}U_r
        # reduced eqns for l0

        # Right-hand side
        Rexp = L * Aop.inv * Tensor(Arhs)
        self.R = Function(Vtrace)
        self._assemble_Rexp = create_assembly_callable(Rexp, tensor=self.R)

        # Schur complement operator obtained from element-wise static condensation
        S = assemble(L * Aop.inv * K)
        # Set up the LinearSolver for the system of Lagrange multipliers
        self.lSolver = LinearSolver(S, solver_parameters=self.solver_parameters,
                                    options_prefix='lambda_solve')
        # a place to keep the multiplier solution
        self.lambdar = Function(Vtrace)

        # Place to put result of u rho reconstruction
        self.urho = Function(M)

        # Reconstruction of broken u and rho
        u_, rho = self.urho.split()

        # Split operators for two-stage reconstruction
        A00 = Aop.block((0, 0))
        A01 = Aop.block((0, 1))
        A10 = Aop.block((1, 0))
        A11 = Aop.block((1, 1))
        K0 = K.block((0, 0))
        Rurho = Tensor(Arhs)
        Ru = Rurho.block((0,))
        Rrho = Rurho.block((1,))
        lambda_vec = AssembledVector(self.lambdar)

        # rho reconstruction
        Srho = A11 - A10 * A00.inv * A01
        rho_expr = Srho.inv * (Rrho - A10 * A00.inv * (Ru - K0 * lambda_vec))
        self._assemble_rho = create_assembly_callable(rho_expr, tensor=rho)

        # u reconstruction
        rho_vec = AssembledVector(rho)
        u_expr = A00.inv * (Ru - A01 * rho_vec - K0 * lambda_vec)
        self._assemble_u = create_assembly_callable(u_expr, tensor=u_)

        # Project broken u into the HDiv space
        # NOTE: We can use the averaging trick here
        self.u_hdiv = Function(Vu)
        self.u_projector = Projector(u_, self.u_hdiv,
                                     solver_parameters={'ksp_type': 'gmres',
                                                        'pc_type': 'lu',
                                                        'ksp_monitor_true_residual': True})

        # Reconstruction of theta
        theta = TrialFunction(Vtheta)
        gamma = TestFunction(Vtheta)

        self.theta = Function(Vtheta)

        u = self.u_hdiv
        theta_eqn = gamma*(theta - theta_in +
                           dot(k, u)*dot(k, grad(thetabar))*beta)*dx

        theta_problem = LinearVariationalProblem(lhs(theta_eqn),
                                                 rhs(theta_eqn),
                                                 self.theta)
        self.theta_solver = LinearVariationalSolver(theta_problem,
                                                    solver_parameters={'ksp_type': 'gmres',
                                                                       'pc_type': 'bjacobi',
                                                                       'pc_sub_type': 'ilu',
                                                                       'ksp_monitor_true_residual': True},
                                                    options_prefix='thetabacksubstitution')

    def solve(self):
        """
        Apply the solver with rhs state.xrhs and result state.dy.
        """
        # Project rhobar average into the trace space
        self.rhobar_solver.solve()

        # Assemble the RHS for lambda into self.R
        self._assemble_Rexp()

        # Solve for lambda
        self.lSolver.solve(self.lambdar, self.R)

        # Reconstruct u and rho
        self._assemble_rho()
        self._assemble_u()

        # Project uhat as self.u_hdiv in H(div)
        self.u_projector.project()

        # copy back into u and rho cpts of dy
        _, rho1 = self.urho.split()
        u, rho, theta = self.state.dy.split()
        u.assign(self.u_hdiv)
        rho.assign(rho1)

        # reconstruct theta
        self.theta_solver.solve()

        # copy into theta cpt of dy
        theta.assign(self.theta)


class IncompressibleSolver(TimesteppingSolver):
    """Timestepping linear solver object for the incompressible
    Boussinesq equations with prognostic variables u, p, b.

    This solver follows the following strategy:
    (1) Analytically eliminate b (introduces error near topography)
    (2) Solve resulting system for (u,p) using a block Hdiv preconditioner
    (3) Reconstruct b

    This currently requires a (parallel) direct solver so is probably
    a bit memory-hungry, we'll improve this with a hybridised solver
    soon.

    :arg state: a :class:`.State` object containing everything else.
    :arg L: the width of the domain, used in the preconditioner.
    :arg solver_parameters: (optional) Solver parameters.
    :arg overwrite_solver_parameters: boolean, if True use only the
    solver_parameters that have been passed in, if False then update
    the default solver parameters with the solver_parameters passed in.
    """

    solver_parameters = {
        'ksp_type': 'gmres',
        'pc_type': 'fieldsplit',
        'pc_fieldsplit_type': 'additive',
        'fieldsplit_0': {'ksp_type': 'preonly',
                         'pc_type': 'lu',
                         'pc_factor_mat_solver_type': 'mumps'},
        'fieldsplit_1': {'ksp_type': 'preonly',
                         'pc_type': 'lu',
                         'pc_factor_mat_solver_type': 'mumps'}
    }

    def __init__(self, state, L, solver_parameters=None,
                 overwrite_solver_parameters=False):

        self.L = L
        super().__init__(state, solver_parameters, overwrite_solver_parameters)

    def _setup_solver(self):
        state = self.state      # just cutting down line length a bit
        dt = state.timestepping.dt
        beta = dt*state.timestepping.alpha
        mu = state.mu
        Vu = state.spaces("HDiv")
        Vb = state.spaces("HDiv_v")
        Vp = state.spaces("DG")

        # Split up the rhs vector (symbolically)
        u_in, p_in, b_in = split(state.xrhs)

        # Build the reduced function space for u,p
        M = MixedFunctionSpace((Vu, Vp))
        w, phi = TestFunctions(M)
        u, p = TrialFunctions(M)

        # Get background fields
        bbar = state.fields("bbar")

        # Analytical (approximate) elimination of theta
        k = state.k             # Upward pointing unit vector
        b = -dot(k, u)*dot(k, grad(bbar))*beta + b_in

        # vertical projection
        def V(u):
            return k*inner(u, k)

        eqn = (
            inner(w, (u - u_in))*dx
            - beta*div(w)*p*dx
            - beta*inner(w, k)*b*dx
            + phi*div(u)*dx
        )

        if mu is not None:
            eqn += dt*mu*inner(w, k)*inner(u, k)*dx
        aeqn = lhs(eqn)
        Leqn = rhs(eqn)

        # Place to put result of u p solver
        self.up = Function(M)

        # Boundary conditions (assumes extruded mesh)
        bcs = [DirichletBC(M.sub(0), 0.0, "bottom"),
               DirichletBC(M.sub(0), 0.0, "top")]

        # preconditioner equation
        L = self.L
        Ap = (
            inner(w, u) + L*L*div(w)*div(u) +
            phi*p/L/L
        )*dx

        # Solver for u, p
        up_problem = LinearVariationalProblem(
            aeqn, Leqn, self.up, bcs=bcs, aP=Ap)

        nullspace = MixedVectorSpaceBasis(M,
                                          [M.sub(0),
                                           VectorSpaceBasis(constant=True)])

        self.up_solver = LinearVariationalSolver(up_problem,
                                                 solver_parameters=self.solver_parameters,
                                                 nullspace=nullspace)

        # Reconstruction of b
        b = TrialFunction(Vb)
        gamma = TestFunction(Vb)

        u, p = self.up.split()
        self.b = Function(Vb)

        b_eqn = gamma*(b - b_in +
                       dot(k, u)*dot(k, grad(bbar))*beta)*dx

        b_problem = LinearVariationalProblem(lhs(b_eqn),
                                             rhs(b_eqn),
                                             self.b)
        self.b_solver = LinearVariationalSolver(b_problem)

    def solve(self):
        """
        Apply the solver with rhs state.xrhs and result state.dy.
        """

        self.up_solver.solve()

        u1, p1 = self.up.split()
        u, p, b = self.state.dy.split()
        u.assign(u1)
        p.assign(p1)

        self.b_solver.solve()
        b.assign(self.b)


class ShallowWaterSolver(TimesteppingSolver):

    solver_parameters = {
        'ksp_type': 'preonly',
        'mat_type': 'matfree',
        'pc_type': 'python',
        'pc_python_type': 'firedrake.HybridizationPC',
        'hybridization': {'ksp_type': 'cg',
                          'pc_type': 'gamg',
                          'ksp_rtol': 1e-8,
                          'mg_levels': {'ksp_type': 'chebyshev',
                                        'ksp_max_it': 2,
                                        'pc_type': 'bjacobi',
                                        'sub_pc_type': 'ilu'}}
    }

    def _setup_solver(self):
        state = self.state
        H = state.parameters.H
        g = state.parameters.g
        beta = state.timestepping.dt*state.timestepping.alpha

        # Split up the rhs vector (symbolically)
        u_in, D_in = split(state.xrhs)

        W = state.W
        w, phi = TestFunctions(W)
        u, D = TrialFunctions(W)

        eqn = (
            inner(w, u) - beta*g*div(w)*D
            - inner(w, u_in)
            + phi*D + beta*H*phi*div(u)
            - phi*D_in
        )*dx

        aeqn = lhs(eqn)
        Leqn = rhs(eqn)

        # Place to put result of u rho solver
        self.uD = Function(W)

        # Solver for u, D
        uD_problem = LinearVariationalProblem(
            aeqn, Leqn, self.state.dy)

        self.uD_solver = LinearVariationalSolver(uD_problem,
                                                 solver_parameters=self.solver_parameters,
                                                 options_prefix='SWimplicit')

    def solve(self):
        """
        Apply the solver with rhs state.xrhs and result state.dy.
        """

        self.uD_solver.solve()
