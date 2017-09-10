from abc import ABCMeta, abstractmethod
from pyop2.profiling import timed_stage
from gusto.linear_solvers import IncompressibleSolver
from firedrake import DirichletBC


__all__ = ["Timestepper", "AdvectionDiffusion", "ImplicitMidpoint"]


class BaseTimestepper(object, metaclass=ABCMeta):
    """
    Base timestepping class for Gusto

    :arg state: a :class:`.State` object
    :arg advected_fields: iterable of ``(field_name, scheme)`` pairs
        indicating the fields to advect, and the
        :class:`~.Advection` to use.
    """

    def __init__(self, state, advected_fields=None, diffused_fields=None,
                 physics_list=None):

        self.state = state
        if advected_fields is None:
            self.advected_fields = ()
        else:
            self.advected_fields = tuple(advected_fields)
        if diffused_fields is None:
            self.diffused_fields = ()
        else:
            self.diffused_fields = tuple(diffused_fields)
        if physics_list is not None:
            self.physics_list = physics_list
        else:
            self.physics_list = []

    def _apply_bcs(self):
        """
        Set the zero boundary conditions in the velocity.
        """
        unp1 = self.state.xnp1.split()[0]

        if unp1.function_space().extruded:
            M = unp1.function_space()
            bcs = [DirichletBC(M, 0.0, "bottom"),
                   DirichletBC(M, 0.0, "top")]

            for bc in bcs:
                bc.apply(unp1)

    def setup_timeloop(self, t, pickup):
        state = self.state
        state.xb.assign(state.xn)
        state.setup_diagnostics()

        with timed_stage("Dump output"):
            state.setup_dump(pickup)
            t = state.dump(t, pickup)
        return t

    @abstractmethod
    def timestep(self, dt):
        pass

    def run(self, t, tmax, pickup=False):
        self.setup_timeloop(t, pickup)
        state = self.state
        dt = state.timestepping.dt
        while t < tmax - 0.5*dt:
            if state.output.Verbose:
                print("STEP", t, dt)

            t += dt
            state.t.assign(t)
            state.xnp1.assign(state.xn)

            self.timestep(dt)
            state.xb.assign(state.xn)
            state.xn.assign(state.xnp1)

            with timed_stage("Diffusion"):
                for name, diffusion in self.diffused_fields:
                    field = getattr(state.fields, name)
                    diffusion.apply(field, field)

            with timed_stage("Physics"):
                for physics in self.physics_list:
                    physics.apply()

            with timed_stage("Dump output"):
                state.dump(t, pickup=False)

        print("TIMELOOP complete. t= " + str(t) + " tmax=" + str(tmax))


class NonlinearTimestepper(BaseTimestepper):
    def __init__(self, state, advected_fields, linear_solver, forcing,
                 diffused_fields=None, physics_list=None):

        super().__init__(state, advected_fields, diffused_fields, physics_list)
        self.linear_solver = linear_solver
        self.forcing = forcing

        # list of fields that are passively advected (and possibly diffused)
        self.passive_advection = [(name, scheme) for name, scheme in self.advected_fields if name not in state.fieldlist]
        # list of fields that are advected as part of the nonlinear iteration
        self.active_advection = [(name, scheme) for name, scheme in self.advected_fields if name in state.fieldlist]

        if isinstance(self.linear_solver, IncompressibleSolver):
            self.incompressible = True
        else:
            self.incompressible = False


class ImplicitMidpoint(NonlinearTimestepper):
    def setup_timeloop(self, t, pickup):
        t = super().setup_timeloop(t, pickup)
        state = self.state
        self.xn_fields = {name: func for (name, func) in
                          zip(state.fieldlist, state.xn.split())}
        self.xrhs_fields = {name: func for (name, func) in
                            zip(state.fieldlist, state.xrhs.split())}

    def timestep(self, dt):
        state = self.state

        state.xrhs.assign(0.)  # xrhs is the residual which goes in the linear solve
        for k in range(state.timestepping.maxk):

            for name, advection in self.active_advection:
                # first computes ubar from state.xn and state.xnp1
                advection.update_ubar(state.xn, state.xnp1, state.timestepping.alpha)
                # advects a field from xn and puts result in xp
                advection.apply(self.xn_fields[name], self.xrhs_fields[name])

            state.xrhs -= state.xnp1

            with timed_stage("Implicit solve"):
                self.linear_solver.solve()  # solves linear system and places result in state.dy

            state.xnp1 += state.dy


class Timestepper(NonlinearTimestepper):
    """
    Build a timestepper to implement an "auxiliary semi-Lagrangian" timestepping
    scheme for the dynamical core.

    :arg state: a :class:`.State` object
    :arg advected_fields: iterable of ``(field_name, scheme)`` pairs
        indicating the fields to advect, and the
        :class:`~.Advection` to use.
    :arg diffused_fields: optional iterable of ``(field_name, scheme)``
        pairs indictaing the fields to diffusion, and the
        :class:`~.Diffusion` to use.
    :arg linear_solver: a :class:`.TimesteppingSolver` object
    :arg forcing: a :class:`.Forcing` object
    """

    def setup_timeloop(self, t, pickup):
        t = super().setup_timeloop(t, pickup)
        state = self.state
        self.xstar_fields = {name: func for (name, func) in
                             zip(state.fieldlist, state.xstar.split())}
        self.xp_fields = {name: func for (name, func) in
                          zip(state.fieldlist, state.xp.split())}

        return t

    def timestep(self, dt):

        state = self.state
        alpha = state.timestepping.alpha
        if state.mu is not None:
            mu_alpha = [0., dt]
        else:
            mu_alpha = [None, None]

        with timed_stage("Apply forcing terms"):
            self.forcing.apply((1-alpha)*dt, state.xn, state.xn,
                               state.xstar, mu_alpha=mu_alpha[0])

        for k in range(state.timestepping.maxk):

            with timed_stage("Advection"):
                for name, advection in self.active_advection:
                    # first computes ubar from state.xn and state.xnp1
                    advection.update_ubar(state.xn, state.xnp1, state.timestepping.alpha)
                    # advects a field from xstar and puts result in xp
                    advection.apply(self.xstar_fields[name], self.xp_fields[name])

            state.xrhs.assign(0.)  # xrhs is the residual which goes in the linear solve

            for i in range(state.timestepping.maxi):

                with timed_stage("Apply forcing terms"):
                    self.forcing.apply(alpha*dt, state.xp, state.xnp1,
                                       state.xrhs, mu_alpha=mu_alpha[1],
                                       incompressible=self.incompressible)

                state.xrhs -= state.xnp1

                with timed_stage("Implicit solve"):
                    self.linear_solver.solve()  # solves linear system and places result in state.dy

                state.xnp1 += state.dy

        self._apply_bcs()

        for name, advection in self.passive_advection:
            field = getattr(state.fields, name)
            # first computes ubar from state.xn and state.xnp1
            advection.update_ubar(state.xn, state.xnp1, state.timestepping.alpha)
            # advects a field from xn and puts result in xnp1
            advection.apply(field, field)


class AdvectionDiffusion(BaseTimestepper):

    def timestep(self, dt):

        state = self.state
        with timed_stage("Advection"):
            for name, advection in self.advected_fields:
                field = getattr(state.fields, name)
                # first computes ubar from state.xn and state.xnp1
                advection.update_ubar(state.xn, state.xnp1, state.timestepping.alpha)
                # advects field
                advection.apply(field, field)
