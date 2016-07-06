from __future__ import absolute_import
from abc import ABCMeta, abstractmethod
from firedrake import Function


class BaseTimestepper(object):
    """
    Base timestepping class for Gusto

    """
    __metaclass__ = ABCMeta

    def __init__(self, state, advection_list):

        self.state = state
        self.advection_list = advection_list

    def _set_ubar(self):
        """
        Update ubar in the advection methods.
        """

        state = self.state
        un = state.xn.split()[0]
        unp1 = state.xnp1.split()[0]

        for advection, index in self.advection_list:
            advection.ubar.assign(un + state.timestepping.alpha*(unp1-un))

    @abstractmethod
    def run(self):
        pass


class Timestepper(BaseTimestepper):
    """
    Build a timestepper to implement an "auxiliary semi-Lagrangian" timestepping
    scheme for the dynamical core.

    :arg state: a :class:`.State` object
    :arg advection_list a list of tuples (scheme, i), where i is an
        :class:`.AdvectionScheme` object, and i is the index indicating
        which component of the mixed function space to advect.
    :arg linear_solver: a :class:`.TimesteppingSolver` object
    :arg forcing: a :class:`.Forcing` object
    """
    def __init__(self, state, advection_list, linear_solver, forcing, diffusion_dict=None):

        super(Timestepper, self).__init__(state, advection_list)
        self.linear_solver = linear_solver
        self.forcing = forcing
        self.diffusion_dict = {}
        if diffusion_dict is not None:
            self.diffusion_dict.update(diffusion_dict)

    def run(self, t, tmax):
        state = self.state

        state.xn.assign(state.x_init)

        xstar_fields = state.xstar.split()
        xp_fields = state.xp.split()

        dt = state.timestepping.dt
        alpha = state.timestepping.alpha
        state.dump()

        while t < tmax + 0.5*dt:
            if state.output.Verbose:
                print "STEP", t, dt

            t += dt
            self.forcing.apply((1-alpha)*dt, state.xn, state.xn, state.xstar)
            state.xnp1.assign(state.xn)

            for k in range(state.timestepping.maxk):
                self._set_ubar()  # computes state.ubar from state.xn and state.xnp1
                for advection, index in self.advection_list:
                    # advects a field from xstar and puts result in xp
                    advection.apply(xstar_fields[index], xp_fields[index])
                state.xrhs.assign(0.)  # xrhs is the residual which goes in the linear solve
                for i in range(state.timestepping.maxi):
                    self.forcing.apply(alpha*dt, state.xp, state.xnp1,
                                       state.xrhs)
                    state.xrhs -= state.xnp1
                    self.linear_solver.solve()  # solves linear system and places result in state.dy
                    state.xnp1 += state.dy

            state.xn.assign(state.xnp1)

            for name, diffusion in self.diffusion_dict.iteritems():
                diffusion.apply(state.field_dict[name], state.field_dict[name])

            state.dump()

        state.diagnostic_dump()


class AdvectionTimestepper(BaseTimestepper):

    def run(self, t, tmax, x_end=False):
        state = self.state

        state.xn.assign(state.x_init)

        xn_fields = state.xn.split()
        xnp1_fields = state.xnp1.split()

        dt = state.timestepping.dt
        state.xnp1.assign(state.xn)
        state.dump()

        while t < tmax + 0.5*dt:
            if state.output.Verbose:
                print "STEP", t, dt

            t += dt

            self._set_ubar()  # computes state.ubar from state.xn and state.xnp1
            for advection, index in self.advection_list:
                # advects a field from xn and puts result in xnp1
                advection.apply(xn_fields[index], xnp1_fields[index])

            state.xn.assign(state.xnp1)

            state.dump()

        state.diagnostic_dump()

        if x_end:
            return state.xn


class MovingMeshAdvectionTimestepper(BaseTimestepper):

    def __init__(self, state, advection_list, mesh_velocity, mesh_velocity_expr):

        self.state = state
        self.advection_list = advection_list
        self.mesh_velocity = mesh_velocity
        self.mesh_velocity_expr = mesh_velocity_expr

    def _set_ubar(self):
        """
        Update ubar in the advection methods.
        """

        state = self.state
        un = state.xn.split()[0]
        unp1 = state.xnp1.split()[0]
        v = self.mesh_velocity

        for advection, index in self.advection_list:
            advection.ubar.assign(un + state.timestepping.alpha*(unp1-un) - v)

    def _project_ubar(self):
        """
        Update ubar in the advection methods after mesh has moved.
        """

        state = self.state
        un = state.xn.split()[0]
        unp1 = state.xnp1.split()[0]
        v = self.mesh_velocity

        for advection, index in self.advection_list:
            advection.ubar.project(un + state.timestepping.alpha*(unp1-un) - v)

    def run(self, t, tmax, x_end=False):
        state = self.state
        mesh_velocity_expr = self.mesh_velocity_expr
        deltax = Function(state.mesh.coordinates.function_space())

        state.xn.assign(state.x_init)

        xn_fields = state.xn.split()
        xstar_fields = state.xstar.split()
        xnp1_fields = state.xnp1.split()

        dt = state.timestepping.dt
        state.xnp1.assign(state.xn)
        state.dump()

        while t < tmax + 0.5*dt:
            if state.output.Verbose:
                print "STEP", t, dt

            t += dt

            self._set_ubar()  # computes state.ubar from state.xn and state.xnp1
            for advection, index in self.advection_list:
                # advects a field from xn and puts result in xstar
                advection.apply(xn_fields[index], xstar_fields[index])

            # Move mesh
            x = state.mesh.coordinates
            mesh_velocity_expr.t = t
            self.mesh_velocity.project(mesh_velocity_expr)
            deltax.project(dt*self.mesh_velocity)
            x += deltax
            self.mesh_velocity.project(mesh_velocity_expr)

            # Second advection step on new mesh
            self._project_ubar()
            for advection, index in self.advection_list:
                # advects a field from xstar and puts result in xnp1
                advection.apply(xstar_fields[index], xnp1_fields[index])

            state.xn.assign(state.xnp1)

            state.dump()

        state.diagnostic_dump()

        if x_end:
            return state.xn
