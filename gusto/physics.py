from abc import ABCMeta, abstractmethod
from gusto.transport_equation import EmbeddedDGAdvection
from gusto.advection import SSPRK3
from firedrake import exp, Interpolator, conditional, Function, \
    min_value, max_value, TestFunction, dx, as_vector, \
    NonlinearVariationalProblem, NonlinearVariationalSolver


__all__ = ["Condensation", "Fallout", "Coalescence"]


class Physics(object, metaclass=ABCMeta):
    """
    Base class for physics processes for Gusto.

    :arg state: :class:`.State` object.
    """

    def __init__(self, state):
        self.state = state

    @abstractmethod
    def apply(self):
        """
        Function computes the value of specific
        fields at the next time step.
        """
        pass


class Condensation(Physics):
    """
    The process of condensation of water vapour
    into liquid water and evaporation of liquid
    water into water vapour, with the associated
    latent heat changes.

    :arg state: :class:`.State.` object.
    :arg weak: Boolean to determine whether weak
               formulation of condensation is used.
    """

    def __init__(self, state, weak=False):
        super(Condensation, self).__init__(state)

        self.weak = weak

        # obtain our fields
        self.theta = state.fields('theta')
        self.water_v = state.fields('water_v')
        self.water_c = state.fields('water_c')
        rho = state.fields('rho')

        # declare function space
        Vt = self.theta.function_space()

        param = self.state.parameters

        # define some parameters as attributes
        dt = self.state.timestepping.dt
        R_d = param.R_d
        p_0 = param.p_0
        kappa = param.kappa
        cp = param.cp
        cv = param.cv
        c_pv = param.c_pv
        c_pl = param.c_pl
        c_vv = param.c_vv
        R_v = param.R_v
        L_v0 = param.L_v0
        T_0 = param.T_0
        w_sat1 = param.w_sat1
        w_sat2 = param.w_sat2
        w_sat3 = param.w_sat3
        w_sat4 = param.w_sat4

        # make useful fields
        Pi = ((R_d * rho * self.theta / p_0)
              ** (kappa / (1.0 - kappa)))
        T = Pi * self.theta * R_d / (R_d + self.water_v * R_v)
        p = p_0 * Pi ** (1.0 / kappa)
        L_v = L_v0 - (c_pl - c_pv) * (T - T_0)
        R_m = R_d + R_v * self.water_v
        c_pml = cp + c_pv * self.water_v + c_pl * self.water_c
        c_vml = cv + c_vv * self.water_v + c_pl * self.water_c

        # use Teten's formula to calculate w_sat
        w_sat = (w_sat1 /
                 (p * exp(w_sat2 * (T - T_0) / (T - w_sat3)) - w_sat4))

        # make appropriate condensation rate
        dot_r_cond = ((self.water_v - w_sat) /
                      (dt * (1.0 + ((L_v ** 2.0 * w_sat) /
                                    (cp * R_v * T ** 2.0)))))

        # introduce a weak condensation which might hold at discrete level
        if self.weak:
            cond_rate_expr = dot_r_cond
            dot_r_cond = Function(Vt)
            phi = TestFunction(Vt)
            quadrature_degree = (4, 4)
            dxp = dx(degree=(quadrature_degree))
            cond_rate_functional = (phi * dot_r_cond * dxp
                                    - phi * cond_rate_expr * dxp)
            cond_rate_problem = NonlinearVariationalProblem(cond_rate_functional, dot_r_cond)
            self.cond_rate_solver = NonlinearVariationalSolver(cond_rate_problem)

        # make cond_rate function, that needs to be the same for all updates in one time step
        self.cond_rate = Function(Vt)

        # adjust cond rate so negative concentrations don't occur
        self.lim_cond_rate = Interpolator(conditional(dot_r_cond < 0,
                                                      max_value(dot_r_cond, - self.water_c / dt),
                                                      min_value(dot_r_cond, self.water_v / dt)), self.cond_rate)

        # tell the prognostic fields what to update to
        self.water_v_new = Interpolator(self.water_v - dt * self.cond_rate, Vt)
        self.water_c_new = Interpolator(self.water_c + dt * self.cond_rate, Vt)
        self.theta_new = Interpolator(self.theta *
                                      (1.0 + dt * self.cond_rate *
                                       (cv * L_v / (c_vml * cp * T) -
                                        R_v * cv * c_pml / (R_m * cp * c_vml))), Vt)

    def apply(self):
        if self.weak:
            self.cond_rate_solver.solve()
        self.lim_cond_rate.interpolate()
        self.theta.assign(self.theta_new.interpolate())
        self.water_v.assign(self.water_v_new.interpolate())
        self.water_c.assign(self.water_c_new.interpolate())


class Fallout(Physics):
    """
    The fallout process of hydrometeors.

    :arg state :class: `.State.` object.
    """

    def __init__(self, state):
        super(Fallout, self).__init__(state)

        self.state = state
        self.rain = state.fields('rain')

        # function spaces
        Vt = self.rain.function_space()
        Vu = state.fields('u').function_space()

        # introduce sedimentation rate
        # for now assume all rain falls at terminal velocity
        terminal_velocity = 10  # in m/s
        self.v = state.fields("rainfall_velocity", Vu)
        self.v.project(as_vector([0, -terminal_velocity]))

        # sedimentation will happen using a full advection method
        advection_equation = EmbeddedDGAdvection(state, Vt, equation_form="advective", outflow=True)
        self.advection_method = SSPRK3(state, self.rain, advection_equation)

    def apply(self):
        for k in range(self.state.timestepping.maxk):
            self.advection_method.update_ubar(self.v, self.v, 0)
            self.advection_method.apply(self.rain, self.rain)


class Coalescence(Physics):
    """
    The process of coalescence of liquid water
    into rain droplets.
    :arg state: :class:`.State.` object.
    """

    def __init__(self, state):
        super(Coalescence, self).__init__(state)

        self.water_c = state.fields('water_c')
        self.rain = state.fields('rain')
        Vt = self.water_c.function_space()

        # constants
        dt = state.timestepping.dt
        k1 = 0.001
        k2 = 2.2
        a = 0.001

        # collection of rainwater
        collection = k2 * self.water_c * self.rain ** 0.875

        # autoconversion of cloud water into rain water
        autoconversion = k1 * (self.water_c - a)


        # make coalescence rate that will be the same function for all updates in one time step
        self.coalescence_rate = Function(Vt)

        # cap coalescence rate so that negative concentrations don't occur
        self.lim_coalescence_rate = Interpolator(conditional(collection + autoconversion < 0,
                                                             max_value(collection + autoconversion, - self.rain / dt),
                                                             min_value(collection + autoconversion, self.water_c / dt)),
                                                 self.coalescence_rate)

        # initiate updating of prognostic variables
        self.water_c_new = Interpolator(self.water_c - dt * self.coalescence_rate, Vt)
        self.rain_new = Interpolator(self.rain + dt * self.coalescence_rate, Vt)

    def apply(self):
        self.lim_coalescence_rate.interpolate()
        self.water_c.assign(self.water_c_new.interpolate())
        self.rain.assign(self.rain_new.interpolate())
