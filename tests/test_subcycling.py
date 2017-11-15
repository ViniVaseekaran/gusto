from gusto import *
from firedrake import PeriodicSquareMesh, exp, SpatialCoordinate, Constant, FunctionSpace


def setup_gaussian(dirname):

    parameters = ShallowWaterParameters(H=1.0, g=1.0, f0=1)
    domain = PlaneDomain(nx=16, ny=16, L=1, W=1, parameters=parameters,
                         rotation_option="f_plane")

    fieldlist = ['u', 'D']
    timestepping = TimesteppingParameters(dt=0.1)
    output = OutputParameters(dirname=dirname+'/sw_plane_gaussian_subcycled')
    diagnostic_fields = [CourantNumber()]

    state = State(domain,
                  horizontal_degree=1,
                  family="BDM",
                  timestepping=timestepping,
                  output=output,
                  diagnostic_fields=diagnostic_fields,
                  fieldlist=fieldlist)

    u0 = state.fields("u")
    D0 = state.fields("D")
    x, y = SpatialCoordinate(domain.mesh)
    H = Constant(parameters.H)
    D0.interpolate(H + exp(-50*((x-0.5)**2 + (y-0.5)**2)))

    state.initialise([("u", u0), ("D", D0)])

    ueqn = EmbeddedDGAdvection(state, u0.function_space())
    Deqn = AdvectionEquation(state, D0.function_space(), equation_form="continuity")
    advected_fields = []
    advected_fields.append(("u", SSPRK3(state, u0, ueqn, subcycles=2)))
    advected_fields.append(("D", SSPRK3(state, D0, Deqn, subcycles=2)))

    linear_solver = ShallowWaterSolver(state)

    # Set up forcing
    sw_forcing = ShallowWaterForcing(state)

    # build time stepper
    stepper = CrankNicolson(state, advected_fields, linear_solver,
                            sw_forcing)

    return stepper


def run(dirname):
    stepper = setup_gaussian(dirname)
    stepper.run(t=0, tmax=0.3)


def test_subcycling(tmpdir):
    dirname = str(tmpdir)
    run(dirname)
