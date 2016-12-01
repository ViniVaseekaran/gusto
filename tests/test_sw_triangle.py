from gusto import *
from firedrake import IcosahedralSphereMesh, Expression, SpatialCoordinate, \
    Constant, as_vector
from math import pi
import json
import pytest


def setup_sw(dirname, euler_poincare):

    refinements = 3  # number of horizontal cells = 20*(4^refinements)

    R = 6371220.
    H = 5960.
    day = 24.*60.*60.
    u_0 = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)

    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=refinements)
    global_normal = Expression(("x[0]", "x[1]", "x[2]"))
    mesh.init_cell_orientations(global_normal)

    fieldlist = ['u', 'D']
    timestepping = TimesteppingParameters(dt=1500.)
    output = OutputParameters(dirname=dirname+"/sw", dumplist_latlon=['D','Derr'], steady_state_dump_err={'D':True,'u':True})
    parameters = ShallowWaterParameters(H=H)
    diagnostics = Diagnostics(*fieldlist)

    state = ShallowWaterState(mesh, vertical_degree=None, horizontal_degree=1,
                              family="BDM",
                              timestepping=timestepping,
                              output=output,
                              parameters=parameters,
                              diagnostics=diagnostics,
                              fieldlist=fieldlist)

    # interpolate initial conditions
    u0, D0 = Function(state.V[0]), Function(state.V[1])
    x = SpatialCoordinate(mesh)
    u_max = Constant(u_0)
    R = Constant(R)
    uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
    h0 = Constant(H)
    Omega = Constant(parameters.Omega)
    g = Constant(parameters.g)
    Dexpr = h0 - ((R * Omega * u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R*R)))/g
    # Coriolis expression
    fexpr = 2*Omega*x[2]/R
    V = FunctionSpace(mesh, "CG", 1)
    state.f = Function(V).interpolate(fexpr)  # Coriolis frequency (1/s)

    u0.project(uexpr)
    D0.interpolate(Dexpr)
    state.initialise([u0, D0])

    if euler_poincare:
        ueqn = EulerPoincare(state, state.V[0])
        sw_forcing = ShallowWaterForcing(state)
    else:
        ueqn = VectorInvariant(state, state.V[0])
        sw_forcing = ShallowWaterForcing(state, euler_poincare=False)

    Deqn = Advection(state, state.V[1], continuity=True)
    advection_dict = {}
    advection_dict["u"] = ThetaMethod(state, u0, ueqn)
    advection_dict["D"] = SSPRK3(state, D0, Deqn)

    linear_solver = ShallowWaterSolver(state)

    # build time stepper
    stepper = Timestepper(state, advection_dict, linear_solver,
                          sw_forcing)

    return stepper, 0.25*day


def run_sw(dirname, euler_poincare):

    stepper, tmax = setup_sw(dirname, euler_poincare)
    stepper.run(t=0, tmax=tmax)


@pytest.mark.parametrize("euler_poincare", [True, False])
def test_sw_setup(tmpdir, euler_poincare):

    dirname = str(tmpdir)
    run_sw(dirname, euler_poincare=euler_poincare)
    with open(path.join(dirname, "sw/diagnostics.json"), "r") as f:
        data = json.load(f)
    Dl2 = data["Derr"]["l2"][-1]/data["D"]["l2"][0]
    ul2 = data["uerr"]["l2"][-1]/data["u"]["l2"][0]

    assert Dl2 < 5.e-4
    assert ul2 < 5.e-3
