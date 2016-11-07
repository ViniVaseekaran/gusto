from gusto import *
from firedrake import IcosahedralSphereMesh, Expression, SpatialCoordinate, \
    Constant, as_vector
from math import pi
import sys

day = 24.*60.*60.
if '--running-tests' in sys.argv:
    ref_dt = {3:3000.}
    tmax = 3000.
else:
    # setup resolution and timestepping parameters for convergence test
    ref_dt = {3:3000., 4:1500., 5:750., 6:375}
    tmax = 5*day

# setup shallow water parameters
R = 6371220.
H = 5960.
u_0 = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)

# setup input that doesn't change with ref level or dt
fieldlist = ['u', 'D']
parameters = ShallowWaterParameters(H=H)
diagnostics = Diagnostics(*fieldlist)

for ref_level, dt in ref_dt.iteritems():

    dirname = "sw_W2_ref%s_dt%s" % (ref_level, dt)
    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=ref_level, degree=3)
    global_normal = Expression(("x[0]", "x[1]", "x[2]"))
    mesh.init_cell_orientations(global_normal)

    timestepping = TimesteppingParameters(dt=dt)
    output = OutputParameters(dirname=dirname, dumplist_latlon=['D','Derr'], steady_state_dump_err={'D':True,'u':True})

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
    R0 = Constant(R)
    uexpr = as_vector([-u_max*x[1]/R0, u_max*x[0]/R0, 0.0])
    h0 = Constant(H)
    Omega = Constant(parameters.Omega)
    g = Constant(parameters.g)
    Dexpr = h0 - ((R0 * Omega * u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R0*R0)))/g
    # Coriolis expression
    fexpr = 2*Omega*x[2]/R0
    V = FunctionSpace(mesh, "CG", 1)
    state.f = Function(V).interpolate(fexpr)  # Coriolis frequency (1/s)

    u0.project(uexpr)
    D0.interpolate(Dexpr)
    state.initialise([u0, D0])

    advection_dict = {}
    advection_dict["u"] = EulerPoincareForm(state, state.V[0])
    advection_dict["D"] = DGAdvection(state, state.V[1], continuity=True)

    linear_solver = ShallowWaterSolver(state)

    # Set up forcing
    sw_forcing = ShallowWaterForcing(state)

    # build time stepper
    stepper = Timestepper(state, advection_dict, linear_solver,
                          sw_forcing)

    stepper.run(t=0, tmax=tmax)
