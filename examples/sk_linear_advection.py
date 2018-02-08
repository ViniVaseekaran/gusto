from gusto import *
from firedrake import PeriodicIntervalMesh, ExtrudedMesh, \
    SpatialCoordinate, exp, sin, Function, FunctionSpace
import numpy as np
import sys

dt = 10.
if '--running-tests' in sys.argv:
    tmax = dt
else:
    tmax = 3600.

if '--hybrid' in sys.argv:
    hybridization = True
else:
    hybridization = False

nlayers = 50  # horizontal layers
columns = 50  # number of columns
L = 3.0e5
m = PeriodicIntervalMesh(columns, L)

# build volume mesh
H = 1.0e4  # Height position of the model top
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

fieldlist = ['u', 'rho', 'theta']
timestepping = TimesteppingParameters(dt=dt)
output = OutputParameters(dirname='sk_linear', dumplist=['u'], perturbation_fields=['theta', 'rho'])
parameters = CompressibleParameters()

state = State(mesh, vertical_degree=1, horizontal_degree=1,
              family="CG",
              timestepping=timestepping,
              output=output,
              parameters=parameters,
              fieldlist=fieldlist)

# Initial conditions
u0 = state.fields("u")
rho0 = state.fields("rho")
theta0 = state.fields("theta")

# spaces
Vu = u0.function_space()
Vt = theta0.function_space()
Vr = rho0.function_space()

# Thermodynamic constants required for setting initial conditions
# and reference profiles
g = parameters.g
N = parameters.N
p_0 = parameters.p_0
c_p = parameters.cp
R_d = parameters.R_d
kappa = parameters.kappa

x, z = SpatialCoordinate(mesh)

# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
Tsurf = 300.
thetab = Tsurf*exp(N**2*z/g)

theta_b = Function(Vt).interpolate(thetab)
rho_b = Function(Vr)

# Calculate hydrostatic Pi
compressible_hydrostatic_balance(state, theta_b, rho_b)

W_DG1 = FunctionSpace(mesh, "DG", 1)
a = 5.0e3
deltaTheta = 1.0e-2
theta_pert = deltaTheta*sin(np.pi*z/H)/(1 + (x - L/2)**2/a**2)
theta0.interpolate(theta_b + theta_pert)
rho0.assign(rho_b)

state.initialise([('u', u0),
                  ('rho', rho0),
                  ('theta', theta0)])
state.set_reference_profiles([('rho', rho_b),
                              ('theta', theta_b)])

# Set up advection schemes
rhoeqn = LinearAdvection(state, Vr, qbar=rho_b, ibp="once", equation_form="continuity")
thetaeqn = LinearAdvection(state, Vt, qbar=theta_b)
advected_fields = []
advected_fields.append(("u", NoAdvection(state, u0, None)))
advected_fields.append(("rho", ForwardEuler(state, rho0, rhoeqn)))
advected_fields.append(("theta", ForwardEuler(state, theta0, thetaeqn)))

# Set up linear solver
if hybridization:
    linear_solver = HybridisedCompressibleSolver(state)
else:
    linear_solver = CompressibleSolver(state)

# Set up forcing
compressible_forcing = CompressibleForcing(state, linear=True)

# build time stepper
stepper = CrankNicolson(state, advected_fields, linear_solver,
                        compressible_forcing)

stepper.run(t=0, tmax=tmax)
