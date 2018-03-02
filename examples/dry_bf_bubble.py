from gusto import *
from gusto import thermodynamics
from firedrake import PeriodicIntervalMesh, ExtrudedMesh, \
    SpatialCoordinate, conditional, cos, pi, sqrt, NonlinearVariationalProblem, \
    NonlinearVariationalSolver, TestFunction, dx, TrialFunction, Constant, Function, \
    LinearVariationalProblem, LinearVariationalSolver, DirichletBC, \
    FunctionSpace, BrokenElement, VectorFunctionSpace
from firedrake.slope_limiter.vertex_based_limiter import VertexBasedLimiter
import sys

dt = 1.0
if '--running-tests' in sys.argv:
    tmax = 10.
    deltax = 1000.
else:
    deltax = 100.
    tmax = 1000.

L = 10000.
H = 10000.
nlayers = int(H/deltax)
ncolumns = int(L/deltax)

m = PeriodicIntervalMesh(ncolumns, L)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
diffusion = False
recovered = True
degree = 0 if recovered else 1

fieldlist = ['u', 'rho', 'theta']
timestepping = TimesteppingParameters(dt=dt, maxk=4, maxi=1)
output = OutputParameters(dirname='dry_bf', dumpfreq=1, dumplist=['u'], perturbation_fields=['theta', 'rho'], log_level='INFO')
params = CompressibleParameters()
diagnostics = Diagnostics(*fieldlist)
diagnostic_fields = []

state = State(mesh, vertical_degree=degree, horizontal_degree=degree,
              family="CG",
              timestepping=timestepping,
              output=output,
              parameters=params,
              diagnostics=diagnostics,
              fieldlist=fieldlist,
              diagnostic_fields=diagnostic_fields)

# Initial conditions
u0 = state.fields("u")
rho0 = state.fields("rho")
theta0 = state.fields("theta")

# spaces
Vu = u0.function_space()
Vt = theta0.function_space()
Vr = rho0.function_space()
x = SpatialCoordinate(mesh)

if recovered:
    VDG1 = FunctionSpace(mesh, "DG", 1)
    VCG1 = FunctionSpace(mesh, "CG", 1)
    Vt_brok = FunctionSpace(mesh, BrokenElement(Vt.ufl_element()))
    Vu_DG1 = VectorFunctionSpace(mesh, "DG", 1)
    Vu_CG1 = VectorFunctionSpace(mesh, "CG", 1)
    Vu_brok = FunctionSpace(mesh, BrokenElement(Vu.ufl_element()))

    u_spaces = (Vu_DG1, Vu_CG1, Vu_brok)
    rho_spaces = (VDG1, VCG1, Vr)
    theta_spaces = (VDG1, VCG1, Vt_brok)

# Define constant theta_e and water_t
Tsurf = 300.0
theta_b = Function(Vt).interpolate(Constant(Tsurf))

# Calculate hydrostatic fields
compressible_hydrostatic_balance(state, theta_b, rho0, solve_for_rho=True)

# make mean fields
rho_b = Function(Vr).assign(rho0)

# define perturbation
xc = L / 2
zc = 2000.
rc = 2000.
Tdash = 2.0
theta_pert = Function(Vt).interpolate(conditional(sqrt((x[0] - xc) ** 2 + (x[1] - zc) ** 2) > rc,
                                                  0.0, Tdash *
                                                  (cos(pi * sqrt(((x[0] - xc) / rc) ** 2 + ((x[1] - zc) / rc) ** 2) / 2.0))
                                                  ** 2))

# define initial theta
theta0.assign(theta_b * (theta_pert / 300.0 + 1.0))

# find perturbed rho
gamma = TestFunction(Vr)
rho_trial = TrialFunction(Vr)
a = gamma * rho_trial * dx
L = gamma * (rho_b * theta_b / theta0) * dx
rho_problem = LinearVariationalProblem(a, L, rho0)
rho_solver = LinearVariationalSolver(rho_problem)
rho_solver.solve()

# initialise fields
state.initialise([('u', u0),
                  ('rho', rho0),
                  ('theta', theta0)])
state.set_reference_profiles([('rho', rho_b),
                              ('theta', theta_b)])

# Set up advection schemes
if recovered:
    ueqn = EmbeddedDGAdvection(state, Vu, equation_form="advective", recovered_spaces=u_spaces)
    rhoeqn = EmbeddedDGAdvection(state, Vr, equation_form="continuity", recovered_spaces=rho_spaces)
    thetaeqn = EmbeddedDGAdvection(state, Vt, equation_form="advective", recovered_spaces=theta_spaces)
else:
    ueqn = EulerPoincare(state, Vu)
    rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity")
    thetaeqn = EmbeddedDGAdvection(state, Vt, equation_form="advective")

advected_fields = [('rho', SSPRK3(state, rho0, rhoeqn)),
                   ('theta', SSPRK3(state, theta0, thetaeqn))]
if recovered:
    advected_fields.append(('u', SSPRK3(state, u0, ueqn)))
else:
    advected_fields.append(('u', ThetaMethod(state, u0, ueqn)))

linear_solver = CompressibleSolver(state)

# Set up forcing
if recovered:
    compressible_forcing = CompressibleForcing(state, euler_poincare=False)
else:
    compressible_forcing = CompressibleForcing(state)

# diffusion
bcs = [DirichletBC(Vu, 0.0, "bottom"),
       DirichletBC(Vu, 0.0, "top")]

diffused_fields = []

if diffusion:
    diffused_fields.append(('u', InteriorPenalty(state, Vu, kappa=Constant(60.),
                                                 mu=Constant(10./deltax), bcs=bcs)))

# build time stepper
stepper = CrankNicolson(state, advected_fields, linear_solver,
                        compressible_forcing,
                        diffused_fields=diffused_fields)

stepper.run(t=0, tmax=tmax)
