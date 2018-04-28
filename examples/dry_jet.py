from gusto import *
from firedrake import PeriodicIntervalMesh, ExtrudedMesh, \
    SpatialCoordinate, conditional, cos, pi, sqrt, \
    TestFunction, dx, TrialFunction, Constant, Function, \
    LinearVariationalProblem, LinearVariationalSolver, DirichletBC, \
    BrokenElement, FunctionSpace, VectorFunctionSpace, \
    NonlinearVariationalProblem, NonlinearVariationalSolver, exp, \
    MixedFunctionSpace, split, TestFunctions, as_vector, sin
from firedrake.slope_limiter.vertex_based_limiter import VertexBasedLimiter
import sys

dt = 1.0
if '--running-tests' in sys.argv:
    tmax = 10.
    deltax = 1000.
else:
    deltax = 20.
    tmax = 1000.

L = 3600.
H = 2400.
nlayers = int(H/deltax)
ncolumns = int(L/deltax)

m = PeriodicIntervalMesh(ncolumns, L)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
recovered = True
degree = 0 if recovered else 1

fieldlist = ['u', 'rho', 'theta']
timestepping = TimesteppingParameters(dt=dt, maxk=4, maxi=1)
output = OutputParameters(dirname='dry_jet', dumpfreq=10, dumplist=['u', 'rho', 'theta'], perturbation_fields=['rho', 'theta'], log_level='INFO')
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
x, z = SpatialCoordinate(mesh)
quadrature_degree = (5, 5)
dxp = dx(degree=(quadrature_degree))

VDG1 = FunctionSpace(mesh, "DG", 1)
VCG1 = FunctionSpace(mesh, "CG", 1)
Vt_brok = FunctionSpace(mesh, BrokenElement(Vt.ufl_element()))
Vu_DG1 = VectorFunctionSpace(mesh, "DG", 1)
Vu_CG1 = VectorFunctionSpace(mesh, "CG", 1)

u_spaces = (Vu_DG1, Vu_CG1, Vu)
rho_spaces = (VDG1, VCG1, Vr)
theta_spaces = (VDG1, VCG1, Vt_brok)

# Define constant theta_e and water_t
Tsurf = 283.0
psurf = 85000.
Pi_surf = (psurf / state.parameters.p_0) ** state.parameters.kappa
theta_surf = thermodynamics.theta(state.parameters, Tsurf, psurf)
S = 1.3e-5
theta0.assign(theta_surf)
                              
# Calculate hydrostatic fields
compressible_hydrostatic_balance(state, theta0, rho0, pi_boundary=Constant(Pi_surf))

# make mean fields
theta_b = Function(Vt).assign(theta0)
rho_b = Function(Vr).assign(rho0)

# define perturbation
xc = L / 4
zc = 600.
rc = 600.
r = sqrt((x - xc) ** 2 + (z - zc) ** 2)
tdash = 2.0

pert_expr = conditional(r < rc, tdash * (cos(pi * r / (2 *rc))) ** 2, 0.0)

theta_pert = Function(Vt).interpolate(pert_expr)

theta0.assign(theta_b + theta_pert)

# find perturbed rho
gamma = TestFunction(Vr)
rho_trial = TrialFunction(Vr)
a = gamma * rho_trial * dx
Lrhs = gamma * (rho_b * theta_b / theta0) * dx
rho_problem = LinearVariationalProblem(a, Lrhs, rho0)
rho_solver = LinearVariationalSolver(rho_problem)
rho_solver.solve()

U = 5.
V = 5.

u0.project(as_vector([U, V*sin(2*pi*x/L - pi/4)*sin(2*pi*z/H)]))

tracer0 = state.fields('tracer', Vt).assign(theta0)

# initialise fields
state.initialise([('u', u0),
                  ('rho', rho0),
                  ('theta', theta0),
                  ('tracer', tracer0)])
state.set_reference_profiles([('rho', rho_b),
                              ('theta', theta_b)])

# Set up advection schemes
ueqn = EmbeddedDGAdvection(state, Vu, equation_form="advective", recovered_spaces=u_spaces)
rhoeqn = EmbeddedDGAdvection(state, Vr, equation_form="continuity", recovered_spaces=rho_spaces)
thetaeqn = EmbeddedDGAdvection(state, Vt, equation_form="advective", recovered_spaces=theta_spaces)

advected_fields = [('u', NoAdvection(state, u0, None)),
                   ('rho', SSPRK3(state, rho0, rhoeqn)),
                   ('theta', SSPRK3(state, theta0, thetaeqn)),
                   ('tracer', SSPRK3(state, tracer0, thetaeqn))]

# define condensation
physics_list = []

# build time stepper
stepper = AdvectionDiffusion(state, advected_fields, physics_list=physics_list)

stepper.run(t=0, tmax=tmax)
