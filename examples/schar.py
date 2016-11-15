from gusto import *
from firedrake import Expression, FunctionSpace, as_vector,\
    VectorFunctionSpace, PeriodicIntervalMesh, ExtrudedMesh, Constant, SpatialCoordinate, exp, cos, pi
import sys

dt = 8.0
if '--running-tests' in sys.argv:
    tmax = dt
else:
    tmax = 18000.

nlayers = 50  # horizontal layers
columns = 100  # number of columns
L = 100000.
m = PeriodicIntervalMesh(columns, L)

# build volume mesh
H = 30000.  # Height position of the model top
ext_mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
Vc = VectorFunctionSpace(ext_mesh, "DG", 2)
coord = SpatialCoordinate(ext_mesh)
x = Function(Vc).interpolate(as_vector([coord[0],coord[1]]))
H = Constant(H)
a = Constant(5000.)
xc = Constant(L/2.)
h = Constant(250.)
ll = Constant(4000.)
smooth_z = False
if smooth_z:
    xexpr = Expression(("x[0]","x[1] < zh ? x[1]+pow(cos(0.5*pi*x[1]/zh),6)*h*exp(-pow((x[0]-xc)/a,2))*pow(cos(pi*(x[0]-xc)/ll),2) : x[1]"), zh=5000., h=h, a=a, xc=xc, H=H, ll=ll)
    new_coords = Function(Vc).interpolate(xexpr)
else:
    new_coords = Function(Vc).interpolate(as_vector([x[0], x[1]+(H-x[1])*h*exp(-pow((x[0]-xc)/a,2))*pow(cos(pi*(x[0]-xc)/ll),2)/H]))

mesh = Mesh(new_coords)

# Space for initialising velocity
W_VectorCG = VectorFunctionSpace(mesh, "CG", 2)
W_CG = FunctionSpace(mesh, "CG", 2)
W_DG = FunctionSpace(mesh, "DG", 2)

# vertical coordinate and normal
z = Function(W_CG).interpolate(Expression("x[1]"))
k = Function(W_VectorCG).interpolate(Expression(("0.","1.")))

mu_top = Expression("x[1] <= zc ? 0.0 : mubar*pow(sin((pi/2.)*(x[1]-zc)/(H-zc)),2)", H=H, zc=(H-10000.), mubar=1.2/dt)
# mu_top = Expression("x[1] <= H-wb ? 0.0 : 0.5*alpha*(1.+cos((x[1]-H)*pi/wb))", H=H, alpha=0.01, wb=7000.)
mu = Function(W_DG).interpolate(mu_top)
fieldlist = ['u', 'rho', 'theta']
timestepping = TimesteppingParameters(dt=dt)
output = OutputParameters(dirname='schar', dumpfreq=18, dumplist=['u'])
parameters = CompressibleParameters(g=9.80665, cp=1004.)
diagnostics = Diagnostics(*fieldlist)
diagnostic_fields = [CourantNumber(), VerticalVelocity()]

state = CompressibleState(mesh, vertical_degree=1, horizontal_degree=1,
                          family="CG",
                          z=z, k=k, mu=mu,
                          timestepping=timestepping,
                          output=output,
                          parameters=parameters,
                          diagnostics=diagnostics,
                          fieldlist=fieldlist,
                          diagnostic_fields=diagnostic_fields,
                          on_sphere=False)

# Initial conditions
u0, rho0, theta0 = Function(state.V[0]), Function(state.V[1]), Function(state.V[2])

# Thermodynamic constants required for setting initial conditions
# and reference profiles
g = parameters.g
N = parameters.N
p_0 = parameters.p_0
c_p = parameters.cp
R_d = parameters.R_d
kappa = parameters.kappa

# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
Tsurf = 288.
thetab = Tsurf*exp(N**2*z/g)
theta_b = Function(state.V[2]).interpolate(thetab)

# Calculate hydrostatic Pi
params = {'pc_type': 'fieldsplit',
          'pc_fieldsplit_type': 'schur',
          'ksp_type': 'gmres',
          'ksp_monitor_true_residual': True,
          'ksp_max_it': 1000,
          'ksp_gmres_restart': 50,
          'pc_fieldsplit_schur_fact_type': 'FULL',
          'pc_fieldsplit_schur_precondition': 'selfp',
          'fieldsplit_0_ksp_type': 'richardson',
          'fieldsplit_0_ksp_max_it': 5,
          'fieldsplit_0_pc_type': 'bjacobi',
          'fieldsplit_0_sub_pc_type': 'ilu',
          'fieldsplit_1_ksp_type': 'richardson',
          'fieldsplit_1_ksp_max_it': 5,
          "fieldsplit_1_ksp_monitor_true_residual": True,
          'fieldsplit_1_pc_type': 'bjacobi',
          'fieldsplit_1_sub_pc_type': 'ilu'}
Pi = Function(state.V[1])
rho_b = Function(state.V[1])
compressible_hydrostatic_balance(state, theta_b, rho_b, Pi, top=True, pi_boundary=Constant(0.5), params=params)


def min(f):
    fmin = op2.Global(1, [1000], dtype=float)
    op2.par_loop(op2.Kernel("""void minify(double *a, double *b)
    {
    a[0] = a[0] > fabs(b[0]) ? fabs(b[0]) : a[0];
    }""", "minify"),
                 f.dof_dset.set, fmin(op2.MIN), f.dat(op2.READ))
    return fmin.data[0]


p0 = min(Pi)
compressible_hydrostatic_balance(state, theta_b, rho_b, Pi, top=True, params=params)
p1 = min(Pi)
alpha = 2.*(p1-p0)
beta = p1-alpha
pi_top = (1.-beta)/alpha
compressible_hydrostatic_balance(state, theta_b, rho_b, Pi, top=True, pi_boundary=Constant(pi_top), solve_for_rho=True, params=params)

theta0.assign(theta_b)
rho0.assign(rho_b)
u0.project(as_vector([10.0,0.0]))
remove_initial_w(u0, state.Vv)

state.initialise([u0, rho0, theta0])
state.set_reference_profiles(rho_b, theta_b)
state.output.meanfields = {'rho':state.rhobar, 'theta':state.thetabar}

# Set up advection schemes
ueqn = MomentumEquation(state, state.V[0], vector_invariant="EulerPoincare")
rhoeqn = AdvectionEquation(state, state.V[1], continuity=True)
supg = True
if supg:
    thetaeqn = AdvectionEquation(state, state.V[2], supg={"dg_directions":[0]}, continuity=False)
else:
    thetaeqn = AdvectionEquation(state, state.V[2], embedded_dg_space="Default", continuity=False)
advection_dict = {}
advection_dict["u"] = ImplicitMidpoint(state, u0, ueqn)
advection_dict["rho"] = SSPRK3(state, rho0, rhoeqn)
advection_dict["theta"] = SSPRK3(state, theta0, thetaeqn)

# Set up linear solver
schur_params = {'pc_type': 'fieldsplit',
                'pc_fieldsplit_type': 'schur',
                'ksp_type': 'gmres',
                'ksp_monitor_true_residual': True,
                'ksp_max_it': 100,
                'ksp_gmres_restart': 50,
                'pc_fieldsplit_schur_fact_type': 'FULL',
                'pc_fieldsplit_schur_precondition': 'selfp',
                'fieldsplit_0_ksp_type': 'richardson',
                'fieldsplit_0_ksp_max_it': 5,
                'fieldsplit_0_pc_type': 'bjacobi',
                'fieldsplit_0_sub_pc_type': 'ilu',
                'fieldsplit_1_ksp_type': 'richardson',
                'fieldsplit_1_ksp_max_it': 5,
                "fieldsplit_1_ksp_monitor_true_residual": True,
                'fieldsplit_1_pc_type': 'gamg',
                'fieldsplit_1_pc_gamg_sym_graph': True,
                'fieldsplit_1_mg_levels_ksp_type': 'chebyshev',
                'fieldsplit_1_mg_levels_ksp_chebyshev_estimate_eigenvalues': True,
                'fieldsplit_1_mg_levels_ksp_chebyshev_estimate_eigenvalues_random': True,
                'fieldsplit_1_mg_levels_ksp_max_it': 5,
                'fieldsplit_1_mg_levels_pc_type': 'bjacobi',
                'fieldsplit_1_mg_levels_sub_pc_type': 'ilu'}

linear_solver = CompressibleSolver(state, params=schur_params)

# Set up forcing
compressible_forcing = CompressibleForcing(state)

# build time stepper
stepper = Timestepper(state, advection_dict, linear_solver,
                      compressible_forcing)

stepper.run(t=0, tmax=tmax)