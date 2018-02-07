from gusto import *
from firedrake import PeriodicIntervalMesh, ExtrudedMesh, \
    cos, sin, exp, pi, SpatialCoordinate, Constant, Function, as_vector, DirichletBC, tanh, cosh, ln, conditional, eq, VectorFunctionSpace
import numpy as np
import sympy as sp
from sympy.stats import Normal
import sys

#dt = 1./20
#dt = 0.01
#dt = 0.005
dt = 0.0075
#dt = 0.003

if '--running-tests' in sys.argv:
    tmax = dt
else:
    tmax = 3600*48.
    #tmax = 1
 
##############################################################################
# set up mesh
##############################################################################
# Construct 1d periodic base mesh for idealised lab experiment of Park et al. (1994)
#columns = 20  # number of columns
#columns = 40
columns = 80
#columns = 100
L = 0.2
#L = 0.1
m = PeriodicIntervalMesh(columns, L)

# build 2D mesh by extruding the base mesh
#nlayers = 45  # horizontal layers
#nlayers = 90
nlayers = 180
#nlayers = 225
H = 0.45  # Height position of the model top
#H = 0.45/2
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

##############################################################################
# set up all the other things that state requires
##############################################################################


# list of prognostic fieldnames
# this is passed to state and used to construct a dictionary,
# state.field_dict so that we can access fields by name
# u is the 2D velocity
# p is the pressure
# b is the buoyancy
fieldlist = ['u', 'p', 'b']

# class containing timestepping parameters
# all values not explicitly set here use the default values provided
# and documented in configuration.py
subcycles = 4
timestepping = TimesteppingParameters(dt=dt*subcycles)

# class containing output parameters
# all values not explicitly set here use the default values provided
# and documented in configuration.py

dumpfreq = int( 5/(dt*subcycles) )

#points = np.array([[0.05,0.22]])
points = np.array([[0.04,0.21]])
#points_x = [0.05]
#points_z = [0.22]
#points = np.array([p for p in itertools.product(points_x, points_z)])

#output = OutputParameters(dirname='tmp', dumpfreq=dumpfreq, dumplist=['u','b'], 
#perturbation_fields=['b'], point_data=[('b', points)], checkpoint=False)
output = OutputParameters(dirname='kh', dumpfreq=1, dumplist=['u','b'], 
perturbation_fields=['b'], checkpoint=False)

# class containing physical parameters
# all values not explicitly set here use the default values provided
# and documented in configuration.py
# Barotropic Kelvin-Helmholtz case:
parameters = CompressibleParameters(N=0)

# class for diagnostics
# fields passed to this class will have basic diagnostics computed
# (eg min, max, l2 norm) and these will be output as a json file
diagnostics = Diagnostics(*fieldlist)

# list of diagnostic fields, each defined in a class in diagnostics.py
diagnostic_fields = [CourantNumber()]

# setup state, passing in the mesh, information on the required finite element
# function spaces, z, k, and the classes above
state = State(mesh, vertical_degree=1, horizontal_degree=1,
              family="CG",
              timestepping=timestepping,
              output=output,
              parameters=parameters,
              diagnostics=diagnostics,
              fieldlist=fieldlist,
              diagnostic_fields=diagnostic_fields)

##############################################################################
# Initial conditions
##############################################################################
# set up functions on the spaces constructed by state
u0 = state.fields("u")
p0 = state.fields("p")
b0 = state.fields("b")

# first setup the background buoyancy profile
# z.grad(bref) = N**2
# the following is symbolic algebra, using the default buoyancy frequency
# from the parameters class. x[1]=z and comes from x=SpatialCoordinate(mesh)
x = SpatialCoordinate(mesh)
N = parameters.N
bref = N**2*(x[1]-H)
# interpolate the expression to the function
Vb = b0.function_space()
b_b = Function(Vb).interpolate(bref)
b0.interpolate(b_b)

incompressible_hydrostatic_balance(state, b_b, p0, top=False)

speed = Constant(0.01)
dz_u = Constant(0.001)
du = 2*speed
alpha = (x[1]-H/2.)/dz_u
u_base = du/2.*tanh(alpha)
u_pert = 0

psi_base = du*dz_u/2*( ln(2) - H/(2*dz_u) + ln(cosh(alpha)) ) 
PsiAbsMax = 0.0022366517968690366
psi_prime_max = 1/20. * PsiAbsMax
lambda1 = 1./5*L
k1 = 2*pi/lambda1

w_base = 0
w_pert = conditional(eq(x[1], H/2.), -psi_prime_max*k1*cos(k1*x[0]), 0.)
Vu = VectorFunctionSpace(mesh, "CG", 2)
u_init = Function(Vu).interpolate(as_vector([u_base + u_pert, w_base + w_pert]))
u0.project(u_init)

kappa_u = 1.e-6
fx = -kappa_u*du/dz_u**2*(tanh(alpha)*(1-tanh(alpha)**2))*L
fxz = (du/(2*dz_u)*(1-tanh(alpha)**2)-k1**2*psi_base)*psi_prime_max*sin(k1*x[0])
p = fx + fxz

# pass these initial conditions to the state.initialise method
state.initialise([("u", u0), ("p", p0), ("b", b0)])

# set the background buoyancy
state.set_reference_profiles([("b", b_b)])

##############################################################################
# Set up advection schemes
##############################################################################
# advection_dict is a dictionary containing field_name: advection class
ueqn = EulerPoincare(state, u0.function_space())
supg = True
if supg:
    beqn = SUPGAdvection(state, Vb,
                         supg_params={"dg_direction":"horizontal"},
                         equation_form="advective")
else:
    beqn = EmbeddedDGAdvection(state, Vb,
                               equation_form="advective")
advected_fields = []
#advected_fields.append(("u", ThetaMethod(state, u0, ueqn)))
#advected_fields.append(("b", SSPRK3(state, b0, beqn)))
advected_fields.append(("u", SSPRK3(state, u0, ueqn, subcycles=subcycles)))
advected_fields.append(("b", SSPRK3(state, b0, beqn, subcycles=subcycles)))

##############################################################################
# Set up linear solver for the timestepping scheme
##############################################################################
linear_solver = IncompressibleSolver(state, L)

##############################################################################
# Set up forcing
#############################################################################
forcing = IncompressibleForcing(state)

##############################################################################
#Set up diffusion scheme
##############################################################################
# mu is a numerical parameter
# kappa is the diffusion constant for each variable
# Note that molecular diffusion coefficients were taken from Lautrup, 2005:
kappa_u = 1.*10**(-6.)
kappa_b = 1.4*10**(-7.)

#Eddy diffusivities:
#kappa_u = 10.**(-2.)
#kappa_b = 10.**(-2.)

Vu = u0.function_space()
Vb = state.spaces("HDiv_v")
delta = L/columns 		#Grid resolution (same in both directions).

bcs_u = [DirichletBC(Vu, 0.0, "bottom"), DirichletBC(Vu, 0.0, "top")]
bcs_b = [DirichletBC(Vb, 0.0, "bottom"), DirichletBC(Vb, 0.0, "top")]

diffused_fields = []
diffused_fields.append(("u", InteriorPenalty(state, Vu, kappa=kappa_u,
                                           mu=Constant(10./delta), bcs=bcs_u)))
#diffused_fields.append(("b", InteriorPenalty(state, Vb, kappa=kappa_b,
#                                             mu=Constant(10./delta), bcs=bcs_b)))


##############################################################################
# build time stepper
##############################################################################
#stepper = CrankNicolson(state, advected_fields, linear_solver, forcing)
stepper = CrankNicolson(state, advected_fields, linear_solver, forcing, diffused_fields)

##############################################################################
# Run!
##############################################################################
stepper.run(t=0, tmax=tmax)
