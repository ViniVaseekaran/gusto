from gusto import *
from firedrake import PeriodicIntervalMesh, ExtrudedMesh, \
    cos, sin, exp, pi, SpatialCoordinate, Constant, Function, as_vector, DirichletBC
import numpy as np
import sympy as sp
from sympy.stats import Normal
import sys

dt = 1./20
#dt = 0.01
#dt = 0.005

if '--running-tests' in sys.argv:
    tmax = dt
else:
    tmax = 3600*48.
    #tmax = 1
 
##############################################################################
# set up mesh
##############################################################################
# Construct 1d periodic base mesh for idealised lab experiment of Park et al. (1994)
columns = 20  # number of columns
#columns = 40
#columns = 80
L = 0.2
m = PeriodicIntervalMesh(columns, L)

# build 2D mesh by extruding the base mesh
nlayers = 45  # horizontal layers
#nlayers = 90
#nlayers = 180
H = 0.45  # Height position of the model top
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
timestepping = TimesteppingParameters(dt=dt)
#timestepping = TimesteppingParameters(dt=4*dt)
#timestepping = TimesteppingParameters(dt=3*dt)

# class containing output parameters
# all values not explicitly set here use the default values provided
# and documented in configuration.py
dumpfreq = 20*10
#dumpfreq = 200
#dumpfreq = 400
output = OutputParameters(dirname='tmp', dumpfreq=dumpfreq, dumplist=['u','b'], perturbation_fields=['b'])
#points = [[0.1,0.22]]
#output = OutputParameters(dirname='tmp', dumpfreq=dumpfreq, dumplist=['u','b'], perturbation_fields=['b'], 
#            point_data={'b': points}, pointwise_everydump=True)

# class containing physical parameters
# all values not explicitly set here use the default values provided
# and documented in configuration.py
rho0 = 1090.95075
#N=1.957 (run 18), N=1.1576 (run 16), N=0.5916 (run 14), N=0.2
parameters = CompressibleParameters(N=1.957, p_0=106141.3045)

# Physical parameters adjusted for idealised lab experiment of Park et al. (1994):
# The value of the background buoyancy frequency N is that for their run number 18, which has clear stair-step features.
# p_0 was found by assuming an initially hydrostatic fluid and a reference density rho0=1090.95075 kg m^(-3).
# The reference density was found by estimating drho/dz from Fig. 7a of Park et al. (1994), converting to SI units,
# and then using the N^2 value above.
# p_0=106141.3045

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

# Define bouyancy perturbation to represent background soup of internal waves in idealised lab scenario of Park et al.
g = parameters.g

rho0_13 = 1006.47
drho0_dz13 = -122.09
dgamma13 = 100./3
dz_b = 2./100
rhoprime13 = dgamma13 * dz_b
bprime13 = g/rho0_13 * rhoprime13

#No clear number for buoyancy perturbation for run 18 -
#Try to scale perturbations using background stratification
#From Park et al run 18:
rho0_18 = 1090.95075
drho0_dz18 = -425.9
dgamma18 = dgamma13 * drho0_dz18/drho0_dz13

bprime_ratio = rho0_18/rho0_13 * dgamma13/dgamma18
bprime18 = bprime13/bprime_ratio
A_z1 = bprime18


#b_pert = A_x1/2.*sin(k1*x[0]) + A_z1/2.*sin(m1*x[1])
#b_pert = A_z1/2. * sin(k1*x[0]+m1*x[1])
#b_pert = A_z1/2. * sin(m1*x[1])

#sigma = 0.01
#b_pert = A_z1*exp( -( x[1] - H/2 )**2 / (2*sigma**2) )

r = Function(b0.function_space()).assign(Constant(0.0))
r.dat.data[:] += np.random.uniform(low=-1., high=1., size=r.dof_dset.size)
b_pert = r*A_z1/2.
#b_pert = r*A_z1/4.
#b_pert = r*A_z1/6.

#b_pert = sp.Piecewise( (0, x[1] < H/2-0.01), (0, x[1] > H/2+0.01), (A_z1, H/2-0.01 >= x[1] <= H/2+0.01, True) ) - doesn't work
#b_pert = sp.integrate( A_z1 * DiracDelta(x[1]-H/2), (x[1],0,H) ) - doesn't work

# interpolate the expression to the function
b0.interpolate(b_b + b_pert)


incompressible_hydrostatic_balance(state, b_b, p0, top=False)

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
advected_fields.append(("u", ThetaMethod(state, u0, ueqn)))
advected_fields.append(("b", SSPRK3(state, b0, beqn)))
#advected_fields.append(("u", SSPRK3(state, u0, ueqn, subcycles=4)))
#advected_fields.append(("b", SSPRK3(state, b0, beqn, subcycles=4)))




##############################################################################
# Set up linear solver for the timestepping scheme
##############################################################################
linear_solver = IncompressibleSolver(state, L)

##############################################################################
# Set up forcing
#############################################################################
lmda_x1 = 2.0/100                               # Horizontal wavelength of internal waves
lmda_z1 = 2.0/100                               # Vertical wavelength of internal waves
k1 = 2*pi/lmda_x1                               # Horizontal wavenumber of internal waves
m1 = 2*pi/lmda_z1                               # Vertical wavenumber of internal waves

omega = 0.2*2*pi
f_ux = 0.
#f_uz = 0.
f_uz = A_z1/2*sin(x[0]*k1 + x[1]*m1 - omega*state.t)
f_u = as_vector([f_ux,f_uz])

#forcing = IncompressibleForcing(state)
forcing = IncompressibleForcing(state, extra_terms=f_u)
#forcing = RandomIncompressibleForcing(state)


##############################################################################
#Set up diffusion scheme
##############################################################################
# mu is a numerical parameter
# kappa is the diffusion constant for each variable
# Note that molecular diffusion coefficients were taken from Lautrup, 2005:
kappa_u = 1.*10**(-6.)/10
kappa_b = 1.4*10**(-7.)/10

#kappa_u = 1.*10**(-6.)/5
#kappa_b = 1.4*10**(-7.)/5

#kappa_u = 1.*10**(-6.)/2
#kappa_b = 1.4*10**(-7.)/2

#kappa_u = 1.*10**(-6.)
#kappa_b = 1.4*10**(-7.)

#kappa_u = 1.*10**(-6.)*10
#kappa_b = 1.4*10**(-7.)*10

#kappa_u = 1.*10**(-6.)*100
#kappa_b = 1.4*10**(-7.)*100

#Eddy diffusivities:
#kappa_u = 10.**(-2.)
#kappa_b = 10.**(-2.)

Vu = u0.function_space()
Vb = state.spaces("HDiv_v")
delta = L/columns 		#Grid resolution (same in both directions).

bcs_u = [DirichletBC(Vu, 0.0, "bottom"), DirichletBC(Vu, 0.0, "top")]
bcs_b = [DirichletBC(Vb, -N**2*H, "bottom"), DirichletBC(Vb, 0.0, "top")]

diffused_fields = []
diffused_fields.append(("u", InteriorPenalty(state, Vu, kappa=kappa_u,
                                           mu=Constant(10./delta), bcs=bcs_u)))
diffused_fields.append(("b", InteriorPenalty(state, Vb, kappa=kappa_b,
                                             mu=Constant(10./delta), bcs=bcs_b)))


##############################################################################
# build time stepper
##############################################################################
#stepper = Timestepper(state, advection_dict, linear_solver, forcing)
stepper = Timestepper(state, advected_fields, linear_solver, forcing, diffused_fields)

##############################################################################
# Run!
##############################################################################
stepper.run(t=0, tmax=tmax)
