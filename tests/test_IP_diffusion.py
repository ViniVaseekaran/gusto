import itertools
from os import path
from gusto import *
from firedrake import SpatialCoordinate,\
    VectorFunctionSpace, File, Constant, Function, exp, as_vector
import pytest


def setup_IPdiffusion(vector, DG):

    dt = 0.01
    L = 10.
    domain = VerticalSliceDomain(L, 10., 50, 50)

    fieldlist = ['u', 'D']
    timestepping = TimesteppingParameters(dt=dt)
    parameters = CompressibleParameters()
    output = OutputParameters(dirname="IPdiffusion")

    state = State(domain,
                  vertical_degree=1, horizontal_degree=1,
                  family="CG",
                  timestepping=timestepping,
                  parameters=parameters,
                  output=output,
                  fieldlist=fieldlist)

    mesh = domain.mesh
    x = SpatialCoordinate(mesh)
    if vector:
        if DG:
            Space = VectorFunctionSpace(mesh, "DG", 1)
        else:
            Space = state.spaces("HDiv")
        f = Function(Space, name="f")
        fexpr = as_vector([exp(-(L/2.-x[0])**2 - (L/2.-x[1])**2), 0.])
    else:
        if DG:
            Space = state.spaces("DG")
        else:
            Space = state.spaces("HDiv_v")
        f = Function(Space, name='f')
        fexpr = exp(-(L/2.-x[0])**2 - (L/2.-x[1])**2)

    try:
        f.interpolate(fexpr)
    except NotImplementedError:
        f.project(fexpr)

    return state, f


def run(dirname, vector, DG):

    state, f = setup_IPdiffusion(vector, DG)

    kappa = 0.05
    if vector:
        kappa = Constant([[0.05, 0.], [0., 0.05]])
    mu = 5.
    dt = state.timestepping.dt
    tmax = 2.5
    t = 0.
    f_diffusion = InteriorPenalty(state, f.function_space(), kappa=kappa, mu=mu)
    outfile = File(path.join(dirname, "IPdiffusion/field_output.pvd"))

    dumpcount = itertools.count()

    outfile.write(f)

    fp1 = Function(f.function_space())

    while t < tmax - 0.5*dt:
        t += dt
        f_diffusion.apply(f, fp1)
        f.assign(fp1)

        if (next(dumpcount) % 25) == 0:
            outfile.write(f)
    return f


@pytest.mark.parametrize("vector", [True, False])
@pytest.mark.parametrize("DG", [True, False])
def test_ipdiffusion(tmpdir, vector, DG):

    dirname = str(tmpdir)
    f = run(dirname, vector, DG)
    assert f.dat.data.max() < 0.7
