from iteropolation import make_knots, make_periodic_spline
from numpy.testing import assert_allclose
import numpy as np
from scipy.interpolate._cubic import CubicSpline



def test_interpolate():
    x = [1, 2, 3, 4, 5, 6, 7,8]
    k = 4
    y = [2.0, 1.0, 4.0, 3.0, 6.0, 7.0, 5.0,1.0]
    spl = make_periodic_spline(x, y, k)
    assert_allclose(y[:-1], [spl(i) for i in x[:-1]])


def test_make_periodic_spline():
    x=[1,3,5,6,7,10,11.5]
    k=4
    y =[2,1,4,3,6,7,5]
    spl = make_periodic_spline(x,y,k)
    assert_allclose(y[:-1],[spl(i) for i in x[:-1]])
    assert_allclose([spl(x[0],i) for i in range(k)], [spl(x[-1],i) for i in range(k)])

def test_make_periodic_spline_sin():
    x=np.linspace(0,2*np.pi,60)
    y=np.sin(x)
    k=3
    spl = make_periodic_spline(x, y, k)
    spl_value=np.asarray([spl(i) for i in x[:-1]])
    assert_allclose(y[:-1], spl_value,atol=1e-14)
    assert_allclose([spl(x[0], i) for i in range(k)], [spl(x[-1], i) for i in range(k)],atol=1e-14)


def test_agais_CubicSpline():
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    k=3
    spl = make_periodic_spline(x, y, k)
    cub_spl = CubicSpline(x,y,bc_type='periodic')
    spl_value = np.asarray([spl(i) for i in x[:-1]])
    cubic_value = np.asarray([cub_spl(i) for i in x[:-1]])
    assert_allclose(cubic_value, spl_value, atol=1e-14)
    assert_allclose([spl(x[0], i) for i in range(k)], [spl(x[-1], i) for i in range(k)], atol=1e-14)