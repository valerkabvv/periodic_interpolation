from iteropolation import make_knots, make_periodic_spline
from numpy.testing import assert_allclose
import numpy as np

def test_make_knots_size():
    x = np.linspace(0, 2 * np.pi, 50)
    points = x[:-1]
    k=3
    rightBorder = x[-1]
    t=make_knots(points,rightBorder,k)
    assert t.shape[0]==len(points)+2*k+1


    points = [1, 3, 4, 6]
    k = 3
    rightBorder = 7
    t = make_knots(points, rightBorder, k)
    assert t.shape[0] == len(points) + 2 * k + 1


def test_make_knots_schoenberg_whitney():
    x = np.linspace(0, 2 * np.pi, 50)
    points = x[:-1]
    k = 3
    rightBorder = x[-1]
    t = make_knots(points, rightBorder, k)
    for i in range(len(points)):
        assert points[i]>t[i] and points[i]<t[i+k+2]

    points = [1, 4, 5, 6]
    k = 3
    rightBorder = 7
    t = make_knots(points, rightBorder, k)
    assert t.shape[0] == len(points) + 2 * k + 1


def test_interpolate():
    x = [1, 2, 3, 4, 5, 6, 7]
    k = 3
    y = [2.0, 1.0, 4.0, 3.0, 6.0, 7.0, 5.0]
    spl = make_periodic_spline(x, y, k)
    assert_allclose(y[:-1], [spl(i) for i in x[:-1]])


def test_make_periodic_spline():
    x=[1, 2,3,4,5,6,7]
    k=3
    y =[2,1,4,3,6,7,5]
    spl = make_periodic_spline(x,y,k)
    assert_allclose(y[:-1],[spl(i) for i in x[:-1]])
    assert_allclose([spl(x[0],i) for i in range(k)], [spl(x[-1],i) for i in range(k)])

def test_make_periodic_spline_sin():
    x=np.linspace(0,2*np.pi,50)
    y=np.sin(x)
    k=3
    spl = make_periodic_spline(x, y, k)
    spl_value=np.asarray([spl(i) for i in x[:-1]])
    assert_allclose(y[:-1], spl_value,atol=1e-14)
    assert_allclose([spl(x[0], i) for i in range(k)], [spl(x[-1], i) for i in range(k)],atol=1e-14)