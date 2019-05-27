import scipy.interpolate._bsplines as interpolate
import scipy.interpolate._bspl as bspl
import numpy as np
import math
from scipy import linalg


def make_knots(points, k):
    per = points[-1] - points[0]
    points = np.asarray(points, dtype=float)
    t_midle = np.concatenate(
        (points[:1], points[k // 2 + 1:-k // 2 - 1], np.linspace(points[-2], points[-1], k + 1)))
    t_1 = np.asarray([x - per for x in t_midle[-k - 1:-1]])
    t_2 = np.asarray([x + per for x in t_midle[1:k + 1]])
    return np.concatenate((t_1, t_midle, t_2))



def make_periodic_spline(x,y,k):
    t = make_knots(x, k)
    x_1=interpolate._as_float_array(x[:-1])
    t_1=interpolate._as_float_array(t[:-k])
    banded_storage = np.zeros((3*k+1,len(y)-1),dtype=np.float_,order='F')
    bspl._colloc(x_1,t_1,k,banded_storage,offset=0)

    interp_matrix = np.zeros(shape=(len(x)-1,len(x)-1))
    bsplines=[]
    for i in [t[i:i+k+2] for i in range(len(x)-1)]:
        bsplines.append(interpolate.BSpline.basis_element(i, False))

    for i in range(len(x)-1):
        for j in range(len(x)-1):
            if not math.isnan(bsplines[j](x[i])):
                interp_matrix[i][j]=bsplines[j](x[i])

    y = np.asarray(y[:-1])
    c = np.linalg.solve(interp_matrix,y)
    c=np.concatenate((c ,c[:k]))

    return interpolate.BSpline(t,c,k)







