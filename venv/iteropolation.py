import scipy.interpolate._bsplines as interpolate
import numpy as np
import math
from scipy import linalg


def make_knots(points, rightBorder ,k):
    if len(points)<k:
        raise ValueError('at least %d point',k+1)
    if(len(points)==k+1):
        t = np.asarray([points[0],points[-1]+(points[0]+points[-1])/(k+1)])
    else:
        t=(np.asarray(points[:len(points)-k-1],float)+np.asarray(points[1:len(points)-k],float))/2
        t= np.concatenate((np.asarray([points[0]]),t,np.asarray([points[-1]+(rightBorder-points[-1])/3])))
    t=np.concatenate((t,np.linspace(t[-1],rightBorder,k+1)[1:]))

    #making knots outside of base interval

    per = t[-1]-t[0]
    t_1=np.asarray([x-per for x in t[-k-1:-1]])
    t_2=np.asarray([x+per for x in t[1:k+1]])

    return np.concatenate((t_1,t,t_2))


def make_knots3(points):
    per = points[-1]-points[0]
    points=np.asarray(points, dtype=float)
    t_1 = np.asarray([x-per for x in points[-4:]])
    t_2 = np.asarray([x + per for x in points[0:4]])
    t_midle = points[2:-3]
    np.append(t_midle,points[-2])
    return np.concatenate((t_1,t_midle,np.linspace(points[-2],points[-1],3,False),t_2))



def make_periodic_spline(x,y,k):
    #t=make_knots(x[:-1],x[-1],k)
    t = make_knots3(x)
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

    return interpolate.BSpline.construct_fast(t,c,k)







