import numpy as np

'''
Implementation of tripartite Partial Information Decomposition
(PID) for univariate continuous Gaussian variables, based on

* Barrett2015   https://doi.org/10.1103/PhysRevE.91.052802
* KayInce2018   https://doi.org/10.3390/e20040240
'''


# Get off-diagonal entries of the covariance matrix and its determinant
def _get_test_cov(x,y,z):
    zscore = lambda x: (x - np.mean(x)) / np.std(x, ddof=1)
    data2D = np.array([zscore(x), zscore(y), zscore(z)])
    cov = np.cov(data2D)

    # print(cov)
    
    # Ensure diagonal are ones
    np.testing.assert_almost_equal(cov[0,0], 1)
    np.testing.assert_almost_equal(cov[1,1], 1)
    np.testing.assert_almost_equal(cov[2,2], 1)
    
    # Ensure determinant is positive
    detCov = np.linalg.det(cov)
    assert detCov > 0
    
    # Extract off-diagonal entries
    p = cov[0, 1]
    q = cov[0, 2]
    r = cov[1, 2]
    p2 = p**2
    q2 = q**2
    r2 = r**2
    return p2,q2,r2,detCov


# I(X:Z)
def _mi_x_z_gaussian(p2,q2,r2,detCov):
    return np.log(1/(1-q2))/2


# I(Y:Z)
def _mi_y_z_gaussian(p2,q2,r2,detCov):
    return np.log(1/(1-r2))/2


# I(XY:Z)
def _mi_xy_z_gaussian(p2,q2,r2,detCov):
    return np.log((1-p2)/detCov)/2


# I(X:Z|Y)
def cmi_x_z_y_gaussian(p2,q2,r2,detCov):
    return np.log((1-p2)*(1-r2)/detCov)/2


# I(Y:Z|X)
def _cmi_y_z_x_gaussian(p2,q2,r2,detCov):
    return np.log((1-p2)*(1-q2)/detCov)/2


# I(X:Y:Z) - co-information, aka interaction information, aka multivariate mutual information
def _deltaI(p2,q2,r2,detCov):
    return np.log((1-p2)*(1-q2)*(1-r2)/detCov)/2


# Implementation of Barrett2015 PID (aka MMI PID)
def pid_barrett_gaussian(x,y,z):
    # Get off-diagonal covariance entries and their squares
    p2,q2,r2,detCov = _get_test_cov(x,y,z)

    # Compute preliminary mutual informations
    miX_Z = _mi_x_z_gaussian(p2,q2,r2,detCov)
    miY_Z = _mi_y_z_gaussian(p2,q2,r2,detCov)
    miXY_Z = _mi_xy_z_gaussian(p2,q2,r2,detCov)
    
    # Compute PID
    red = min(miX_Z, miY_Z)
    unqXZ = miX_Z - red
    unqYZ = miY_Z - red
    syn = miXY_Z - unqXZ - unqYZ - red
    return np.array([unqXZ, unqYZ, red, syn])


# Implementation of KayInce2018 PID (aka DEP PID)
def pid_kayince_gaussian(x,y,z):
    # Get off-diagonal covariance entries and their squares
    p2,q2,r2,detCov = _get_test_cov(x,y,z)

    # Compute preliminary mutual informations
    miX_Z = _mi_x_z_gaussian(p2,q2,r2,detCov)
    miY_Z = _mi_y_z_gaussian(p2,q2,r2,detCov)
    miBR = np.log((1 - q2*r2)/(1-q2)/(1-r2))/2
    miXY_Z = _mi_xy_z_gaussian(p2,q2,r2,detCov)
    
    b = miX_Z
    d = miX_Z
    i = miBR - miY_Z
    k = miXY_Z - miY_Z
    
    c = miY_Z
    f = miY_Z
    h = miBR - miX_Z
    j = miXY_Z - miX_Z
    
    # Compute PID both ways
    unq1XZ = min(b,d,i,k)
    red1 = miX_Z - unq1XZ
    unq1YZ = miY_Z - red1
    syn1 = miXY_Z - unq1XZ - unq1YZ - red1
    
    unq2YZ = min(c,f,h,j)
    red2 = miY_Z - unq2YZ
    unq2XZ = miX_Z - red2
    syn2 = miXY_Z - unq2XZ - unq2YZ - red2
    
    #return average over both estimates for stability
    return np.mean([
        [unq1XZ, unq1YZ, red1, syn1],
        [unq2XZ, unq2YZ, red2, syn2]
    ], axis=0)
