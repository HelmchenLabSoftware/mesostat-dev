import numpy as np
from sklearn.linear_model import Ridge
from patsy import cr


# Get the approximation of y, fitted using x
def polyfit_transform(x, y, ord=1):
    param = np.polyfit(x, y, ord)
    poly = np.poly1d(param)
    return poly(x)


def natural_cubic_spline_fit_reg(x, y, dof=5, alpha=0.01):
    basis = cr(x, df=dof, constraints="center")
    model = Ridge(alpha=alpha).fit(basis, y)
    return model.predict(basis)