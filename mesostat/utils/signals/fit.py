import numpy as np
from sklearn.linear_model import Ridge
from patsy import cr


# Get the approximation of y, fitted using x
def polyfit_transform(x, y, ord=1):
    param = np.polyfit(x, y, ord)
    poly = np.poly1d(param)
    return poly(x)


def natural_cubic_spline_fit_reg(x, y, dof=5, alpha=0.01):
    if not np.any(np.isnan([x, y])):
        basis = cr(x, df=dof, constraints="center")
        model = Ridge(alpha=alpha).fit(basis, y)
        return model.predict(basis)
    else:
        # Insurance against NAN
        # Solution: Fit only non-nan pairs, leave rest as NAN

        print("Warning, have NAN")

        iNanX = np.isnan(x)
        iNanY = np.isnan(y)
        iNan = np.logical_or(iNanX, iNanY)

        yRez = np.copy(y)
        yRez[iNan] = np.nan
        yRez[~iNan] = natural_cubic_spline_fit_reg(x[~iNan], y[~iNan], dof=dof, alpha=alpha)
        return yRez
