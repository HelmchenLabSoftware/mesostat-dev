from mesostat.utils.signals.fit import natural_cubic_spline_fit_reg


import numpy as np
import matplotlib.pyplot as plt

n_obs = 600
np.random.seed(0)
x = np.linspace(-3, 3, n_obs)
y = 1 / (x ** 2 + 1) * np.cos(np.pi * x) + np.random.normal(0, 0.2, size=n_obs)


plt.scatter(x, y, s=4, color="tab:blue")

for dof in (5, 7, 10, 25, 100):
    yhat = natural_cubic_spline_fit_reg(x, y, dof, alpha=0.1)
    plt.plot(x, yhat, label=f"dof={dof}")

plt.legend()
plt.title(f"Natural cubic spline with varying degrees of freedom")
plt.show()
