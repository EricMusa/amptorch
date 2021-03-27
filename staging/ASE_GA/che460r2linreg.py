import numpy as np
from sklearn.linear_model import LinearRegression

# reg = LinearRegression().fit(X, y)

headers = [
    "Temp (C)",
    "Catalyst (fr. wt.)",
    "Molar ratio",
    "runtime",
    "equilibrated",
    "Act purity",
    "Pred purity",
    "Purity Residual",
    "Act yield",
    "Pred yield",
    "Yield Residual",
]
t1 = [
    50,
    0.005,
    6,
    30,
    True,
    0.9186061762,
    0.978911,
    -0.06160399036,
    0.7863693108,
    1.131874,
    -0.3052501331,
]
t2 = [
    60,
    0.005,
    4,
    15,
    True,
    0.8646555207,
    0.682143,
    0.2675575659,
    0.9017607021,
    1.089398,
    -0.1722394367,
]
t3 = [
    60,
    0.001,
    4,
    15,
    True,
    0.6253381724,
    0.486723,
    0.2847927309,
    0.4738769699,
    1.0827332,
    -0.5623326505,
]
t4 = [
    60,
    0.001,
    8,
    15,
    False,
    0.6013399878,
    1.1241538,
    -0.4650732064,
    0.3950172651,
    1.2479812,
    -0.6834749873,
]
t5 = [
    60,
    0.005,
    8,
    20,
    True,
    0.9814279724,
    1.286161,
    -0.2369322562,
    1.128590077,
    1.249206,
    -0.09655406948,
]
t6 = [
    60,
    0.002,
    6,
    30,
    False,
    0.7579169838,
    0.8919528,
    -0.3258163574,
    0.6622851636,
    1.1721794,
    -0.6630061361,
]
ts = np.vstack([t1, t2, t3, t4, t5, t6])
inputs = ts[:, :3]
outputs = ts[:, (5)]  # , 8)]

reg = LinearRegression().fit(inputs, outputs)
print(outputs)
print(reg.score(inputs, outputs))
print(reg.coef_)
print(reg.predict([[60, 0.001, 8]]))
print(reg.predict([[60, 0.01, 8]]))


x1_plot = np.linspace(0.0055, 0.0005, 12)
x2_plot = np.linspace(3.5, 8.5, 12)


def func(X2, X3, X1=60):
    return X1 * reg.coef_[0] + X2 * reg.coef_[1] + X3 * reg.coef_[2] + reg.intercept_


X1_plot, X2_plot = np.meshgrid(x1_plot, x2_plot)
Y = func(X1_plot, X2_plot)

import matplotlib.pyplot as plt

# plt.rcParams['figure.dpi'] = 250
plt.rcParams["xtick.labelsize"] = "large"
plt.rcParams["ytick.labelsize"] = "large"
plt.rcParams["axes.labelsize"] = "large"
plt.rcParams["axes.titlesize"] = "x-large"
plt.rcParams["figure.titlesize"] = "xx-large"
# plt.rcParams['xtick.direction'] = 'inout'
# plt.rcParams['ytick.direction'] = 'inout'
#:     out     # direction: {in, out, inout}
# xtick.direction:     out     # direction: {in, out, inout}

plt.xticks([0.001, 0.002, 0.004, 0.005])
plt.yticks([4, 6, 8])
plt.gca().invert_yaxis()
contourfs = plt.contourf(X1_plot, X2_plot, Y, 20)
contours = plt.contour(X1_plot, X2_plot, Y, 20, colors="black")
plt.clabel(contours, inline=True, fontsize=8)
# plt.imshow(Y, extent=[0.0005, 0.0055, 3.5, 8.5], origin='lower',
#            cmap='viridis')

plt.xlabel("Catalyst Amount (wt% NaOMe)")
plt.ylabel("Molar Ratio (MeOH:TAG)")
plt.title("Contour plot of Purity (X1=Cat, X2=MR)")
# plt.colorbar()

annotations = [
    [0.001, 4, "A", "white", 0.1],
    [0.005, 4, "B", "white", 0.1],
    [0.001, 8, "C", "white", 0.1],
    [0.0049, 8, "D", "white", 1],
    [0.00195, 5.8, "E", "white", 0.1],
    [0.0039, 5.8, "RR", "white", 0.1],
]
for x, y, text, color, alpha in annotations:
    t = plt.text(x, y, text, fontsize=20, alpha=alpha)
    t.set_bbox(dict(facecolor=color, alpha=alpha, edgecolor="black"))
plt.tight_layout()
plt.show()
