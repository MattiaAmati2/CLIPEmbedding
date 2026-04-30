import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def update_posterior(mu_prior, cov_prior, cov_obs, points):
    n = len(points)
    if n == 0:
        return mu_prior, cov_prior

    x_bar = np.mean(points, axis=0)

    inv_cov_0 = np.linalg.inv(cov_prior)
    inv_cov_obs = np.linalg.inv(cov_obs)

    cov_n = np.linalg.inv(inv_cov_0 + (n * inv_cov_obs))
    mu_n = cov_n @ (inv_cov_0 @ mu_prior + n * (inv_cov_obs @ x_bar))

    return mu_n, cov_n


fig, ax = plt.subplots(figsize=(7, 7))
plt.subplots_adjust(bottom=0.25)

# The Prior "Text" Anchor is fixed at (0,0)
mu_0 = np.array([0.0, 0.0])
observation_points = []


def draw_gaussian_contour(mu, cov, color, label):
    x, y = np.mgrid[-5:5:.05, -5:5:.05]
    pos = np.dstack((x, y))

    diff = pos - mu
    inv_cov = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)

    # Einstein summation to quickly compute (x-mu)^T * Sigma^-1 * (x-mu) for the whole grid
    exponent = np.einsum('...i,ij,...j->...', diff, inv_cov, diff)
    pdf_grid = np.exp(-0.5 * exponent) / (2 * np.pi * np.sqrt(det_cov))

    peak_value = 1.0 / (2 * np.pi * np.sqrt(det_cov))
    ax.contour(x, y, pdf_grid, levels=[peak_value * np.exp(-0.5)], colors=color, alpha=0.8)
    ax.plot(mu[0], mu[1], marker='X', color=color, markersize=10, label=label, linestyle='None')


def redraw_plot():
    ax.clear()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_title("Interactive Bayesian Update (Click to add images)")
    ax.grid(True, linestyle='--', alpha=0.5)

    base_prior_cov = np.array([
        [0.2, 0.0],
        [0.0, 5.0]
    ])
    cov_0 = base_prior_cov * prior_slider.val
    draw_gaussian_contour(mu_0, cov_0, color='blue', label='Prior (Text)')

    reg_lambda = obs_slider.val

    if len(observation_points) >= 2:
        pts_array = np.array(observation_points)
        empirical_cov = np.cov(pts_array, rowvar=False)
        cov_obs = empirical_cov + (np.eye(2) * reg_lambda)
    else:
        cov_obs = np.eye(2) * reg_lambda

    if observation_points:
        pts = np.array(observation_points)
        ax.scatter(pts[:, 0], pts[:, 1], color='black', marker='o', label='Images (Obs)')
        mu_n, cov_n = update_posterior(mu_0, cov_0, cov_obs, pts)
        draw_gaussian_contour(mu_n, cov_n, color='red', label='Posterior (New Anchor)')

    ax.legend(loc='upper right')
    fig.canvas.draw_idle()


def on_click(event):
    if event.inaxes == ax:
        observation_points.append([event.xdata, event.ydata])
        redraw_plot()


fig.canvas.mpl_connect('button_press_event', on_click)

# Prior Variance Slider
ax_prior = plt.axes((0.15, 0.1, 0.65, 0.03))
prior_slider = Slider(ax_prior, 'Prior Var', 0.1, 5.0, valinit=1.0)
prior_slider.on_changed(lambda val: redraw_plot())

# Observation Variance Slider (Regularization Factor)
ax_obs = plt.axes((0.15, 0.05, 0.65, 0.03))
obs_slider = Slider(ax_obs, 'Obs Var', 0.1, 5.0, valinit=0.5)
obs_slider.on_changed(lambda val: redraw_plot())

redraw_plot()
plt.show(block=True)