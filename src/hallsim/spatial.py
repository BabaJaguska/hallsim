import jax.numpy as jnp
from jax import lax
import matplotlib.pyplot as plt
import numpy as np

LAPLACIAN_2D = jnp.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=jnp.float32)


def diffuse(u, diff_coeff, dt=1.0, dx=1.0):
    """Perform one time step of diffusion on a 2D grid using finite differences.

    Args:
        u: 2D array representing the scalar field to diffuse.
        diff_coeff: Diffusion coefficient.
        dt: Time step size.
        dx: Spatial grid spacing.

    Returns:
        Updated 2D array after diffusion step.
    """
    # add dummy axes for convolution
    u4d = u[jnp.newaxis, :, :, jnp.newaxis]
    laplacian_kernel = LAPLACIAN_2D[:, :, jnp.newaxis, jnp.newaxis] / (dx * dx)

    laplacian_u = lax.conv_general_dilated(
        u4d,
        laplacian_kernel,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )[
        :, :, :, 0
    ]  # remove dummy axes

    u_new = u + diff_coeff * laplacian_u * dt
    u_new = jnp.clip(u_new, 0.0, None)  # Ensure non-negativity
    u_new = u_new.squeeze()
    return u_new


def plot_test(u, diff_coeff, dt, dx, num_steps):
    # Simple test case

    n_cols = 10
    n_rows = num_steps // n_cols + 1
    plt.figure(figsize=(12, 12))
    # fix the color bar limits
    max_val = jnp.max(np.log1p(u))  # maximum value for color bar scaling
    for i in range(num_steps):
        plt.subplot(n_rows, n_cols, i + 1)
        log_u = np.log1p(u)
        plt.imshow(log_u, cmap="hot", interpolation="nearest")
        plt.title(f"Step {i}", fontsize=6)
        plt.axis("off")
        plt.clim(0, max_val)
        u = diffuse(u, diff_coeff, dt, dx)

    plt.tight_layout()
    plt.show()


def video_test(u, diff_coeff, dt, dx, num_steps):
    import matplotlib.animation as animation

    fig, ax = plt.subplots()
    max_val = jnp.max(np.log1p(u))  # maximum value for color bar scaling
    im = ax.imshow(
        np.log1p(u), cmap="hot", interpolation="bilinear", vmin=0, vmax=max_val
    )
    plt.axis("off")

    def update(frame):
        nonlocal u
        u = diffuse(u, diff_coeff, dt, dx)
        im.set_array(np.log1p(u))
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=num_steps, blit=True)
    # save
    ani.save("diffusion_simulation.mp4", writer="ffmpeg", fps=3)
    # plt.show()


if __name__ == "__main__":

    # Parameters
    grid_size = 20
    u = jnp.zeros((grid_size, grid_size), dtype=jnp.float32)
    u = u.at[grid_size // 2, grid_size // 2].set(
        100.0
    )  # Initial condition: point source
    u = u.at[grid_size // 4, grid_size // 3].set(50.0)
    u = u.at[3 * grid_size // 4, grid_size // 8].set(10.0)

    diff_coeff = 0.17
    dt = 1.0
    dx = 1.0

    num_steps = 20

    # plot_test(u, diff_coeff, dt, dx, num_steps)
    video_test(u, diff_coeff, dt, dx, num_steps)
