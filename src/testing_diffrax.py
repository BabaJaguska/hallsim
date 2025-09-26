import jax
import jax.numpy as jnp
import diffrax as dfx
from matplotlib import pyplot as plt

print(f"Using device: {jax.default_backend()}")


# simulate population rise  dP/dt = r*P*(1 - P/K)


def population_growth(_t, y, args):
    r, K = args
    return r * y * (1 - y / K)


# Initial population
P0 = jnp.array([10.0])
# Growth rate
r = 0.1
# Carrying capacity
K = 100.0

# Time span for the simulation
t0 = 0.0
t1 = 100.0
# Time step
dt = 1.0
# Solver
solver = dfx.Dopri5()
# Arguments for the ODE function
args = (r, K)
# Create the differential equation problem
term = dfx.ODETerm(population_growth)
saveat = dfx.SaveAt(ts=jnp.arange(t0, t1 + dt, dt))
stepsize_controller = dfx.PIDController(rtol=1e-5, atol=1e-5)
# Create the solver instance
solution = dfx.diffeqsolve(
    term,
    solver,
    t0,
    t1,
    dt0=dt,
    y0=P0,
    args=args,
    saveat=saveat,
    stepsize_controller=stepsize_controller,
)

# Print the results
for t, P in zip(solution.ts, solution.ys):
    print(f"Time: {t:.1f}, Population: {P[0]:.2f}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(solution.ts, solution.ys)
plt.xlabel("Time")
plt.ylabel("Population")
plt.title("Population Growth Over Time")
plt.grid()
plt.show()
