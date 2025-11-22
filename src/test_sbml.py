import sbmltoodejax
from sbmltoodejax.utils import load_biomodel
from matplotlib import pyplot as plt
import diffrax
import jax.numpy as jnp


print("SBML to ODE JAX Test")
print("Version: ", sbmltoodejax.__version__)

# Load a sample SBML model
# load and simulate model
model, y0, w0, c = load_biomodel(10)
print("Loaded model. Params: ")
print("y0 shape: ", y0.shape)
print("w0 shape: ", w0.shape)
print("c shape: ", c.shape)
print("Model deltaT: ", model.deltaT)
print("Model state variables: ", model.modelstepfunc.y_indexes)


# list methods and attributes of the model
print("Model methods and attributes: ")
print(dir(model))


n_secs = 150 * 60
n_steps = int(n_secs / model.deltaT)
print("Inspecting model inside modelstepfunc: ")
print(dir(model.modelstepfunc))
print("Model ratefunc: ")
print(model.modelstepfunc.ratefunc)  # <-- RHS
print("assignmentfunc: ")
print(model.modelstepfunc.assignmentfunc)
print("C indexes: ")
print(model.modelstepfunc.c_indexes)
print("W indexes: ")
print(model.modelstepfunc.w_indexes)

ys, ws, ts = model(n_steps)

# plot time course simulation as in original paper
y_indexes = model.modelstepfunc.y_indexes  # <-- LHS
plt.figure(figsize=(6, 4))
plt.plot(ts / 60, ys[y_indexes["MAPK"]], color="lawngreen", label="MAPK")
plt.plot(ts / 60, ys[y_indexes["MAPK_PP"]], color="blue", label="MAPK-PP")
plt.plot(ts / 60, ys[y_indexes["MKKK"]], color="orange", label="MKKK")
plt.xlim([0, 150])
plt.ylim([0, 300])
plt.xlabel("Reaction time (mins)")
plt.ylabel("Concentration")
plt.title("SBMLtoODEJAX Simulation")
plt.legend()
plt.show()


# implement own integrator using diffrax
def rhs(t, y, args):
    model, w, c = args
    dydt = model.modelstepfunc.ratefunc(y, t, w, c)
    return dydt


solver = diffrax.Dopri5()
initial_state = ys[:, 0]
solution = diffrax.diffeqsolve(
    diffrax.ODETerm(rhs),
    solver,
    t0=0.0,
    t1=n_secs,
    dt0=model.deltaT,
    y0=initial_state,
    args=(model, w0, c),
    saveat=diffrax.SaveAt(ts=jnp.linspace(0, n_secs, n_steps)),
    max_steps=1000000,
)
ys_diffrax = solution.ys
# plot time course simulation as in original paper
plt.figure(figsize=(6, 4))
plt.plot(
    ts / 60, ys_diffrax[:, y_indexes["MAPK"]], color="lawngreen", label="MAPK"
)
plt.plot(
    ts / 60, ys_diffrax[:, y_indexes["MAPK_PP"]], color="blue", label="MAPK-PP"
)
plt.plot(ts / 60, ys_diffrax[:, y_indexes["MKK"]], color="orange", label="MKK")
plt.plot(ts / 60, ys_diffrax[:, y_indexes["MKKK"]], color="red", label="MKKK")
plt.plot(
    ts / 60, ys_diffrax[:, y_indexes["MKKK_P"]], color="purple", label="MKKK-P"
)

plt.xlim([0, 150])
plt.ylim([0, 300])
plt.xlabel("Reaction time (mins)")
plt.ylabel("Concentration")
plt.title("Diffrax Simulation")
plt.legend()
plt.show()
