import sbmltoodejax
from sbmltoodejax.utils import load_biomodel
from matplotlib import pyplot as plt


print("SBML to ODE JAX Test")
print("Version: ", sbmltoodejax.__version__)

# Load a sample SBML model
# load and simulate model
model, _, _, _ = load_biomodel(10)
n_secs = 150 * 60
n_steps = int(n_secs / model.deltaT)
ys, ws, ts = model(n_steps)

print(model)
# plot time course simulation as in original paper
y_indexes = model.modelstepfunc.y_indexes
plt.figure(figsize=(6, 4))
plt.plot(ts / 60, ys[y_indexes["MAPK"]], color="lawngreen", label="MAPK")
plt.plot(ts / 60, ys[y_indexes["MAPK_PP"]], color="blue", label="MAPK-PP")
plt.plot(ts / 60, ys[y_indexes["MKKK"]], color="orange", label="MKKK")
plt.xlim([0, 150])
plt.ylim([0, 300])
plt.xlabel("Reaction time (mins)")
plt.ylabel("Concentration")
plt.legend()
plt.show()
