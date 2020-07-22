import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from py_project.simplified_model import SimplifiedModel
from py_project.weighted_model import WeightedModel
from py_project.psychoactive_model import PsychoactiveModel
from py_project.potential_decrease_model import PotentialDecreaseModel

# Fixing random state for reproducibility
np.random.seed(19680801)

MIN_SIZE = 50
FRAMES_PER_UPDATE = 5

model = PotentialDecreaseModel(200, 0.1, 0.95, 0, 0.05)
model.start_syst()

# Create new Figure and an Axes which fills it.
fig = plt.figure(figsize=(7, 7))
ax = fig.subplots()
ax.set_xlim(0, 1)
ax.set_xticks([])
ax.set_ylim(0, 1)
ax.set_yticks([])

color = ['red' if model.syst_state[i] == 1 else
         ('green' if model.syst_potential[i] >= 0 else 'blue') for i in range(model.N)]
size = [abs(x) + MIN_SIZE for x in model.syst_potential]

# Initialize the raindrops in random positions and with
# random growth rates.
position = np.random.uniform(0, 1, (model.N, 2))

# Construct the scatter which we will update during animation
# as the raindrops develop.
scat = ax.scatter(position[:, 0], position[:, 1],
                  s=size, lw=0.5, c=color, edgecolors=color)


def update(frame_number):
    if model.all_neurones_rest():
        model.start_syst_1()
    if model.non_transmittable():
        model.start_syst()
    else:
        model.update_system_one_step()
    color = ['red' if model.syst_state[i] == 1 else
             ('green' if model.syst_potential[i] >= 0 else 'blue') for i in range(model.N)]
    size = [abs(x) + MIN_SIZE for x in model.syst_potential]
    scat.set_sizes(size)
    scat.set_edgecolors(color)
    scat.set_color(color)
    return scat,


# # Construct the animation, using the update function as the animation director.
animation = FuncAnimation(fig, update, interval=1000, blit=True)
plt.show()
