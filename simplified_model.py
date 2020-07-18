import numpy as np
import copy
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from copy import deepcopy

# DEFINE GLOBAL PARAMETERS OF THE MODEL
N = 25  # number of neurons in the network
threshold = 60  # the threshold of activation of neuron, in mV, used in functions
Vmax = 120  # the potential of neuron when pass the seuil
Vmin = -10  # the min potential of neuron
nb_steps = 500  # number of simulation
beta = 0.6  # the lost coefficient when transmitting potential from a neuron to another
gamma = 0.9  # the coefficient of leaking potential


class SimplifiedModel:
    threshold = 60  # the threshold of activation of neuron, in mV, used in functions
    Vmax = 120  # the potential of neuron when pass the seuil
    Vmin = -10  # the min potential of neuron

    def __init__(self, N, beta, gamma):
        self.N = N
        self.beta = beta
        self.gamma = gamma
        self.syst_links = self.init_system_links()
        self.syst_state = self.init_syst_state()
        self.syst_potential = self.init_syst_potential()

    def __set_N(self, N):
        if not isinstance(N, int):
            raise AttributeError
        self._N = N

    def __get_N(self):
        return self._N

    N = property(__get_N, __set_N)

    def __set_beta(self, beta):
        if not isinstance(beta, float):
            raise AttributeError
        self._beta = beta if 0.0 <= beta <= 1.0 else 0.5

    def __get_beta(self):
        return self._beta

    beta = property(__get_beta, __set_beta)

    def __set_gamma(self, gamma):
        if not isinstance(gamma, float):
            raise AttributeError
        self._gamma = gamma if 0.0 <= gamma <= 1.0 else 0.9

    def __get_gamma(self):
        return self._gamma

    gamma = property(__get_gamma, __set_gamma)

    def __str__(self):
        return f"""The neuron network has {N} neurones with \
               neurons' leakage coefficient of {gamma} and \
               transmission lost coefficient of {beta}."""

    def __repr__(self):
        return """---In the first implemented model of a neural network,
                we try to keep it as simple as possible.---\n 
                A neuron's functionning capability is controlled by its potential 
                stored within. Its potential can be positive or negative and 
                measured by mV (mili-voltage). A neuron can emit and receive 
                potential to and from other neurons in the network under certain 
                conditions. At a given moment t, an inactive neuron i of potential 
                V(i, t) receives a weighted sum of potentials from all other neurons 
                that connect to its dendrites (entrances) and add that sum to its 
                current potential, which equals to the neuron's potential at the 
                moment (t+1). \n
                If the potential surpasses a certain threshold, it will 
                immediately increase to a maximum potential called Vmax. At this moment, 
                the neuron i becomes "overcharged" and switches to "active" mode that 
                allows it to emit its potential Vmax to all neurons that connect to its synapse 
                (exit). We define an activate function for a neuron: 
                f(x) = Vmax if x >= threshold else x. In the first model, a neuron can 
                only keep its activated state in one step of simulation. After that, its 
                potential will fall back to 0. We keep track of the state of a neuron 
                by a variable d(i) where d(i, t) = 1 if neuron i is active at step t 
                and =0 otherwise.\n
                However, the receivers will not receive a potential equal to Vmax. 
                If a neuron can transmit Vmax, a postsynaptic neuron can only receive a 
                potential equal Beta * Vmax where Beta is the lost coefficient during the 
                transmission. \n
                After the depolarisation, its potential gradually falls down 
                to its rest value Vrest while the neurone itself becomes inefficient to the 
                reception of new messages for a period a millisecond. We will consider this
                period a step of simulation. In a step of the simulation, the potentials 
                of all the neurons in the network must be updated at the same time from 
                their values of the preceding step. \n
                We must take into account that even in the case that a neuron does not 
                send nor receive signals in a step, its potential will not be totally 
                preserved to the next step but be leaked. To simplify the decrease of potential,
                we define a coefficient gamma as a decline linear factor to measure 
                this leakage. In a step of a millisecond, a neuron i leaks 
                (gamma * V(i, t)) mV. Biologically, k is between the interval [0.9, 0.95].\n
                In bref, the potential of neuron i at step t+1 is equal to the sum of its
                "leftover" potential from the preceding step and the weighted sum of potential
                of all neuron that can send signals to it.\n
                V(i, t+1) = f(gamma * (1 - d(i, t)) * V(i,t) + beta * sum(d(k, t) * V(k, t)))
                where k is of all neurons that transmit to neuron i."""

    def init_system_links(self):
        """Create a matrix of 2 dimensions (NxN) which shows the connections between
         neurons in the system. Initiated randomly and stay the same throughout the
         simulation. (a transpose matrix of the regular matrix)
        syst_links[i][j] = 1: j connects and can send signal to i, not in reverse
        syst_links[i][j] = 0: j doesnt connect to i
        syst_links[i][i] = gamma/beta where gamma is the factor of leaking potential
        of the neuron i."""
        return [[random.choice([0.0, 1.0]) if j != i else gamma / beta for j in range(N)] for i in range(N)]

    def init_syst_potential(self):
        """Create a matrix of 2D that represents the potential of each neuron at
         moment t of the simulation.
        syst_potentiel[i][j] = 0 for every i != j;
        syst_potentiel[i][i] = value of potential of neuron i
        The matrix will be initiated with all zeros."""
        return [[0.0 for j in range(self.N)] for i in range(self.N)]

    def init_syst_state(self):
        """Create a matrix of size (N,) which shows the state of activation of a neuron
        at the moment t of the simulation.
        Initiated with all zeros.
        syst_act[i][0] = 0: neuron i isn't activated, it can't send signal to others
        syst_act[i][0] = 1: neuron i is activated and can send signal to others
        When the potential of a neuron i passes the threshold, syst_state[i][0] = 1 at the next step
        After release all its potential, syst_state[i][0] = 0 at the next step"""
        return [0 for i in range(self.N)]

    def matrix_Ni(self, i):
        """Create a matrix which helps keeping the potential of
        each neuron of the previous step to the next step.
        Size: (N,)
        Only the i of the neuron in current execution is set to 1.
        All the others elements are 0."""
        return [1 if _i == i else 0 for _i in range(N)]

    def func_act(self, val_poten):
        """float => float
        If the potential of a neuron is superior than the threshold, it's activated.
        When a neuron is activated, its potential increase immediately to Vmax"""
        return val_poten if val_poten < threshold else Vmax

    def start_syst(self):
        """Send in the information in form electric ranged between 0 and Vmax (mV)
         to kick off the system."""
        for i in range(N):
            self.syst_potential[i][i] = self.func_act(self.syst_potential[i][i] + random.uniform(0.0, Vmax))
            self.syst_state[i] = 1 if self.syst_potential[i][i] >= threshold else 0

    def update_system_one_step(self):
        """Calculate the potentials of all the neurons at the time t+1 and also update
        theirs state at time t+1 (activated or not).
        All neurons will be update simultaneously.
        Update the potentials and their states of the whole system in form matrix."""
        new_potential = deepcopy(self.syst_potential)
        new_state = deepcopy(self.syst_state)
        for i in range(N):

            var = np.dot(syst_links[i], np.dot(syst_potentiel, syst_state + matrix_Ni(i, syst_state[i][0])))[0]
            var = func_act(var)
            if var == beta * Vmax:
                new_syst_state[i][0] = 1
            else:
                new_syst_state[i][0] = 0

            new_syst_potentiel[i][i] = var

        return (new_syst_potentiel, new_syst_state)



if __name__ == '__main__':
    model = SimplifiedModel(N, gamma, beta)
    print(model.__repr__())
