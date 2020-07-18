import numpy as np
import copy
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math


#DEFINE GLOBAL PARAMETERS OF THE MODEL
N = 25      #number of neurons in the network
seuil = 60  #the threshold of activation of neuron, in mV, used in functions
Vmax = 120  #the potential of neuron when pass the seuil
Vmin=-10    #the min potential of neuron
nb_steps = 500 #number of simulation
beta = 0.6  #the lost coefficient when transmitting potential from a neuron to another
gamma = 0.9 #the coefficient of leaking potential

class SimplifiedModel:
    def __init__(self, N, beta, gamma):
        self.N = N
        self.beta = beta
        self.gamma = gamma
        self.syst_links = self.init_system_links()

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
        if not isinstance(gamma, int):
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
        return """---In the first implemented model of a neural network, \
                we try to keep it as simple as possible.---\n \
                A neuron's functionning capability is controlled by its potential \
                stored within. Its potential can be positive or negative and \
                measured by mV (mili-voltage). A neuron can emit and receive \
                potential to and from other neurons in the network under certain \
                conditions. At a given moment t, an inactive neuron i of potential \
                V(i, t) receives a weigted sum of potentials from all other neurons \
                that connect to its dendrites (entrances) and add that sum to its \
                current potential, which .equals to the neuron's potential at the \
                moment (t+1). If the potential surpasses a certain threshold, it will \
                immediately increase to a maximum potential called Vmax. At this moment, \
                the neuron i becomes "overcharged" and switches to "active" mode that \
                allows it to emit its potential to all neurons that connect to its synapse \
                (exit). However, the receivers will not receive a potential equal to Vmax. \
                If a neuron can transmit Vmax, a postsynaptic neuron can only receive a \
                potential equal Beta * Vmax where Beta is the lost coefficient during the \
                transmission. After the depolarisation, its potential gradually falls down \
                to its rest value Vrest while the neurone itself becomes inefficient to the \
                reception of new messages for a period a millisecond. We will consider this \
                period a step of simulation. In a step of the simulation, the potentials \
                of all the neurons in the network must be updated at the same time from \
                their values of the preceding step. We must take into account that even \
                in the case that a neuron does not send nor receive signals in a step, its \
                potential will not be totally preserved to the next step but be leaked. \
                We define a coefficient gamma to measure this leakage. In a step of a \
                millisecond, a neuron i leaks (gamma * V(i, t)) mV. \
                Biologically, k is between the interval [0.9, 0.95].
"""

    def init_system_links(self):
        """Create a matrix of 2 dimensions (NxN) which shows the connections between
         neurons in the system. Initiated randomly and stay the same throughout the
         simulation. (a transpose matrix of the regular matrix)
        syst_links[i][j] = 1: j connects and can send signal to i, not in reverse
        syst_links[i][j] = 0: j doesnt connect to i
        syst_links[i][i] = gamma/beta where gamme is the factor of leaking potential
        of the neuron i."""
        syst_links = np.random.rand(N, N)
        for i in range(N):
            for j in range(N):
                if i != j:
                    syst_links[i][j] = random.choice([0.0, 1.0])
                else:
                    syst_links[i][j] = random.uniform(0.9, 0.95) / self.beta
        return syst_links

