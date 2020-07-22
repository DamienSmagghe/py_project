import numpy as np
from py_project.weighted_model import WeightedModel



class PsychoactiveModel(WeightedModel):
    """In our third model, we extend our last WeightedModel by introducing a new "player" in the game: psychoactive substance.
    A psychoactive substance can have different effects on the neural network based on its types that we can group
    into 3 categories:\n
        The stimulants (cafeine, nicotine, cocaine) have advocating effect that create a hyperactive state in the network.
    In other words, these molecules can push the transmission between neurons, therefore a receptor under effect can
    receive more potential from another neuron than it supposes to.
        The depressors (alcohol, morphine, opium) decrease the global activity of the brain. They have a reverse effect
    to the stimulant, meaning that they block the receptors under effect from completely receive the potentials that is
    sent to it.
        The pertubators, which are quite a combination of both the stimulants and the depressors, have a stimulative effect
    on some and a blocking effect on others.\n
    To be concrete, we will take into account the effect of the stimulants and the depressors and discard the pertubators
    in our model.

    We define a new parameter for the model: the alcohol concentration (aka "ca") which will be the determining factor
    of how many neurons in the network are affected and for how much. ca will be between -1 and 1. 0 means no effect,
    1 means stimulate x2, and -1 means block completely from receiving potential

    Also, the weights (or connection) matrix will be updated in function of this new parameter. Neurons that are under
    effect of a psychoactive will receive more or less (ca * 100)% potentials. This has a side effect on the way that
    weights work in our last model. In this model, the sum of weights in each column (not couting the weight of a neuron
    on itself that is 1) will not equal 1.
    """


    def __init__(self, N, beta, gamma, ca):
        super().__init__(N, beta, gamma)
        self.ca = ca
        self.init_system_links_ca()

    def __get_ca(self):
        return self._ca

    def __set_ca(self, ca):
        if -1 <= ca <= 1:
            self._ca = ca
        else:
            self._ca = 0

    ca = property(__get_ca, __set_ca)

    def init_system_links_ca(self):
        nb_affected = round(self.N * self.ca)
        affected = []
        while len(affected) < nb_affected:
            k = np.random.randint(0, self.N)
            if k not in affected:
                affected.append(k)
        for i in affected:
            for j in range(self.N):
                if i != j:
                    self.syst_links[i][j] *= 1+self.ca





