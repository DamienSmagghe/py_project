import numpy as np
from py_project.simplified_model import SimplifiedModel


class WeightedModel(SimplifiedModel):
    """In the first implemented model, we consider that the coefficients
    present in matrix of connection are equal 1 to any link between 2 neurons.
    However, in the real case, a neuron cannot send its full potential to each
    of every neuron that connect to its synapse. Instead, its potential will be
    "shared" between the receivers. For example, if a neuron A can send signals
    to B, C and D then P_A = P_B + P_C + P_D.

    We fix this issue by replacing the connection between a neuron i and neuron j
    by a weight w(i, j) that represents the weighted potential sent from j to i.
    For a certain j, sum of W(i, j) with i in [1;N] and i != j equals to 1.
    In taking into account the parameter W, the potential that neuron i can
    receive from neuron j is beta * w(i, j) * Vmax.

    Another notion that we have not considered in the first implementation is that
    the signal sent by a neuron can have inhibiting or exciting effects on the receivers.
    To correctly approach this, the sign of each weight W(i, j) will be assigned
    randomly in sort that the total absolute value of weights still equals to 1.

    In reality, a neuron can also have a negative potential. In fact, after
    depolarising all of its potential, a neuron will not return to 0 but
    fall down to a negative potential Vrest. To simplify, we will determine a value
    Vmin that the potential of a neuron cannot be less than this value. In another word,
    if a neuron's potential falls below this threshold, it will be consider to
    equal to Vmin. The Vrest will be in use later in an updated model.
    """
    Vmin = -30.

    def __init__(self, N, beta, gamma):
        super().__init__(N, beta, gamma)
        self.init_system_links_weighted()

    def init_system_links_weighted(self):
        """Create a matrix of 2 dimensions (NxN) which shows the connections between
        neurons in the system. Initiated randomly and stay the same throughout the
        simulation.

        syst_links[i][j] != 0: j connects and can send signal to i, not in reverse

        syst_links[i][j] = 0: j doesnt connect to i

        syst_links[i][i] = gamma/beta where gamma is the factor of leaking potential
        of the neuron i.

        sum(syst_links[i in range(N), i != j][j] = 1"""

        def count_connections(connections):
            """List[List[int]] -> List[int]
            Count the number of 1 in each column, except for the case
            where no row = no column
            connections has #cols = #rows"""
            size = len(connections)
            res = [0 for i in range(size)]
            for i in range(size):
                for j in range(size):
                    if i != j and connections[i][j] == 1:
                        res[j] += 1
            return res

        def decompose(n):
            """int -> list[float]
            Return a list of random floats whose sum equals to 1."""
            res = np.random.random(n)
            res /= res.sum()
            return res

        count_conn = count_connections(self.syst_links)
        for col in range(self.N):
            L = decompose(count_conn[col])
            k = 0
            for row in range(self.N):
                if row != col and self.syst_links[row][col] == 1:
                    self.syst_links[row][col] = np.random.choice([-1, 1]) * L[k]
                    k += 1

    def func_act(self, val_poten: float):
        """float => float
        If the potential of a neuron is superior than the threshold, it's activated.
        When a neuron is activated, its potential increase immediately to Vmax
        When a neuron's potential is below the lower threshold, it will be set
        to equal to the lower threshold."""
        if val_poten > WeightedModel.threshold:
            return WeightedModel.Vmax
        elif val_poten < WeightedModel.Vmin:
            return WeightedModel.Vmin
        else:
            return val_poten

    def start_syst(self):
        """Send in the information in form electric ranged between Vmin and Vmax (mV)
        to kick off the system."""
        print("Feed potentials to the system.")
        # print("potential before: \n", np.array(self.syst_potential))
        # np.random.seed(19680801)
        added_values = []
        for i in range(self.N):
            x = np.random.randint(WeightedModel.Vmin, WeightedModel.Vmax)
            self.syst_potential[i] = self.func_act(self.syst_potential[i] + x)
            added_values.append(x)
            self.syst_state[i] = 1 if self.syst_potential[i] == WeightedModel.Vmax else 0
        # print("added_values: \n", np.array(added_values))
        # print("potential after: \n", np.array(self.syst_potential))


if __name__ == '__main__':
    N = 20
    gamma = 0.9
    beta = 0.3
    model = WeightedModel(N, gamma, beta)
    i = 0
    print("link=\n", np.array(model.syst_links))
    for p in model.simulation(10):
        i+=1
        state, potential = p
        print(i, np.array(potential), "\n", state, "\n")
