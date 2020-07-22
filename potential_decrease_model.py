from py_project.psychoactive_model import PsychoactiveModel
from py_project.weighted_model import WeightedModel
import numpy as np
import math
import random
import copy

class PotentialDecreaseModel(PsychoactiveModel):
    """So far until now, we consider that from the moment a neuron's potential reach the threshold,
    increase immediately to Vmax and start sending signals, until it falls back to Vrest (0 in our
    previous model) is 1 millisecond aka one step of our simulation. This is not a bad assumption
    but obviously oversimplify and does not fully cover the evolution of a neuron. In reality, a lot
    happens since the neuron depolarizes to Vmax until it repolarizes to its resting potential and
    even after that in a period we call hyperpolarization.

    In this model, we break this hold period into 5 phases, each last for a different time and
    presents different behavior of a neuron.
        Phase 1 - Start Active phase: This phase last from the moment the neuron's potential reachs
        the threshold, start going up to Vmax until it starts to decrease from Vmax. In this phase, the
        neuron can send signals to others as long as its potential is more than Vmax but cannot receive.
        Thus, it cannot have another depolarization during this phase. This phase is the start of what
        we have been assuming a "complete" circle of evolution in our previous models but now we have
        to correct it.

        Phase 2: This phase last from when the activate potential start to decrease from its peak at Vmax
        until it falls back to the threshold. Like phase 1, the neuron is still active so it cannot
        receive but can send signals to other.

        Phase 3 - Post-Depolarization: This is the phase from the moment an active neuron falls below the
        activate threshold until it's down to the resting potential. The neuron can now receive potential
        from others, thus is capable of depolarize again. However,the maximum potential that a neuron can
        achieve in this phase if it is activated is less than the usual Vmax. This new maximum value of
        potential is determined in function of the moment the neurone is activated during this phase.
        Furthermore, this mechanism works cumulatively until the neuron can successfully process through
        the whole evolution of 5 phases without a depolarization. We introduce a parameter lambda to keep
        track of maximum achievable potential for each neuron of the network.

        Phase 3.5 - Resting phase: The neuron stays at Vrest (< 0) during this phase for a period of time
        that equals to the time the neurons can stay at Vmax.

        Phase 4 - Hyperpolarization: The neuron's potential increases gradually from Vrest to 0 by itself.
        If it receives potentials from others and depolarized in this phase, the Vmax it can attain will
        still be penalty because it has not yet totally complete the cycle of evolution. This phase is quite
        the opposite of what happened in phase 2.

        Phase 0 - Normal phase: A neuron at this phase has gone through the whole cycle of evolution and
        bares no effect of its previous depolarization. It will receive potentials from others and depolarizes
        as usual without penalty. In fact, there is a loophole in this cycle that if a neuron during phase 3
        and 4 receives potentials from others but not enough to push it to the threshold, the neuron will then
        be turned into phase 0 without going through phase 5.

    There are a lot of new parameters that can be taken into account in this model. Simulate a system with
    too many parameters will be hard to keep track and may produce unwanted results. Therefore, we will try
    to fix as many parameters as possible, those that can be roughly estimated in real case through research
    and graphs. Here come the new players in game:
        Vmin_ma: the minimum potential a neuron can have \n
        Vrest: the resting potential, which is negative but not as negative as Vmin_ma \n
        tau_min/tau_max: the minimal time and the maximum time we set for a neuron to go up/down a certain amount
        of potential. These 2 values help control precisely how a neuron decrease its potential in order to
        avoid having a "shock" decrease aka a free fall or a scenario where no matter how many stimulations, the
        neuron cannot reach to the threshold
        tPA: Activation time. This time equals to the sum of phase 1 and 2 and represent the period when
        a neuron can send signal to others. \n
        tPH: Post-Depolarization time. This time equals to phase 3 (not included phase 3.5)  \n
        tH: Hyperpolarization time. This time equals to phase 4 \n
        deltaT: Equals to the time a neuron stays at its peak Vmax and also equals to the time a neuron
        stays resting at Vrest. This is the only new parameter that we can control. But in fact, by fixing
        others parameters and let loose this one, we have already affect a lot the internal interaction of
        the whole system as the calculations of time bases much on this parameter. \n
        phase: a vector of size equal to the number of neuron in the system, which represent the phase
        each neuron is in (run from 0 to 4) \n
        lamb: a vector of size equal to the number of neuron in the system, which represent the percentage
        of Vmax that a neuron can attain at its next depolarization. \n
        time_rest: a vector of size equal to the number of neuron in the system, which counts the steps
        that each neuron of the system has taken since its last depolarization. \n

    Due to the new level of complexity, the old activate function will no longer in use in this model. We
    introduces 5 new activate functions, each corresponds to a specific phase.


    """

    Vrest = -5.
    Vmin_ma = -15.
    tau_min = 10
    tau_max = 20
    tPA = 1.
    tPD = 0.9
    tH = 1.1

    def __init__(self, N, beta, gamma, ca, deltaT):
        super().__init__(N, beta, gamma, ca)
        self.deltaT = deltaT #time step 
        self.phase = self.init_system_phase()
        self.lamb = self.init_system_lambda()
        self.time_rest = self.init_system_rest()

    def init_system_phase(self):
        """Create a vector of size N which keeps tracks of the phase of all the neurons in the system.
        There are 5 phases stated in the document of algorithm."""
        return np.zeros(self.N, dtype=float)

    def init_system_lambda(self):
        """Create a vector of size N which helps keep tracks of the coefficient
        lambda which decides the percentage of Vmax can be attained in the next depolarisation."""
        return np.ones(self.N, dtype=float)

    def init_system_rest(self):
        """Create a vector of size N which count the steps taken by all neurons
        of the system after they were depolarised"""
        return np.zeros(self.N, dtype=float)

    def func_act_0(self, potentiel):
        """float -> float
        The function activate of the system is to modify the potential of each neuron corresponding
        to its current phase = 0 and depending on its sum of reception from others"""
        # V_new: potential of neuron after affected by the activate function
        V_new = 0
        if potentiel > 0:
            if potentiel < PotentialDecreaseModel.threshold:
                V_new = potentiel * math.exp(-1. / 
                                             (PotentialDecreaseModel.tau_min + 
                                              PotentialDecreaseModel.tau_max * potentiel / PotentialDecreaseModel.threshold) * 
                                             math.log(100 * PotentialDecreaseModel.threshold))
            else:
                V_new = PotentialDecreaseModel.threshold
        elif potentiel < 0:
            if potentiel > PotentialDecreaseModel.Vmin_ma:
                V_new = potentiel * math.exp(-1. / 
                                             (PotentialDecreaseModel.tau_min + 
                                              PotentialDecreaseModel.tau_max*potentiel/PotentialDecreaseModel.threshold) * 
                                             math.log(100 * abs(PotentialDecreaseModel.Vmin_ma)))
            else:
                V_new = PotentialDecreaseModel.Vmin_ma
        return V_new

    def func_act_1(self, potentiel, i):
        """float -> float
        The function activate of the system is to modify the potential of each neuron corresponding
        to its current phase = 1 and depending on its sum of reception from others"""
        # V_new: potential of neuron after affected by the activate function
        V_new = potentiel + (2 * self.lamb[i] *
                             (PotentialDecreaseModel.Vmax - PotentialDecreaseModel.threshold) * 
                             self.deltaT / (PotentialDecreaseModel.tPA - self.deltaT))
        # var temporary to stock the value of Vmax of the current depolarisation
        var = self.lamb[i] * (PotentialDecreaseModel.Vmax - PotentialDecreaseModel.threshold) + PotentialDecreaseModel.threshold
        if V_new >= var:
            V_new = var
        return V_new

    def func_act_2(self, potentiel, i):
        """float -> float
        The function activate of the system is to modify the potential of each neuron corresponding
        to its current phase = 2 and depending on its sum of reception from others"""
        # V_new: potential of neuron after affected by the activate function
        V_new = potentiel - (2 * self.lamb[i] * 
                             (PotentialDecreaseModel.Vmax - PotentialDecreaseModel.threshold) * 
                             self.deltaT / (PotentialDecreaseModel.tPA - self.deltaT))
        if V_new <= PotentialDecreaseModel.threshold:
            V_new = PotentialDecreaseModel.threshold
        return V_new

    def func_act_3(self, potentiel,i):
        """float -> float
        The function activate of the system is to modify the potential of each neuron corresponding
        to its current phase = 3 and depending on its sum of reception from others"""
        # V_new: potential of neuron after affected by the activate function
        V_new = potentiel - self.deltaT * \
                (PotentialDecreaseModel.threshold - self.lamb[i] * PotentialDecreaseModel.Vrest) / \
                (math.exp(self.gamma * (1-self.lamb[i])) * PotentialDecreaseModel.tPD)
        if V_new < self.lamb[i] * PotentialDecreaseModel.Vrest:
            V_new = self.lamb[i] * PotentialDecreaseModel.Vrest
        return V_new

    def func_act_4(self, potentiel,i):
        """float -> float
        The function activate of the system is to modify the potential of each neuron corresponding
        to its current phase = 4 and depending on its sum of reception from others"""
        # V_new: potential of neuron after affected by the activate function
        V_new = potentiel - \
                (self.lamb[i] * PotentialDecreaseModel.Vrest * self.deltaT / 
                 (math.exp(self.gamma * (1-self.lamb[i])) * PotentialDecreaseModel.tH))
        if V_new > 0:
            V_new = 0
        return V_new

    def update_lamb(self, i):
        self.lamb[i] = 1 - (self.time_rest[i] /
                            (math.exp(self.gamma * (1 - self.lamb[i])) *
                             (PotentialDecreaseModel.tPD + PotentialDecreaseModel.tH) + self.deltaT))

    def give_time_ar(self, i):
        self.time_rest[i] = math.exp(self.gamma * (1 - self.lamb[i])) * \
                            (PotentialDecreaseModel.tPD + PotentialDecreaseModel.tH) + self.deltaT

    def start_syst(self):
        """Send in the information in form electric ranged between 0 and (PotentialDecreaseModel.threshold + Vmax)/2
        (mV) to kick off the system. we suppose that this function will only be called when there are no transmission
        in between the neurons and all the neurons are at phase 0 (we still wait until the neurones at phase 3 and 4
        rest at 0 and turns to phase 0, meanwhile, others neurons at phase 0 will still decrease as defined)
        Return None as the parametres given to the function is already modified """
        print("System non transmittable. Feed signals")
        for i in range(self.N):
            self.syst_potential[i] = self.func_act_0(self.syst_potential[i] + 
                                                   random.uniform(0, (PotentialDecreaseModel.threshold + PotentialDecreaseModel.Vmax)/2) * 
                                                   random.choice([0,1]))
            if self.syst_potential[i] == PotentialDecreaseModel.threshold:
                self.syst_state[i] = 1
                self.phase[i] = 1

    def start_syst_1(self):
        """Send in the information in form electric ranged between 0 and PotentialDecreaseModel.threshold (mV) to
        kick off the system. we suppose that this function will only be called when there are no transmission in
        between the neurons which means all neurons are at phase {0,3,4}. Indeed, if a neuron in phase 3 or 4 is
        activated after this functions is called, the new Vmax will be calculated, else, we have to set its lambda
        back to 1
        Return None as the parametres given to the function is already modified """
        print("System at rest. Feed signals")
        for i in range(self.N):
            self.syst_potential[i] = self.func_act_0(self.syst_potential[i] + 
                                                   random.uniform(0, PotentialDecreaseModel.threshold) * 
                                                   random.choice([-1,0,1]))
            if self.syst_potential[i] == PotentialDecreaseModel.threshold:
                if self.phase[i] == 3 or self.phase[i] == 4:
                    self.update_lamb(i)
                self.syst_state[i] = 1
                self.phase[i] = 1
            else:
                self.lamb[i] = 1
                self.phase[i] = 0

    def non_transmittable(self):
        """Verify if there is no transmission between neurons and all neurons are at phase 0
        This function is compatible with the function start_syst
        return a bool"""
        res = True
        for i in range(self.N):
            if self.syst_state[i] == 1 or self.phase[i] != 0:
                res = False
                break
        return res

    def all_neurones_rest(self):
        """Verify if all the neurons' potentiels are 0
        return a bool"""
        res = True
        for i in range(self.N):
            if self.syst_potential[i] != 0:
                res = False
                break
        return res

    def update_system_one_step(self):
        """matrix(N,N) ^2 * matrix(N,1) ^5 -> tuple(matrix(N,N), matrix(N,1))
        Calculate the potentials of all the neurons at the time t+1 and also update theirs state at time t+1 (activated or not)
        All neurons will be update simultaneously.
        Return the potentials and their states of the whole system in form matrix."""
        new_syst_potentiel = self.init_syst_potential()
        new_syst_state = copy.deepcopy(self.syst_state)
        var = 0  # variable temporary

        for i in range(self.N):
            if self.syst_state[i] == 0:
                # if a neuron is not in the potential of action, it will receive from others (phase = {0,3,4})
                var = self.beta * np.dot(
                    self.syst_links[i],
                    (
                        self.syst_potential *
                        np.add(
                            self.syst_state,
                            np.dot((-1) ** self.syst_state[i], self.matrix_Ni(i)))
                     + np.dot(-PotentialDecreaseModel.threshold, self.syst_state)))
                # manipulate the time_rest of neuron as it's in phase {0,3,4}
                if self.time_rest[i] > 0:
                    # time_rest will be subtracted every step as long as syst_state[i][0] == 0
                    self.time_rest[i] -= self.deltaT
                else:
                    # the neuron i has waited enough time to reach the Vmax again, time_rest will be set at 0
                    # until it will be reset when the neuron depolarise again
                    self.lamb[i] = 1
                    self.time_rest[i] = 0
            else:
                # else it will not receive transmission from others and behaves as defined ( phase = {1,2})
                var = self.syst_potential[i]
            # var stocks the sum of potential that a neuron has after receiving from others (period of transmission between
            # neurones) and before affected by func_act

            if self.phase[i] == 0:
                new_syst_potentiel[i] = self.func_act_0(var)
                if new_syst_potentiel[i] == PotentialDecreaseModel.threshold:
                    new_syst_state[i] = 1
                    self.phase[i] = 1
                    self.update_lamb(i)

            elif self.phase[i] == 1:
                new_syst_potentiel[i] = self.func_act_1(var, i)
                Vmax_current = self.lamb[i] * (PotentialDecreaseModel.Vmax - PotentialDecreaseModel.threshold) + \
                               PotentialDecreaseModel.threshold
                if new_syst_potentiel[i] == Vmax_current:
                    self.phase[i] = 2

            elif self.phase[i] == 2:
                new_syst_potentiel[i] = self.func_act_2(var, i)
                if new_syst_potentiel[i] == PotentialDecreaseModel.threshold:
                    new_syst_state[i] = 0
                    self.phase[i] = 3
                    # the neurone enters phase 3, we set time_rest to the starting point and start countdown
                    self.give_time_ar(i)

            elif self.phase[i] == 3:
                if var != self.syst_potential[i]:
                    # the neuron in phase 3 receives potential non zero from others so it breaks off from the
                    # phase 3 and return into a neuron of phase 0
                    self.phase[i] = 0
                    new_syst_potentiel[i] = self.func_act_0(var)
                    if new_syst_potentiel[i] == PotentialDecreaseModel.threshold:
                        # if its potential increase directly to pass the PotentialDecreaseModel.threshold, it steps into phase 1 but with new value of
                        # Vmax calculated after coeff lambda corresponding
                        new_syst_state[i] = 1
                        self.phase[i] = 1
                        # the neurone is depolarised for a new period when it's still in phase 3 so we recalculate lambda
                        self.update_lamb(i)
                        # as the neuron is depolarised again, the time will be reset but this step will take place when
                        # the neuron finishes its 2nd phase (decrese from Vmax to PotentialDecreaseModel.threshold)
                else:
                    # the neuron in phase 3 doesn't receive any potential from others so it will continue to decrease by the
                    # function defined for this phase
                    new_syst_potentiel[i] = self.func_act_3(var, i)
                    if new_syst_potentiel[i] == self.lamb[i] * PotentialDecreaseModel.Vrest:
                        self.phase[i] = 4

            elif self.phase[i] == 4:
                # same mecanisme as phase 3
                if var != self.syst_potential[i]:
                    self.phase[i] = 0
                    new_syst_potentiel[i] = self.func_act_0(var)
                    if new_syst_potentiel[i] == PotentialDecreaseModel.threshold:
                        new_syst_state[i] = 1
                        self.phase[i] = 1
                        self.update_lamb(i)
                else:
                    new_syst_potentiel[i] = self.func_act_4(var, i)
                    if new_syst_potentiel[i] == 0:
                        # after a neuron of phase 4 reaches 0, everything is set back to starting point
                        self.phase[i] = 0
                        self.lamb[i] = 1

        self.syst_potential = new_syst_potentiel
        self.syst_state = new_syst_state

    def simulation(self, nb_steps):
        """Return a list of all the matrixes, each matrix shows the potentials of the system
        at moment t"""
        for i in range(nb_steps):
            if self.all_neurones_rest():
                self.start_syst_1()
            if self.non_transmittable():
                self.start_syst()
            else:
                self.update_system_one_step()
            yield (self.syst_state.copy(), self.syst_potential.copy())


