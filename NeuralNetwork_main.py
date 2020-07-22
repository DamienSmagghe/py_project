from py_project.dialog_UI import UI_NeuralNetwork
from py_project.weighted_model import WeightedModel
from py_project.simplified_model import SimplifiedModel
from py_project.psychoactive_model import PsychoactiveModel
from py_project.potential_decrease_model import PotentialDecreaseModel

from PyQt5 import QtWidgets, QtGui, uic, QtCore
import sys
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

class NeuralNetwork():
    """Create an user interface to interact with the simulation of Neural Network."""
    MIN_SCATTER_SIZE = 50
    FRAMES_PER_UPDATE = 5

    def __init__(self):
        self.model = None
        self.dlg = UI_NeuralNetwork()
        # Parameters of model changed
        self.dlg.nb_neurons.valueChanged.connect(self.change_parameters)
        self.dlg.gamma.valueChanged.connect(self.change_parameters)
        self.dlg.beta.valueChanged.connect(self.change_parameters)
        self.dlg.alco_concen.valueChanged.connect(self.change_parameters)
        self.dlg.deltaT.valueChanged.connect(self.change_parameters)

        # Type of model changes based on these options -> recreate model
        self.dlg.dist.valueChanged.connect(self.init_model)
        self.dlg.poids.clicked.connect(self.init_model)
        self.dlg.psycho.clicked.connect(self.init_model)
        self.dlg.decrease_poten.clicked.connect(self.init_model)

        self.dlg.networkMap.clicked.connect(self.change_view)
        self.dlg.nbActive.clicked.connect(self.change_view)

        #Integrate matplotlib into QGraphicsView of dialog
        self.fig = plt.figure(figsize=(5, 5))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.dlg.graphicsView)
        self.toolbar = NavigationToolbar(self.canvas, self.dlg)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('white')
        self.ax.set_xlim(0, 500)
        self.ax.set_xticks([])
        self.ax.set_ylim(0, 500)
        self.ax.set_yticks([])
        # self.dlg.scene.addWidget(self.toolbar)
        # self.dlg.scene.addWidget(self.canvas)
        # self.plot_scene = QtWidgets.QGraphicsScene()
        # self.dlg.graphicsView.setScene(self.plot_scene)
        # self.dlg.graphicsView.scene().addWidget(self.toolbar)
        # self.dlg.graphicsView.scene().addWidget(self.canvas)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.dlg.graphicsView.setLayout(layout)

        # Timer to repeatedly call the plot function
        self.timer = QtCore.QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.plot)

        self.dlg.start.clicked.connect(self.start)
        self.dlg.pause.clicked.connect(self.pause)

    def start(self):
        """If model is already created and there is no parameter changes since the last init then send kick-off signals
        to the network.
        Otherwise, reinit the model."""
        if self.model is None:
            self.init_model()
        self.model.start_syst()
        self.timer.start()
        # self.plot()

    def plot(self):
        """Plot the graphic of simulation of model in regards of the graphic choice"""
        color = ['red' if self.model.syst_state[i] == 1 else
                 ('green' if self.model.syst_potential[i] >= 0 else 'blue') for i in range(self.model.N)]
        size = [abs(x) + NeuralNetwork.MIN_SCATTER_SIZE for x in self.model.syst_potential]
        self.ax.cla()
        self.ax.scatter(self.coord_X, self.coord_Y, s=size, lw=0.5, c=color, edgecolors=None)
        self.canvas.draw()
        if not self.model.non_transmittable():
            self.model.update_system_one_step()
        else:
            print("The network is dead. No neuron is capable of sending signals. Please click start to continue feeding signals.")
            self.timer.stop()

        # while not self.model.non_transmittable():
        #     self.model.update_system_one_step()
        #     color = ['red' if self.model.syst_state[i] == 1 else
        #              ('green' if self.model.syst_potential[i] >= 0 else 'blue') for i in range(self.model.N)]
        #     size = [abs(x) + NeuralNetwork.MIN_SCATTER_SIZE for x in self.model.syst_potential]
        #     self.ax.cla()
        #     self.ax.scatter(self.coord_X, self.coord_Y, s=size, lw=0.5, c=color, edgecolors=None)
        #     self.canvas.draw()

    def init_model(self):
        """Create model when START clicked or type of model changed"""
        self.coord_X = self.init_coord(self.dlg.nb_neurons.value())
        self.coord_Y = self.init_coord(self.dlg.nb_neurons.value())
        if not (self.dlg.poids.isChecked() or self.dlg.psycho.isChecked() or self.dlg.decrease_poten.isChecked()):
            self.model = SimplifiedModel(
                self.dlg.nb_neurons.value(),
                self.dlg.beta.value(),
                self.dlg.gamma.value()
            )
            self.init_syst_links_dist()

        elif self.dlg.poids.isChecked():
            self.model = WeightedModel(
                self.dlg.nb_neurons.value(),
                self.dlg.beta.value(),
                self.dlg.gamma.value()
            )
            self.init_syst_links_dist()
            self.model.init_system_links_weighted()

        elif self.dlg.psycho.isChecked() and not self.dlg.decrease_poten.isChecked():
            self.model = PsychoactiveModel(
                self.dlg.nb_neurons.value(),
                self.dlg.beta.value(),
                self.dlg.alco_concen.value()
            )
            self.init_syst_links_dist()
            self.model.init_system_links_ca()

        elif self.dlg.decrease_poten.isChecked():
            if self.dlg.psycho.isChecked():
                self.model = PotentialDecreaseModel(
                    self.dlg.nb_neurons.value(),
                    self.dlg.beta.value(),
                    self.dlg.gamma.value(),
                    self.dlg.alco_concen.value(),
                    self.dlg.deltaT.value()
                )
            else:
                self.model = PotentialDecreaseModel(
                    self.dlg.nb_neurons.value(),
                    self.dlg.beta.value(),
                    self.dlg.gamma.value(),
                    0,
                    self.dlg.deltaT.value()
                )
            self.init_syst_links_dist()
            self.model.init_system_links_ca()


    def init_coord(self, N):
        """Create a list which stocks the coordinates x or y of all the neurons of the network"""
        return [random.randint(0, 500) for i in range(N)]

    def init_syst_links_dist(self):
        """Create a matrix of 2 dimensions which shows the connections between neurons in the system.
        Only neurons within the R radian of another neuron i can send or receive signal from and to i
        syst_links[i][j] = gamme/beta: j connects and can send signal to i, not in reverse
        syst_links[i][j] = 0: j doesnt connect to i"""

        # alias: Point = (x,y)
        def distance_carre(P1, P2):
            x1, y1 = P1
            x2, y2 = P2
            return (x1 - x2) ** 2 + (y1 - y2) ** 2

        # Create a matrix that only allows connections between 2 neurons if whose distance is inferior to R
        syst_links = []
        R = self.dlg.dist.value()
        for i in range(self.model.N):
            syst_links.append([])
            for j in range(self.model.N):
                if i != j:
                    if distance_carre((self.coord_X[i], self.coord_Y[i]), (self.coord_X[j], self.coord_Y[j])) <= R ** 2:
                        syst_links[i].append(random.choice([0, 1, 1, 1, 1]))  # increase the chance of two neurons connected
                    else:
                        syst_links[i].append(0)
                else:
                    syst_links[i].append(self.model.gamma / self.model.beta)
        self.model.syst_links = syst_links

    def change_parameters(self):
        """Change parameters of model internally"""
        if self.model.N != self.dlg.nb_neurons.value():
            self.init_model()
        if self.model.gamma != self.dlg.gamma.value():
            self.model.gamma = self.dlg.gamma.value()
        if self.model.beta != self.dlg.beta.value():
            self.model.beta = self.dlg.beta.value()
        if self.dlg.psycho.isChecked() and self.model.ca != self.dlg.alco_concen.value():
            self.model.ca = self.dlg.alco_concen.value()
            self.model.init_system_links_ca()
        if self.dlg.decrease_poten.isChecked() and self.model.deltaT != self.dlg.deltaT.value():
            self.model.deltaT = self.dlg.deltaT.value()


    def change_view(self):
        pass

    def pause(self):
        pass


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    nn = NeuralNetwork()
    # plt.ion()
    # nn.fig.show(warn=True)
    nn.dlg.show()
    sys.exit(app.exec_())
