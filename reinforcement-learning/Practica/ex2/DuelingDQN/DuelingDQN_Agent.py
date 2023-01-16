from DuelingDQN_NN import DuelingDQNetwork

import sys
sys.path.append("../DQN3")
from DQN_Agent import Agent
import torch.optim as optim
from Buffer import ReplayBuffer

class DuelingAgent(Agent):
    """ Agent que interactua amb l'entorn i apren a través de DQN"""    
    """ Es sobreescriuen els mètodes de l'agent DQN per a que utilitzi la xarxa neuronal Dueling """
    """ Fem servir la herencia per a que l'agent Dueling hereti de l'agent DQN """
   
    def __init__(self, env, seed, learning_rate= 1e-3, gamma=0.99, tau=1e-3, buffer_size=100000, batch_size=64, dnn_upd=4):
        """ Inicialitza l'agent per a l'aprenentatge per DQN
            L'agent inicialitza la xarxa neuronal local i target, el buffer de memòria i l'optimitzador    
        Params
        ======
            env: Entorn de gym
            n_state (int): Dimensions de l'espai d'estats
            n_action (int): Dimensions de l'espai d'accions
            seed (int): Random seed per a inicialitzar els valors aleatoris
            learning_rate (float): Velocitat d'aprenentatge
            gamma (float): Valor gamma de l'equació de Bellman
            tau (float): Valor de tau per a soft update del target network
            buffer_size (int): Màxima capacitat del buffer
            batch_size (int): Conjunt a agafar del buffer per a la xarxa neuronal     
            dnn_upd (int): Freqüència d'actualització de la xarxa neuronal       
        """
        super().__init__(env, seed, learning_rate, gamma, tau, buffer_size, batch_size, dnn_upd)
        self.__initialize_networks()

       
    def __initialize_networks(self):
        print("Sobreescrivim la xarxa neuronal local i target per a que siguin de tipus Dueling")        
        # Sobreescrivim la xarxa neuronal local i target per a que siguin de tipus Dueling
        self.qnetwork_local = DuelingDQNetwork(self.n_state, self.n_action, self.seed).to(self.device)
        self.qnetwork_target = DuelingDQNetwork(self.n_state, self.n_action, self.seed).to(self.device)
        # Inicialització de l'optimitzador
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr = self.learning_rate)

        # Inicialització del buffer de memòria
        self.memory = ReplayBuffer(self.n_action, self.buffer_size, self.batch_size, self.seed)
    
        # Inicialització del comptador de pasos per a l'actualització de la xarxa neuronal
        self.t_step = 0