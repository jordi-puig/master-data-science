import torch as T
import torch.nn as nn
import torch.nn.functional as F

class DQNetwork(nn.Module):
    """ Deep Q-Network model  per a l'entrenament de l'agent DQN """
 
    def __init__(self, n_state, n_action, seed, n_layer1=64, n_layer2=64):
        """
        Inicialització de la xarxa neuronal
        Params
        =======
            n_state (int): Dimensions de l'espai d'estats
            n_action (int): Dimensions de l'espai d'accions
            n_layer1 (int): Nombre de nodes en la primera capa oculta
            n_layer2 (int): Nombre de nodes en la segona capa oculta
            seed (int): Random seed per a inicialitzar els valors aleatoris
        """
        super(DQNetwork, self).__init__()
        self.seed = T.manual_seed(seed)
        self.fc1 = nn.Linear(n_state, n_layer1)
        self.fc2 = nn.Linear(n_layer1, n_layer2)
        self.fc3 = nn.Linear(n_layer2, n_action)

    def forward(self, state):
        """
        Forward pass de la xarxa neuronal amb una capa oculta de 64 nodes i una capa de sortida de 4 nodes (una per cada acció)
        amb activació ReLU en les dues capes ocultes i activació lineal en la capa de sortida 
        """
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        return self.fc3(state)