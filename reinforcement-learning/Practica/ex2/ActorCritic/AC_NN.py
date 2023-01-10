import torch as T
import torch.nn as nn
import torch.nn.functional as F

class ACNetwork(nn.Module):
    """ Deep Q-Network model  per a l'entrenament de l'agent DQN """
 
    def __init__(self, n_state, n_action, seed, n_layer1=64, n_layer2=64, n_layer3=64):
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
        super(ACNetwork, self).__init__()
        self.seed = T.manual_seed(seed)
        self.fl1 = nn.Linear(n_state, n_layer1)
        self.fl2 = nn.Linear(n_layer1, n_layer2)

        self.advantage1 = nn.Linear(n_layer2, n_layer3)
        self.advantage2 = nn.Linear(n_layer3, n_action)

        self.value1 = nn.Linear(n_layer2, n_layer3)
        self.value2 = nn.Linear(n_layer3, 1)

    def forward(self, state):
        """
        Forward pass de la xarxa neuronal amb una capa oculta de 64 nodes i una capa de sortida de 4 nodes (una per cada acció)
        amb activació ReLU en les dues capes ocultes i activació lineal en la capa de sortida 
        """
        state = F.relu(self.fl1(state))
        state = F.relu(self.fl2(state))

        advantage = F.relu(self.advantage1(state))
        advantage = self.advantage2(advantage)

        value = F.relu(self.value1(state))
        value = self.value2(value)

        return value + advantage - advantage.mean()