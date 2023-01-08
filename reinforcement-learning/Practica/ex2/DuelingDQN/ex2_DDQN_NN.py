import torch as T
import torch.nn as nn
import torch.nn.functional as F

class DDQNetwork(nn.Module):
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
        super(DDQNetwork, self).__init__()
        self.seed = T.manual_seed(seed)
        self.fc1 = nn.Linear(n_state, n_layer1)
        self.fc2 = nn.Linear(n_layer1, n_layer2)

        self.advantage = nn.Sequential(
            nn.Linear(n_layer2, n_layer3),
            nn.ReLU(),
            nn.Linear(n_layer3, n_action)
        )

        self.value = nn.Sequential(
            nn.Linear(n_layer2, n_layer3),
            nn.ReLU(),
            nn.Linear(n_layer3, 1)
        )

    def forward(self, state):
        """
        Forward pass de la xarxa neuronal amb una capa oculta de 64 nodes i una capa de sortida de 4 nodes (una per cada acció)
        amb activació ReLU en les dues capes ocultes i activació lineal en la capa de sortida 
        """
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))

        advantage = self.advantage(state)
        value = self.value(state)
        

        return value + advantage - advantage.mean()