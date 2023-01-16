from AC_NN import ACNetwork

import sys
sys.path.append("../DQN3")
from DQN_Agent import Agent
import torch.optim as optim
from Buffer import ReplayBuffer

class ACAgent(Agent):
    """ Agent que interactua amb l'entorn i apren a través de l'algorisme Actor-Critic"""
    """ Sobrescriu el mètode __init__ per a que inicialitzi la xarxa neuronal local i target, el buffer de memòria i l'optimitzador """   
    """ Fem servir la herencia per a que l'agent Actor-Critic tingui tots els mètodes de l'agent DQN """
   
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
        self.qnetwork_local = ACNetwork(self.n_state, self.n_action, self.seed).to(self.device)
        self.qnetwork_target = ACNetwork(self.n_state, self.n_action, self.seed).to(self.device)

        # Inicialització de l'optimitzador
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr = self.learning_rate)

        # Inicialització del buffer de memòria
        self.memory = ReplayBuffer(self.n_action, self.buffer_size, self.batch_size, self.seed)
    
        # Inicialització del comptador de pasos per a l'actualització de la xarxa neuronal
        self.t_step = 0


    def calculateLoss(self, gamma=0.99):
        
        # calculating discounted rewards:
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)
                
        # normalizing the rewards:
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std())
        
        loss = 0
        for logprob, value, reward in zip(self.logprobs, self.state_values, rewards):
            advantage = reward  - value.item()
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value, reward)
            loss += (action_loss + value_loss)   
        return loss


   def get_action(self, state, eps):
        

        action = action_distribution.sample()
        
        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)
        
        return action.item()


        """
        Retorna l'acció segons l'estat actual i l'epsilon-greedy
        """

        # Epsilon-greedy per a seleccionar l'acció. 
        # Si el valor aleatori és més gran que l'epsilon agafar l'acció amb el valor més alt segons la xarxa neuronal
        # Si no, agafar una acció aleatòria
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.n_action))

    def get_action: 

        action_probs = F.softmax(self.action_layer(state))
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        
        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)
        
        return action.item()

    def clearMemory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]