import random
import numpy as np
from collections import deque

import torch as T
import torch.nn.functional as F
import torch.optim as optim

from ex2_DQN_NN import DQNetwork
from ex2_Buffer import ReplayBuffer

class Agent:
    """ Agent que interactua amb l'entorn i apren a través de DQN"""    
    def __init__(self, env, seed, learning_rate= 1e-3, gamma=0.99, 
                tau=1e-3, buffer_size=100000, batch_size=64, dnn_upd=4):
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
        self.env = env
        self.seed = seed         
        self.n_state = env.observation_space.shape[0] 
        self.n_action = env.action_space.n
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size        
        self.dnn_upd = dnn_upd
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu") # Si hi ha GPU, utilitza-la

         
        random.seed(seed)   

        if T.cuda.is_available():
            print(f'Running on {T.cuda.get_device_name(0)}')            
        else:
            print('Running on CPU')
               
        # Inicialització de les xarxes locals i target i de l'optimitzador
        self.__initialize_networks()

    def __initialize_networks(self):
        # Inicialització de les xarxes locals i target            
        self.qnetwork_local = DQNetwork(self.n_state, self.n_action, self.seed).to(self.device)
        self.qnetwork_target = DQNetwork(self.n_state, self.n_action, self.seed).to(self.device)
        # Inicialització de l'optimitzador
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr = self.learning_rate)

        # Inicialització del buffer de memòria
        self.memory = ReplayBuffer(self.n_action, self.buffer_size, self.batch_size, self.seed)
        
        # Inicialització del comptador de pasos per a l'actualització de la xarxa neuronal
        self.t_step = 0

    def take_step(self, state, action, reward, next_state, done):
        """
        Afegeix l'experiència a la memòria i actualitza la xarxa neuronal
        """
        # emmagatzemar l'experiència en el buffer de memòria
        self.memory.append(state, action, reward, next_state, done)

        # Actualitzar la xarxa neuronal cada dnn_upd pasos
        self.t_step = (self.t_step + 1) % self.dnn_upd
        if self.t_step == 0:
            # Si hi ha suficients experiències en el buffer, agafar un lot i actualitzar la xarxa neuronal
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample_batch()
                self.update(experiences, self.gamma)

    def get_action(self, state, eps):
        """
        Retorna l'acció segons l'estat actual i l'epsilon-greedy
        """
        state = T.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with T.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy per a seleccionar l'acció. 
        # Si el valor aleatori és més gran que l'epsilon agafar l'acció amb el valor més alt segons la xarxa neuronal
        # Si no, agafar una acció aleatòria
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.n_action))

    def update(self, experiences, gamma):
        """
        Actualitza els pesos de la xarxa neuronal local i target
        """
        states, actions, rewards, next_states, dones = experiences

        # obtenir els valors Q de l'estat següent segons la xarxa neuronal target
        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # calcular els valors Q segons l'equació de Bellman teni en compte si l'estat és terminal i el parametre gamma
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        # obtenir els valors Q de l'estat actual segons la xarxa neuronal local
        q_expected = self.qnetwork_local(states).gather(1, actions)

        # calcular la funció de pèrdua segons l'error quadràtic mitjà
        loss = F.mse_loss(q_expected, q_targets)

        # minimitzar la funció de pèrdua amb l'optimitzador
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # actualitzar els pesos de la xarxa neuronal target amb un soft update per a reduir el problema de l'estabilitat
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update dels pesos de la xarxa neuronal target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)    


    def train(self, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        """Deep Q-Learning.

        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = eps_start  # initialize epsilon
        for i_episode in range(1, n_episodes + 1):
            state = self.env.reset()
            score = 0
            for t in range(max_t):
                action = self.get_action(state, eps)
                next_state, reward, done, _ = self.env.step(action)
                self.take_step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score
            eps = max(eps_end, eps_decay * eps)  # decrease epsilon
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window) >= 200.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100, np.mean(scores_window)))
                T.save(self.qnetwork_local.state_dict(), 'data.pth')
                break
        return scores