import random
import numpy as np
from datetime import datetime
import torch as T
import torch.nn.functional as F
import torch.optim as optim

from DQN_NN import DQNetwork
from Buffer import ReplayBuffer

class Agent:
    """ Agent que interactua amb l'entorn i apren a través de DQN"""    
    def __init__(self, env, seed, learning_rate=1e-3, gamma=0.99, tau=1e-3, buffer_size=100000, batch_size=64, dnn_upd=4):
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
        # Inicialització de paràmetres i printem tots els parametres d'entrada
        print(f'Agent: {datetime.now()}')
        print("seed: ", seed)
        print("learning_rate: ", learning_rate)
        print("gamma: ", gamma)
        print("tau: ", tau)
        print("buffer_size: ", buffer_size)
        print("batch_size: ", batch_size)
        print("dnn_upd: ", dnn_upd)

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
        print("Inicialització de la xarxa neuronal DQN")

        # Inicialització de les xarxes locals i target            
        self.qnetwork_local = DQNetwork(self.n_state, self.n_action, self.seed).to(self.device)
        self.qnetwork_target = DQNetwork(self.n_state, self.n_action, self.seed).to(self.device)
        # Inicialització de l'optimitzador
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr = self.learning_rate)

        # Inicialització del buffer de memòria
        self.memory = ReplayBuffer(self.n_action, self.buffer_size, self.batch_size, self.seed)
        
        # Inicialització del comptador de pasos per a l'actualització de la xarxa neuronal
        self.t_step = 0

    def __take_step(self, state, action, reward, next_state, done):
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
                self.__update(experiences, self.gamma)

    def get_action(self, state, eps):
        """
        Retorna l'acció segons l'estat actual i l'epsilon-greedy
        """
        # Convertir l'estat a un tensor de PyTorch
        state = T.from_numpy(state).float().unsqueeze(0).to(self.device)    
        # Passar a la fase d'avaluació per a desactivar el dropout    
        # Amb T.no_grad() no es calculen els gradients per a no fer backpropagation
        # Això ens permet agilitzar el càlcul de l'acció        
        self.qnetwork_local.eval()
        with T.no_grad():
            # Obtenir els valors Q de l'estat actual per a cada acció a partir de la xarxa neuronal local
            action_values = self.qnetwork_local(state)
        # tornar a la fase d'entrenament
        self.qnetwork_local.train()

        # Epsilon-greedy per a seleccionar l'acció. 
        # Si el valor aleatori és més gran que l'epsilon agafar l'acció amb el valor més alt segons la xarxa neuronal
        # Si no, agafar una acció aleatòria
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.n_action))

    def __update(self, experiences, gamma):
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

        if self.device == 'cuda':
            self.update_loss.append(loss.detach().cpu().numpy())
        else:
            self.update_loss.append(loss.detach().numpy())

        # actualitzar els pesos de la xarxa neuronal target amb un soft update per a reduir el problema de l'estabilitat
        self.__soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def __normal_update(self, local_model, target_model):
        """
        Actualitza els pesos de la xarxa neuronal target
        """
        target_model.load_state_dict(local_model.state_dict())

    def __soft_update(self, local_model, target_model, tau):
        """
        Soft update dels pesos de la xarxa neuronal target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)    


    def train(self, n_episodes=2000, max_t=1000, eps_start=1.0, eps_min=0.01, eps_decay=0.995, nblock =100, min_episodes=250, reward_threshold=200.0, solved_by_mean_reward=True):
        """Deep Q-Learning.

        Params
        ======
            n_episodes (int): nombre màxim d'episodis
            max_t (int): maxim nombre de pasos per episodi
            eps_start (float): valor inicial d'epsilon
            eps_min (float): valor mínim d'epsilon
            eps_decay (float): factor de decaig d'epsilon
        """
        self.solved_by_mean_reward = solved_by_mean_reward
        self.reward_threshold = reward_threshold
        self.eps = eps_start  # inicialitzar epsilon
        self.nblock = nblock
        self.total_episodes = 0
        
        self.update_loss = [] 
        self.mean_update_loss = [] # llista amb els valors de la funció de pèrdua per episodi
        
        self.sync_eps = [] 

        self.training_rewards = []  # llista amb els reward per episodi
        self.mean_training_rewards = []  # llista amb la mitjana dels reward per episodi

        start_time = datetime.now()
        print("Training...")
        
        for episode in range(1, n_episodes + 1):
            state = self.env.reset()
            self.total_reward = 0   
            self.total_time = 0
            
            for t in range(max_t):
                action = self.get_action(state, self.eps)
                next_state, reward, done, _ = self.env.step(action)
                self.__take_step(state, action, reward, next_state, done)
                state = next_state
                self.total_reward += reward
                if done:
                    break

            # actualitzar epsilon
            self.eps = max(eps_min, eps_decay * self.eps)  # decrease epsilon            
            
            # afegir el reward de l'episodi a la llista
            self.__save_statistics()
            
            # mostrar informació de l'episodi actual
            self.__log_info(start_time, episode)
            
            ### comprovar si s'ha assolit el màxim d'episodis
            if self.solved_by_mean_reward:
                training = not self.__is_solved_by_episode(episode, n_episodes) and not self.__is_solved_by_mean_reward(episode, min_episodes, self.__get_mean_training_rewards())
            else:
                training = not self.__is_solved_by_episode(episode, n_episodes) and not self.__is_solved_by_reward(episode, min_episodes, 97)                        
            ### si no s'ha assolit el màxim d'episodis, continuar entrenant
            if not training:
                print('\nTraining finished.')
                self.total_time = datetime.now() - start_time
                self.total_episodes = episode
                break

            if episode % 100 == 0:
                print('\rEpisode {}\tMean Rewards: {:.2f}\t'.format(episode, self.__get_mean_training_rewards()))
  

    ######## Recuperar la mitjana dels rewards de l'últim bloc d'episodis ########
    def __get_mean_training_rewards(self):
        return np.mean(self.training_rewards[-self.nblock:])

    ######## Emmagatzemar epsilon, training rewards i loss#######
    def __save_statistics(self):
        self.sync_eps.append(self.eps)              
        self.training_rewards.append(self.total_reward)         
        self.mean_training_rewards.append(np.mean(self.training_rewards[-self.nblock:]))
        self.mean_update_loss.append(np.mean(self.update_loss))                                         
        self.update_loss = []

    
    ######## Comprovar si s'ha arribat al llindar de recompensa i un mínim d'episodis
    def __is_solved_by_mean_reward(self, episode, min_episodios, mean_rewards):  
        if mean_rewards >= self.reward_threshold and min_episodios <  episode:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, mean_rewards))
            T.save(self.qnetwork_local.state_dict(), 'data.pth')
            return True
        else:
            return False

    ######## Cert si s'ha assolit un porcentatge de episodis amb un reward superior al threshold ########
    def __is_solved_by_reward(self, episode, min_episodes, percent):
        if (len(self.training_rewards) < self.nblock):
            return False
        sum_episodes = 0
        for i in range(self.nblock):
            if self.training_rewards[-i] > self.reward_threshold:
                sum_episodes += 1
        print("sum_episodes that are greater than the reward threshold: ", sum_episodes)
        return (sum_episodes / self.nblock * 100) >= percent and episode >= min_episodes                


    ######## Comprovar si s'ha arribat al màxim d'episodis
    def __is_solved_by_episode(self, episode, max_episodes):
        if episode >= max_episodes:
            print('\nEpisode limit reached.')
            return True
        else:
            return False        


    ######## Mostrar informació de l'episodi actual
    def __log_info(self, start_time, episode):
        end_time = datetime.now()
        # get difference time
        delta = end_time - start_time 
        # time difference in minutes
        total_minutes = delta.total_seconds() / 60           
        print('\rEpisode {}\tMean Rewards: {:.2f}\tEpsilon {}\tTime {} minutes\t'
              .format(episode, self.__get_mean_training_rewards(), round(self.eps,4), round(total_minutes,2)), end="")                    



