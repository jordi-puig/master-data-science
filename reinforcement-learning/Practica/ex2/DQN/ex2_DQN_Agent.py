import torch
from copy import deepcopy
import numpy as np
from datetime import datetime

class DQNAgent:
    ###################################################
    ###### Declarar variables ##################
    def __init__(self, env, main_network,
                 buffer, reward_threshold,
                 epsilon=0.1, eps_decay=0.99, batch_size=32, nblock=100):
        """"
        Params
        ======
        env: entorn
        main_network: classe amb la xarxa neuronal dissenyada
        target_network: xarxa objectiu
        buffer: classe amb el buffer de repetició d'experiències
        epsilon: epsilon
        eps_decay: epsilon decay
        batch_size: batch size
        nblock: bloc dels X darrers episodis dels quals es calcularà la mitjana de recompensa
        reward_threshold: llindar de recompensa on es considera que s'ha assolit el problema
        """
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.env = env
        self.main_network = main_network
        self.target_network = deepcopy(main_network) # xarxa objectiu (còpia de la principal)
        self.buffer = buffer
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.nblock = nblock # són els blocs dels X episodis dels quals es calcularà la mitjana de recompensa
        self.reward_threshold = reward_threshold # llindar de recompensa, si es supera es para el procés d'entrenament
        self.initialize()

    ###################################################################
    ##### Inicialitzar variables extra que són necessàries per a l'entrenament
    def initialize(self):
        self.update_loss = []
        self.training_update_loss = []
        self.training_rewards = []
        self.mean_training_rewards = []
        self.sync_eps = []
        self.total_reward = 0
        self.step_count = 0
        self.state0 = self.env.reset() # reset de l'entorn i obtenció de l'estat inicial

    #################################################################################
    ###### Prendre una nova acció ############################################
    def take_step(self, eps, mode='train'):
        if mode == 'explore':
            action = self.env.action_space.sample()  # acció aleatòria al burn-in
        else:
            action = self.main_network.get_action(self.state0, eps) # acció a partir del valor de Q (elecció de l'acció amb millor Q)
            self.step_count += 1

        # Realització de l'acció i obtenció del nou estat i la recompensa.
        new_state, reward, done, _ = self.env.step(action) 
        self.total_reward += reward
        self.buffer.append(self.state0, action, reward, done, new_state) # guardar experiència al buffer
        self.state0 = new_state

        # Resetejar entorn 'if done'
        if done:
            self.state0 = self.env.reset()
        return done
    
    ## Entrenament
    def train(self, gamma=0.99, max_episodes=50000, dnn_update_frequency=4, dnn_sync_frequency=2000, min_episodios=250, min_epsilon = 0.01):
        self.gamma = gamma
        # Omplim el buffer amb N experiències aleatòries per a fer burn-in
        print("Filling replay buffer...")
        while self.buffer.burn_in_capacity() < 1:
            self.take_step(self.epsilon, mode='explore')

        episode = 0
        training = True
        start_time = datetime.now()
        print("Training...")
        while training:
            self.state0 = self.env.reset()
            self.total_reward = 0
            gamedone = False
            while gamedone == False:
                # L'agent pren una acció
                gamedone = self.take_step(self.epsilon, mode='train')
                #################################################################################
                #### Actualitzar xarxa principal segons la freqüència establerta #########
                if self.step_count % dnn_update_frequency == 0:
                    self.update()
                
                ########################################################################################
                ### Sincronitzar xarxa principal i xarxa objectiu segons la freqüència establerta
                if self.step_count % dnn_sync_frequency == 0:
                    self.target_network.load_state_dict(self.main_network.state_dict())                                        
                                       
                if gamedone:
                    episode += 1                    
                    ##################################################################
                    ######## Emmagatzemar epsilon, training rewards i loss#######
                    self.save_statistics()

                    #######################################################################################
                    ### Calcular la mitjana de recompensa dels últims X episodis, i emmagatzemar####
                    mean_rewards = np.mean(self.training_rewards[-self.nblock:])
                    self.calculate_mean_rewards(mean_rewards)

                    ##################################################################
                    ### calcular el temps restant i mostrar-lo
                    self.calculate_remaining_time(start_time, episode, max_episodes, mean_rewards)
              
                    #################################################################################
                    ### comprovar si s'ha assolit el màxim d'episodis
                    training = not self.is_solved_by_episode(episode, max_episodes)

                    #################################################################################
                    ### si no s'ha assolit el màxim d'episodis, continuar entrenant
                    if not training:
                        print('\nTraining finished.')
                        break

                    #################################################################################
                    ### comprovar si s'ha assolit el llindar de recompensa i un mínim d'episodis
                    training = not self.is_solved_by_reward(episode, min_episodios, mean_rewards)

                    #################################################################################
                    ### si no s'ha assolit el màxim d'episodis, continuar entrenant
                    if not training:
                        print('\nTraining finished.')
                        break
                    
                    #################################################################################
                    ###### Actualitzar epsilon ########
                    # actualitzar epsilon segons la velocitat de descens fixada on no pot ser inferior a min_epsilon
                    self.epsilon = max(self.epsilon * self.eps_decay, min_epsilon)


    ##################################################################
    ######## Comprovar si s'ha arribat al llindar de recompensa i un mínim d'episodis
    def is_solved_by_reward(self, episode, min_episodios, mean_rewards):  
        if mean_rewards >= self.reward_threshold and min_episodios <  episode:
            print('\nEnvironment solved in {} episodes!'.format(episode))
            return True
        else:
            return False

    ##################################################################
    ######## Comprovar si s'ha arribat al màxim d'episodis
    def is_solved_by_episode(self, episode, max_episodes):
        if episode >= max_episodes:
            print('\nEpisode limit reached.')
            return True
        else:
            return False

    ###############################################################
    ######## Calcular el temps restant i mostrar-lo
    def calculate_remaining_time(self, start_time, episode, max_episodes, mean_rewards):
        end_time = datetime.now()
        # get difference time
        delta = end_time - start_time 
        # time difference in minutes
        total_minutes = delta.total_seconds() / 60
        estimated_remain_time = total_minutes / episode * (max_episodes - episode)                    
        print("\rEpisode {:d} Mean Rewards {:.2f} Epsilon {} Time {} minutes Remaining Time {} minutes\t\t"
              .format(episode, mean_rewards, round(self.epsilon,4),round(total_minutes,2), round(estimated_remain_time,2)), end="")               
        

    ##################################################################
    ######## Calcular la mitjana de recompensa dels últims X episodis, i emmagatzemar####
    def calculate_mean_rewards(self, mean_rewards):
        self.mean_training_rewards.append(mean_rewards)

    ##################################################################
    ######## Emmagatzemar epsilon, training rewards i loss#######
    def save_statistics(self):
        self.sync_eps.append(self.epsilon)
        self.training_rewards.append(self.total_reward)         
        self.training_update_loss.append(np.mean(self.update_loss))                                              
        self.update_loss = []

    ####################################
    #### Càlcul de la pèrdua ####
    def calculate_loss(self, batch):
        # Separem les variables de l'experiència i les convertim a tensors
        states, actions, rewards, dones, next_states = [i for i in batch]
        rewards_vals = torch.FloatTensor(rewards).to(device=self.device)
        actions_vals = torch.LongTensor(np.array(actions)).to(device=self.device).reshape(-1,1)
        dones_t = torch.ByteTensor(dones).to(device=self.device)

        # Obtenim els valors de Q de la xarxa principal
        qvals = torch.gather(self.main_network.get_qvals(states), 1, actions_vals)
        # Obtenim els valors de Q de la xarxa objectiu
        # El paràmetre detach() evita que aquests valors actualitzin la xarxa objectiu
        qvals_next = torch.max(self.target_network.get_qvals(next_states), dim=-1)[0].detach()
        qvals_next[dones_t] = 0 # 0 en estats terminals

        #################################################################################
        ### Calcular l'equació de Bellman
        expected_qvals = rewards_vals + self.gamma * qvals_next
        
        loss = torch.nn.MSELoss()(qvals, expected_qvals.reshape(-1,1))
        return loss

    def update(self):
        self.main_network.optimizer.zero_grad()  # eliminem qualsevol gradient passat
        batch = self.buffer.sample_batch(batch_size=self.batch_size) # seleccionem un conjunt del buffer
        loss = self.calculate_loss(batch)# calculem la pèrdua
        loss.backward() # calculem la diferència per obtenir els gradients
        self.main_network.optimizer.step() # apliquem els gradients a la xarxa neuronal
        # Guardem els valors de pèrdua
        if self.device == 'cuda':
            self.update_loss.append(loss.detach().cpu().numpy())
        else:
            self.update_loss.append(loss.detach().numpy())