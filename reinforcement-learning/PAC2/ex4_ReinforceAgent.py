import numpy as np
import torch
from ex2_Preprocess import preprocess_observation, stack_frame
from datetime import datetime

class ReinforceAgent:

    ###################################################
    ######TODO 1: declarar variables ##################
    def __init__(self, env, dnnetwork, nblock=100, reward_threshold=350):
        """
        Params
        ======
        env: entorno
        dnnetwork: clase con la red neuronal diseñada
        nblock: bloque de los X últimos episodios de los que se calculará la media de recompensa
        reward_threshold: umbral de recompensa definido en el entorno
        """
        self.env = env
        self.dnnetwork = dnnetwork
        self.nblock = nblock
        self.reward_threshold = reward_threshold
        self.initialize()
     #######

    ###############################################################
    ####TODO 2: inicialitzar variables extra que es necessitin ####
    def initialize(self):
        self.batch_rewards = []
        self.batch_actions = []
        self.batch_states = []
        self.batch_counter = 1

        self.training_update_loss = []
        self.training_rewards = []
        self.mean_training_rewards = []
        self.update_loss = []       

    ######

    ## Entrenament
    def train(self, gamma=0.99, max_episodes=2000, batch_size=10):
        self.gamma = gamma
        self.batch_size = batch_size

        episode = 0
        action_space = np.arange(self.env.action_space.n)
        training = True
        start_time = datetime.now()
        print("Training...")
        while training:
            start_game = preprocess_observation(self.env.reset())
            state0 = stack_frame(None, start_game, True)
            episode_states = []
            episode_rewards = []
            episode_actions = []
            gamedone = False

            while gamedone == False:
                ##########################################################
                ######TODO 3: Prendre una nova acció #####################
                action_probs = self.dnnetwork.get_action_prob(state0).detach().numpy()   #distribució de probabilitat de les accions donat l'estat actual                                             
                action_probs = np.squeeze(action_probs)                    
                action = np.random.choice(action_space, p=action_probs) #acció aleatòria de la distribució de probabilitat
                next_state, reward, gamedone, _ = self.env.step(action)
                #######

                # Emmagatzemem experiències que es van obtenint en aquest episodi
                episode_states.append(state0)
                episode_rewards.append(reward)
                episode_actions.append(action)
                next_state = stack_frame(state0, preprocess_observation(next_state), False)
                state0 = next_state

                if gamedone:
                    episode += 1
                    # Calculem el terme del retorn menys la línia de base
                    self.batch_rewards.extend(self.discount_rewards(episode_rewards))
                    self.batch_states.extend(episode_states)
                    self.batch_actions.extend(episode_actions)
                    self.batch_counter += 1

                    #####################################################################################
                    ###TODO 5: calcular mitjana de recompenses dels últims X episodis, i emmagatzemar####
                    self.training_rewards.append(sum(episode_rewards)) # guardamos las recompensas obtenidas   
                    mean_rewards = np.mean(self.training_rewards[-self.nblock:])
                    self.mean_training_rewards.append(mean_rewards)
                    ######


                    # Actualitzem la xarxa quan es completa la mida del batch
                    if self.batch_counter == self.batch_size:
                        self.update(self.batch_states, self.batch_rewards, self.batch_actions)

                        #######################################
                        ###TODO : emmagatzemar training_loss###
                        ###########
                        self.training_update_loss.append(np.mean(self.update_loss))                                         
                        self.update_loss = []

                        # Resetejem les variables de l'episodi
                        self.batch_rewards = []
                        self.batch_actions = []
                        self.batch_states = []
                        self.batch_counter = 1

                    end_time = datetime.now()
                    # get difference time
                    delta = end_time - start_time 
                    # time difference in minutes
                    total_minutes = delta.total_seconds() / 60 # temps total d'entrenament en minuts
                    estimated_remain_time = total_minutes / episode * (max_episodes - episode) # temps estimat que resta per a acabar l'entrenament en minuts               
                    print("\rEpisode {:d} Mean Rewards {:.2f} Time {} minutes Remaining Time {} minutes\t\t".format(episode, mean_rewards, round(total_minutes,2), round(estimated_remain_time,2)), end="")

                    # Comprovem que encara queden episodis
                    if episode >= max_episodes:
                        training = False
                        print('\nEpisode limit reached.')
                        break

                    # Acaba el joc si la mitjana de recompenses ha arribat al llindar fixat per a aquest joc
                    if mean_rewards >= self.reward_threshold:
                        training = False
                        print('\nEnvironment solved in {} episodes!'.format(episode))
                        break

    ########################################################
    ###TODO 4: càlcul del retorn menys la línia de base ####
    def discount_rewards(self, rewards):
        discount_r = np.zeros_like(rewards)
        timesteps = range(len(rewards))
        reward_sum = 0
        for i in reversed(timesteps):
            reward_sum = rewards[i] + self.gamma*reward_sum
            discount_r[i] = reward_sum
        baseline = (discount_r - np.mean(discount_r)) / np.std(discount_r) # línia de base del retorn (normalitzat)     
        return baseline                          

    ##########


    ## Actualizació
    def update(self, batch_s, batch_r, batch_a):
        self.dnnetwork.optimizer.zero_grad()  # eliminem qualsevol gradient passat
        state_t = torch.FloatTensor(batch_s)
        reward_t = torch.FloatTensor(batch_r)
        action_t = torch.LongTensor(batch_a)
        loss = self.calculate_loss(state_t, action_t, reward_t) # calculem la pèrdua
        loss.backward() # calculem la diferència per obtenir els gradients
        self.dnnetwork.optimizer.step() # apliquem els gradients a la xarxa neuronal
        #Desem els valors de pèrdua
        if self.dnnetwork.device == 'cuda':
            self.update_loss.append(loss.detach().cpu().numpy())
        else:
            self.update_loss.append(loss.detach().numpy())

    #################################################
    ###TODO 6: Càlcul de la pèrdua ##################
    # Recordatori: cada actualització és proporcional al producte del retorn i el gradient de la probabilitat
    # de prendre l'acció presa, dividit per la probabilitat de prendre aquesta acció (logaritme natural)
    def calculate_loss(self, state_t, action_t, reward_t):
        logprob = torch.log(self.dnnetwork.get_action_prob(state_t)) # logaritme natural de la probabilitat de les accions
        selected_logprobs = reward_t * logprob[np.arange(len(action_t)), action_t] # producte del retorn i el gradient de la probabilitat
        loss = -selected_logprobs.mean()
        return loss # retornem la pèrdua
     ########


