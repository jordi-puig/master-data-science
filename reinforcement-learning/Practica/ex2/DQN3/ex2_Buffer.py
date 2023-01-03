from collections import deque, namedtuple
import random
import numpy as np

import torch as T

class ReplayBuffer:
            
    """ Definim la classe ReplayBuffer que ens permetrà guardar les experiències de l'agent i poder-les reutilitzar posteriorment. """

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """ Inicialitzem els paràmetres de la classe ReplayBuffer.
        Params
        ======
            action_size (int): dimension de l'espai d'accions
            buffer_size (int): mida del buffer
            batch_size (int): tamany de la mostra del batch
            seed (int): seed per inicialitzar el generador de nombres aleatoris
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

    def append(self, state, action, reward, next_state, done):
        """ Afegeix una experiència al buffer de memòria. """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample_batch(self):
        """ Retorna una mostra aleatòria de tamany batch_size experiències del buffer de memòria.  """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = T.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = T.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = T.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = T.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = T.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """ Retorna el tamany actual del buffer de memòria. """
        return len(self.memory)