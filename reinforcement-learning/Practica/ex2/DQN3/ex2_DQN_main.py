import gym
from collections import deque
from ex2_DQN_Agent import Agent
import numpy as np
import torch as T

BUFFER_SIZE = 100000  # Màxima capacitat del buffer
BATCH_SIZE = 64       # Conjunt a agafar del buffer per a la xarxa neuronal
GAMMA = 0.99          # Valor gamma de l'equació de Bellman
TAU = 1e-3            # Valor de tau per a soft update del target network
LEARNING_RATE = 5e-4  # Velocitat d'aprenentatge
DNN_UPD = 4           # Freqüència d'actualització de la xarxa neuronal

# initialize environment
env = gym.make('LunarLander-v2')
env.seed(0)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

# initialize agent
agent = Agent(env, seed=0, learning_rate=LEARNING_RATE, gamma=GAMMA, tau=TAU, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, dnn_upd=DNN_UPD)

def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
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
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
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
            T.save(agent.qnetwork_local.state_dict(), 'data.pth')
            break
    return scores        

# run the training session
scores = dqn()