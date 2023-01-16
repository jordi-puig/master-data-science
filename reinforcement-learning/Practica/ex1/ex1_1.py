#instal·lació de llibreries.
import warnings

import gym


env = gym.make('LunarLander-v2')
env.seed(0)

""" print("- Rang de recompenses o llindar de les recompenses: {} ".format(env.reward_range))
print("- Màxim nombre de passos per episodi: {} ".format(env.spec.max_episode_steps)) 
print("- Espai d'accions: {} ".format(env.action_space.n))
print("- Espai d'accions: {} ".format(env.action_space))
print("- Espai d'estats continuu: {} ".format(env.continuous)) """

# inicialitzem l'entorn
obs = env.reset()
t, total_reward, done = 0, 0, False

# mostrem informació inicial
print("Obs: {} Acció: {} Recompensa: {} Done: {} Info: {}".format(obs, None, None, done, None))

while not done:
    
    # escollim la acció aleatoria
    action = env.action_space.sample()
    
    # executem la acció i obtenim el nou estat, la recompensa i si hem acabat
    new_obs, reward, done, info = env.step(action)
    
    # mostrem informació
    print("Obs: {} Acció: {} Recompensa: {} Done: {} Info: {}".format(new_obs, action, reward, done, info))

    # Actualizar variables
    obs = new_obs
    total_reward += reward
    t += 1
    
print("Episodi finalitzat després de {} passos i recompensa de {} ".format(t, total_reward))
env.close()