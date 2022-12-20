import warnings
import gym
import sys

warnings.filterwarnings('ignore')
env = gym.make('SpaceInvaders-v4')

def play_games(num_games):    
    steps_list = []
    total_reward_list = []    
    for i_game in range(num_games): 
        if i_game % 2 == 0:
            print("\rEpisode {}/{}.".format(i_game, num_games), end="")
            sys.stdout.flush()
            
        total_reward, steps, done = 0, 0, False
        env.reset()
        while not done:
            env.render(mode='rgb_array')
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            if done:    
                steps_list.append(steps)
                total_reward_list.append(total_reward)
    return steps_list, total_reward_list      
          
steps_list, total_reward_list = play_games(1000)
env.close()   