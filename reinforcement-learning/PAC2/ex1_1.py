import warnings
import gym
import os
import imageio
from PIL import Image
import PIL.ImageDraw as ImageDraw

warnings.filterwarnings('ignore')
env = gym.make('SpaceInvaders-v4')

####
print("Rang de recompenses: {} ".format(env.reward_range))
print("Màxim nombre de passos per episodi: {} ".format(env.metadata))
print("Espai d'accions: {} ".format(env.action_space))
print("Espai d'observacions {} ".format(env.observation_space))

#Mètode per generar la imatge a partir dun estat amb un text informatiu
def _label_with_text(frame):
    '''
    frame: estat de l'entorn GYM.
    '''
    im = Image.fromarray(frame)
    im = im.resize((im.size[0]*2,im.size[1]*2))
    drawer = ImageDraw.Draw(im)
    drawer.text((1, 1), "Uoc Aprenentage Per Reforç.", fill=(255, 255, 255, 128))
    return im

#Mètode que permet crear un gif amb l'evolució d'una partida donat un entorn GYM.
def save_random_agent_gif(env):
    frames = []
    done = False
    env.reset()
    ###########################################   
    while not done:
        action = env.action_space.sample()
        frame = env.render(mode='rgb_array')
        frames.append(_label_with_text(frame))
        state, _, done, _ = env.step(action)
    ##############################################

    env.close()
    imageio.mimwrite(os.path.join('./videos/', 'random_agent_space_invader_usuari.gif'), frames, fps=60)
    
    
env = gym.make('SpaceInvaders-v4', render_mode='rgb_array')
try:
    os.makedirs('videos')
except:
    pass
save_random_agent_gif(env)