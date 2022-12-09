from skimage import transform
import numpy as np


def scale_lumininance(obs):
     """
     Separem cada una de les dimensions de RGB    
     Posteriorment calculem el valor del gris en una única dimensió          
     """ 
     # return rgb2gray(obs).astype(np.uint8)
     r, g, b = obs[:,:,0], obs[:,:,1], obs[:,:,2]
     bn_obs = 0.2989 * r + 0.5870 * g + 0.1140 * b 
     return bn_obs.astype(np.uint8)

def resize(obs):
    """ 
    Resize imatge a 84 * 84
    Fem servir la funció resize del package transform (inclós a skimage)
    """   
    return transform.resize(obs,(84,84))

def normalize(obs):
    """ 
    Normalitzem la imatge
    """
    obs /= (obs.max()/255.0)
    return obs.astype(np.uint8)

# Funció que realitza tot el pre-processament d'una observació
def preprocess_observation(obs):
    obs_proc = scale_lumininance(obs)
    obs_proc = resize(obs_proc)
    obs_proc = normalize(obs_proc)
    return obs_proc

def stack_frame(stacked_frames, frame, is_new):
    """Stacking Frames.
    Params
    ======
        stacked_frames (array): array de frames (al retornar-lo ha de contenir 4 frames)
        frame: Nova imatge per a afegir a l'array (s'ha d'esborrar la més antiga)
        is_new: Primera vegada que s'utilitza l'array.
    """        
    if is_new:
        stacked_frames = np.array([frame, frame, frame, frame])                                    
    else:
        stacked_frames[:-1] = stacked_frames[1:]
        stacked_frames[-1] = frame                                
    return stacked_frames
