import gym
from gym import spaces
from gym.utils import seeding

def cmp(a, b):
    """
    Funció que compara els enters a i b. Retorna -1 si a > b, 1 si b > a i 0 si son iguals.   
    """
    return int((a > b)) - int((a < b))

# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def draw_card(np_random):
    """
    Retorna un valor aleatori de l'array deck definit previament.
    """
    return np_random.choice(deck)


def draw_hand(np_random):
    """
    Retorna un array de 2 elements format per dos valors aleatoris de l'array deck.
    """
    return [draw_card(np_random), draw_card(np_random)]


def usable_ace(hand):
    """
    Funció que espera com a paràmetre un array d'enters.
    Retorna true si 1 és troba en l'array (hand - mà - paràmetre d'entrada) i  el valors de la suma dels elements és menor o igual a 21, altrament retorna false.
    """   
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):
    """
    Funció que espera com a paràmetre un array d'enters.
    Retorna la suma dels elements de l'array si la funció usabla_ace retorna false, altrament fa la suma dels elements més 10.
    """   
    if usable_ace(hand):
            return sum(hand) + 10
    return sum(hand)


def is_bust(hand):
    """
    Funció que espera com a paràmetre un array d'enters. 
    Retorna true si la suma dels dos elements és més gran que 21, altrament retorna false.
    """ 
    return sum_hand(hand) > 21


def score(hand): 
    """
    Funció que espera com a paràmetre un array d'enters. 
    Retorna 0 si la funció is_bust retorna true,és a dir, la suma dels element de la mà és més gran de 21. Sino retorna la suma de la mà.
    """
    return 0 if is_bust(hand) else sum_hand(hand)


class BlackjackEnv(gym.Env):
    """
    Aquest classe conté el codi per implementar l'environtment. 
    Tenim el constructor de la classe (__init__) on definim les variables per contenir tant les accions (action_space) com l'espai observacional (observation_space). A més es reseteja l'entorn executant el mètode _reset de la propia classe.
    Després és defineixen les funcions (step, reset) que encapsulen les crides als mètodes (_step, _reset) que contenen la lògica real per a generar aquests processos.    
    """
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2)))
        self._seed()
        self._reset()

    def reset(self):
        return self._reset()

    def step(self, action):
        return self._step(action)

    def _seed(self, seed=None):
        """
        Retorna el seed aleatori creat amb mètode np_random. 
        Es fa servir inicialitzar el generador de números aleatoris.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        """
        Executa la següent acció.
        Mentre sigui hit (action = 1):
            - li donem una altre carta al jugador. 
            - si s'ha passat de 21 (is_bust = true) acabem amb done = true i reward és -1- 
            - si no ha passat de 21 done = false i reward = 0
        Si l'action és stick (action = 0) el jugador ha finalitzat la partida i és el torn del dealer:
            - mentre sigui menor de 17 li anem donant cartes al dealaer
            - el reward serà la comparació entre el valor del player i el dealer on si:
                - player > dealer = 1
                - player = dealer = 0
                - player < dealer = -1
        Finalment retorna la observació de l'estat actual.
        """
        assert self.action_space.contains(action), "Fallo, Action = {}".format(action)
        if action:  # Descripción
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                done = True
                reward = -1
            else:
                done = False
                reward = 0
        else:  # Descripción
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))
            reward = cmp(score(self.player), score(self.dealer))
        return self._get_obs(), reward, done, {}


    def _get_obs(self):
        """
        Recupera la observació de l'entorn en el moment actual amb els valors:
            - suma de la mà en el moment actual.
            - valor que ha tret el dealer.
            - si és un ace usable o no.
        """       
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player))

    def _reset(self):
        """
        Inicialitza la partida. 
        Primer donem una mà per al dealer i després per al jugador (player).
        A més mentre la mà actual del player sigui menor de 12 anirem afegint cartes
        Com a resultat retornem els tres elements de la observació
        """
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)
        while sum_hand(self.player) < 12:
            self.player.append(draw_card(self.np_random))
        return self._get_obs()