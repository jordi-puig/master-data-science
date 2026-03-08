# PAC 2: Deep Reinforcement Learning y Métodos de Aproximación

Este directorio contiene el desarrollo de la **PAC2** centrada en aplicar Redes Neuronales como aproximadores de funciones de valor en entornos complejos.

## Desarrollo y Modelos 

Esta evaluación sube un nivel de dificultad introduciendo PyTorch/Tensorflow e incorporando Redes Convolucionales (CNN) para procesar secuencias de píxeles:

1. **DQN (Deep Q-Network)**: Algoritmo base de value-based reinforcement learning con Replay Buffers clásicos. Ficheros: `ex2_DQN_Agent.py`, `ex2_DQN_CNN.py`, etc.
2. **DDQN (Double Deep Q-Network)**: Modificación para solucionar el *overestimation optimista* del Q-Learning usando dos redes neuronales (Target y Behavior).
3. **Policy Gradient (PG Reinforce)**: Algoritmos *Policy-based* que optimizan la política directamente calculando el gradiente sin basarse estrictamente en Q-values (`ex4_PGReinforce.py`).

El notebook principal `_PAC2-Sol.ipynb` explica qué hiperparámetros fueron utilizados, como se implementan los buffers de memoria, curvas de recompensa, loss del error MSE en entrenamiento, grabaciones (videos) y validaciones sobre tests.
