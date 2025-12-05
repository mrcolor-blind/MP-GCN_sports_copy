import numpy as np
import random

class PlaygroundAugmentor:
    def __init__(self, 
                 p_flip=0.5,
                 p_shuffle=0.3,
                 p_joint_mask=0.2,
                 p_temporal_shift=0.8,
                 max_shift=10):
        """
        Parámetros:
        - p_flip: probabilidad de flip horizontal
        - p_shuffle: probabilidad de barajar IDs de personas
        - p_joint_mask: probabilidad de enmascarar un joint
        - p_temporal_shift: probabilidad de aplicar temporal shift
        - max_shift: cantidad máxima de frames para el shift
        """

        self.p_flip = p_flip
        self.p_shuffle = p_shuffle
        self.p_joint_mask = p_joint_mask
        self.p_temporal_shift = p_temporal_shift
        self.max_shift = max_shift


    def random_horizontal_flip(self, data):
        """
        Flip horizontal: multiplica coordenadas X por -1.
        data: (T, M, V, C)
        """
        if random.random() < self.p_flip:
            data[..., 0] *= -1
        return data


    def random_temporal_shift(self, data):
        """
        Desplaza toda la secuencia unos frames hacia adelante o atrás.
        Se rellena con ceros para mantener la forma final.
        """
        if random.random() >= self.p_temporal_shift:
            return data

        T, M, V, C = data.shape
        shift = random.randint(-self.max_shift, self.max_shift)

        if shift == 0:
            return data

        shifted = np.zeros_like(data)

        if shift > 0:
            # Desplazar hacia adelante
            shifted[shift:] = data[:T-shift]
        else:
            # Desplazar hacia atrás
            shifted[:T+shift] = data[-shift:]

        return shifted


    def random_shuffle_ids(self, data):
        """
        Shuffle de IDs:
        Mezcla el orden de las personas M.
        El shape no se altera.
        """
        if random.random() >= self.p_shuffle:
            return data

        T, M, V, C = data.shape
        perm = np.random.permutation(M)
        data = data[:, perm, :, :]
        return data


    def random_joint_mask(self, data):
        """
        Enmascara un joint aleatorio (pone 0 en todo el T).
        """
        if random.random() < self.p_joint_mask:
            T, M, V, C = data.shape
            joint_idx = random.randint(0, V - 1)
            data[:, :, joint_idx, :] = 0
        return data

    def __call__(self, data):
        """
        Aplica todas las augmentations encadenadas.
        data: numpy array (T, M, V, C)
        """
        data = self.random_horizontal_flip(data)
        data = self.random_temporal_shift(data)
        data = self.random_shuffle_ids(data)
        data = self.random_joint_mask(data)
        return data
