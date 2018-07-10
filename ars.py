# Going with version 2 of the ARS algorithm in the research paper as normalisation yields better performance.

# Importing libraries - Output folder of videos
import os
import numpy

# Hyper parameters


class HyperParameters:
    def __init__(self):
        """ Initialisation of all of the hyper parameters of our algorithm.
            These are parameters which wont be changed during training.

            :var self.epoch: The number of training loops
            :type self.steps: int

            :var self.episode_length: The maximum length of an episode
            :type self.episode_length: int

            :var self.alpha: The learning rate
            :type self.alpha: float

            :var self.nb_directions: The number of perturbations applied on the
            weigh matrix
            :type self.nb_directions: int

            :var self.k_best_directions: Directions with lowest yields discarded
            :type self.k_best_directions: int

            :var self.noise: Standard deviation sigma in the gaussian distribution
            :type self.noise: float

            :var self.seed: For reproducibility
            :type self.seed: int

            :var self.env_name: Name of the environment which we wish to use
            :type self.env_name: str
        """

        self.epoch = 1000
        self.episode_length = 1000
        self.alpha = 0.02
        self.nb_directions = 16
        self.k_best_directions = 16

        # Making sure that the top directions is lower than the number of directions
        assert self.k_best_directions <= self.nb_directions

        self.noise = 0.03
        self.seed = 1
        self.env_name = ''
