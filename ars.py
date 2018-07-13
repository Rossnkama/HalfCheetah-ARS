# Going with version 2 of the ARS algorithm in the research paper as normalisation yields better performance.

# Importing libraries - Output folder of videos
import os
import numpy as np

# Hyper parameters


class HyperParameters(object):

    def __init__(self):
        """ Initialisation of all of the hyper parameters of our algorithm.
            These are parameters which wont be changed during training.

            :var self.epoch: The number of training loops

            :var self.episode_length: The maximum length of an episode
            :type self.episode_length: int

            :var self.alpha: The learning rate
            :type self.alpha: float

            :var self.nb_directions: The number of perturbations applied on the weigh matrix

            :var self.k_best_directions: Directions with lowest yields discarded

            :var self.noise: Standard deviation sigma in the gaussian distribution
            :type self.noise: float

            :var self.seed: For reproducibility

            :var self.env_name: Name of the environment which we wish to use
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


# Online normalisation of states


class Normaliser(object):

    def __init__(self, nb_inputs):
        """
            :param nb_inputs: Number of inputs of the perceptron
            :type nb_inputs: int

            :var self.mean: Keeps track of the current mean for each item in the input vector
            :var self.mean_diff: Numerator to variance

            :var self.counter: Keeps track of how many states we've encountered since the beginning
        """
        self.n_states = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.variance = np.zeros(nb_inputs)

        self.states_sum = 0

    def observe(self, new_state):
        """

            :param new_state: New state observed by agent
            :return: None
        """

        # To increment counter by 1f so that we can calculate the mean
        self.n_states += 1.0

        self.states_sum += new_state
        last_mean = self.mean.copy()

        # Online mean computation
        self.mean += (new_state - self.mean) / self.n_states
        # self.mean += (self.states_sum / self.n_states) / 2
