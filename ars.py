# Going with version 2 of the ARS algorithm in the research paper as normalisation yields better performance.

# Importing libraries - Output folder of videos
import os
import numpy as np

# Hyper parameters


class HyperParameters(object):

    def __init__(self):
        """ Initialisation of all of the hyper parameters of our algorithm.
            These are parameters which wont be changed during training.
        """

        self.epoch = 1000
        self.episode_length = 1000

        # Learning rate
        self.alpha = 0.02

        # the number of perturbations applied to the weight matrix
        self.nb_directions = 16

        # Directions with highest yield
        self.k_best_directions = 16

        # Making sure that the top directions is lower than the number of directions
        assert self.k_best_directions <= self.nb_directions

        # Standard deviation in gaussian distribution
        self.noise = 0.03

        # For reproducibility
        self.seed = 1

        self.env_name = ''


# Online normalisation of states


class Normaliser(object):

    def __init__(self, nb_inputs):
        """
            :param nb_inputs: Number of inputs of the perceptron
            :type nb_inputs: int
        """
        self.n_states = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.variance = np.zeros(nb_inputs)

        self.states_sum = 0

    def observe(self, new_state):
        """ Observes new signals inbound to the perceptron

            :param new_state: New state observed by agent
            :return: None
        """

        # To increment counter by 1f so that we can calculate the mean
        self.n_states += 1.

        # Sum of all states
        self.states_sum += new_state
        last_mean = self.mean.copy()

        # Online mean and variance computation
        self.mean = (self.mean + (self.states_sum / self.n_states)) / 2

        # Sum of square differences between value and mean
        self.mean_diff += (new_state - last_mean) * (new_state - self.mean)

        # Variance must never be equal to zero
        self.variance = (self.mean_diff / new_state).clip(min=1e-2)
