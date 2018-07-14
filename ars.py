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

    def normalise(self, input_values):
        """ Normalises perceptron's input values in accordance to the value's z(standard)-score:
            - How many standard deviations the value of focus is from the mean

            :param input_values:
            :return: z-score/standard score
        """
        observed_mean = self.mean

        # Standard deviation
        std = np.sqrt(self.variance)

        # Returning normalised states
        return (input_values - observed_mean) / std


# The AI (Policy)


class Policy:

    def __init__(self, input_size, output_size):
        """ Architecture of AI Perceptron

            :param input_size: Number of signals describing state of agent:
                - The angles between the axis of the virtual robot
                - The muscle's impulses
            :param output_size: Number of actions agent can play
        """

        # Matrix of weights between input and output nodes output*inputs
        # np.zeros takes a tuple coupling hence the double order parenthesise
        self.theta = np.zeros((output_size, input_size))

    def evaluate(self, input_vec, delta=None, direction=None):
        """ Evaluates the rewards of different perturbations of the input vectors.

            :param input_vec: Matrix to be perturbed (Will be raw state in our case)
            :param delta: Perturbation matrix of small numbers following the normal distribution
            :param direction: Dictates whether perturbation will be positive or negative
            :return:
        """
        if direction is None:
            return self.theta.dot(input_vec)
        elif direction == "positive":
            return (self.theta + hp.noise*delta).dot(input_vec)
        else:
            return (self.theta - hp.noise*delta).dot(input_vec)

    def gen_delta_sample(self):
        """ Generates a sample of small perturbations following a normal distribution:
                - A gaussian distribution of mean 0 and variance 1

                :return: Sampled perturbations
        """

        # randn ==> random normal distribution
        # *self.theta.shape returns the shape of our matrix of weights
        return [np.random.randn(*self.theta.shape) for _ in range(hp.nb_directions)]
