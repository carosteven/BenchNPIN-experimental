from abc import ABC, abstractmethod


class BasePolicy(ABC):

    """
    A base policy
    """

    def __init__(self) -> None:
        ...


    def train(self):
        """
        Train the policy
        """
        raise NotImplementedError


    @abstractmethod
    def evaluate(self, num_eps: int, model_eps: str ='latest') -> list:
        """
        Implement this function to evaluate the policy for `num_epochs`
        :param num_eps: total number of episodes to evaluate
        :param model_eps: the specific model checkpoint. Provide an integer step number as a string
        :return: a list of epsiode rewards
        """
        raise NotImplementedError


    @abstractmethod
    def act(self, observation, **kwargs):
        """
        Implement this function to compute an action given the observation
        """
        raise NotImplementedError
