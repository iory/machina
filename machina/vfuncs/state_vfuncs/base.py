import torch.nn as nn

from machina.utils import get_device


class BaseSVfunc(nn.Module):
    """
    Base function of State Value Function.
    It takes observations and then output value.
    For example V Func.

    Parameters
    ----------
    observation_space : gym.Space
    net : torch.nn.Module
    rnn : bool
    """

    def __init__(self, observation_space, net, rnn=False):
        nn.Module.__init__(self)
        self.observation_space = observation_space
        self.net = net

        self.rnn = rnn
        self.hs = None

    def reset(self):
        """
        reset for rnn's hidden state.
        """
        if self.rnn:
            self.hs = None

    def _check_obs_shape(self, obs):
        """
        Reshape input appropriately.
        """
        if self.rnn:
            additional_shape = 2
        else:
            additional_shape = 1
        if len(obs.shape) < additional_shape + len(self.observation_space.shape):
            for _ in range(additional_shape + len(self.observation_space.shape) - len(obs.shape)):
                obs = obs.unsqueeze(0)
        return obs
