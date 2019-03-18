from collections import defaultdict

import torch


class RingBuffer(object):

    def __init__(self, max_steps):
        self.max_steps = max_steps
        if self.max_steps <= 0:
            raise ValueError('max_steps should be greater than 0. '
                             'but get {}'.format(self.max_steps))
        self.num_step = 0
        self.top = 0
        self.data = defaultdict(lambda: None)

    def to(self, device):
        for value in self.data.values():
            value.to(device)

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def keys(self):
        return self.data.keys()

    def __getitem__(self, key):
        if self.data[key] is None:
            return self.data[key]
        if self.num_step == self.max_steps:
            return torch.cat(
                (self.data[key][self.top:],
                 self.data[key][:self.top]))
        else:
            return self.data[key][:self.top]

    def _append(self, x, key=None):
        if self.data[key] is None:
            self.data[key] = torch.zeros(self.max_steps, *x.shape)
        self.data[key][self.top] = x

    def append(self, data, key=None):
        t = type(data)
        if t is RingBuffer:
            other_rb = data
            if other_rb.num_step == 0:  # not initialized
                return
            elif self.num_step == 0:
                self.num_step = other_rb.num_step
                if self.max_steps <= other_rb.num_step:
                    for k in other_rb.keys():
                        self.data[k] = other_rb[k][-self.max_steps:]
                    self.top = 0
                else:
                    for k in other_rb.keys():
                        self.data[k] = torch.zeros(self.max_steps,
                                                   *other_rb[k][0].shape)
                        self.data[k][:other_rb.num_step] = other_rb[k]
                    self.top = other_rb.num_step
            else:
                add_num_step = other_rb.num_step
                total_step = self.num_step + add_num_step
                next_top = (self.top + other_rb.num_step) % self.max_steps
                if total_step < self.max_steps:
                    for k in other_rb.keys():
                        self.data[k][self.top:next_top] = other_rb[k]
                elif add_num_step <= self.max_steps:
                    for k in other_rb.keys():
                        if self.top == 0:
                            self.data[k][:add_num_step] = other_rb[k]
                        else:
                            if (self.top + other_rb.num_step) < self.max_steps:
                                self.data[k][self.top:
                                             self.top + add_num_step] = \
                                    other_rb[k]
                            else:
                                other_data = other_rb[k]
                                self.data[k][self.top:
                                             self.max_steps] = \
                                    other_data[:self.max_steps - self.top]
                                self.data[k][:next_top] = \
                                    other_data[self.max_steps - self.top:]
                else:
                    for k in other_rb.keys():
                        other_data = other_rb[k]
                        if next_top == 0:
                            self.data[k] = other_data[-self.max_steps:]
                        else:
                            self.data[k][:next_top] = other_data[-next_top:]
                            self.data[k][next_top:] = other_data[
                                -self.max_steps:-next_top]
                self.top = next_top
                self.num_step = total_step
        elif t is tuple or t is list:
            if key is None:
                key = list(range(len(data)))
            key_type = type(key)
            if not (key_type is tuple or t is list):
                key = ['{}-{}'.format(key, index)
                       for index in range(len(data))]
            for (k, d) in zip(key, data):
                self._append(d, k)
            self.top += 1
            self.num_step += 1
        else:
            self._append(data, key)
            self.top += 1
            self.num_step += 1
        if self.top == self.max_steps:
            self.top = 0
        self.num_step = min(self.num_step, self.max_steps)
