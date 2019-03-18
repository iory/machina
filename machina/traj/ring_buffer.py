from collections import defaultdict

import torch

LARGE_NUMBER = 1000000000000


class RingBuffer(object):

    def __init__(self, max_steps=None,
                 default_buffer_length=1024):
        self.max_steps = max_steps if max_steps is not None \
            else LARGE_NUMBER

        if self.max_steps <= 0:
            raise ValueError('max_steps should be greater than 0. '
                             'but get {}'.format(self.max_steps))
        self.num_step = 0
        self.top = 0
        self.shape = defaultdict(lambda: None)
        self.data = defaultdict(lambda: None)
        default_buffer_length = min(default_buffer_length, self.max_steps)
        self.buffer_length = defaultdict(
            lambda: default_buffer_length)

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

    def _update_allocate_buffer(self, key, min_data_length,
                                shape=None):
        if min_data_length < self.buffer_length[key]:
            return
        min_data_length = min(min_data_length, self.max_steps)
        current_buffer_length = self.buffer_length[key]
        if current_buffer_length == self.max_steps:
            return
        next_buffer_length = current_buffer_length
        while next_buffer_length <= min_data_length:
            next_buffer_length = next_buffer_length * 2
        next_buffer_length = min(next_buffer_length, self.max_steps)
        if shape is None:
            shape = self.shape[key]
        self.data[key] = torch.cat([
            self[key], torch.zeros(next_buffer_length - self.top,
                                   *shape)])
        self.buffer_length[key] = next_buffer_length

    def _append(self, x, key=None):
        if self.data[key] is None:
            self.data[key] = torch.zeros(
                self.buffer_length[key], *x.shape)
            self.shape[key] = x.shape
        self.data[key][self.top] = x

    def append(self, data, key=None):
        t = type(data)
        if t is RingBuffer:
            other_rb = data
            if other_rb.num_step == 0:  # not initialized
                return
            elif self.num_step == 0:  # self is not initialized case
                self.num_step = other_rb.num_step
                if self.max_steps <= other_rb.num_step:
                    for k in other_rb.keys():
                        self._append(other_rb[k][0], key=k)  # for allocation
                        self._update_allocate_buffer(k, other_rb.num_step)
                        self.data[k] = other_rb[k][-self.max_steps:]
                    self.top = 0
                else:
                    for k in other_rb.keys():
                        self._append(other_rb[k][0], key=k)  # for allocation
                        self._update_allocate_buffer(k, other_rb.num_step)
                        self.data[k][:other_rb.num_step] = other_rb[k]
                    self.top = other_rb.num_step
            else:
                add_num_step = other_rb.num_step
                total_step = self.num_step + add_num_step
                next_top = (self.top + other_rb.num_step) % self.max_steps
                if total_step < self.max_steps:
                    for k in other_rb.keys():
                        self._update_allocate_buffer(k, next_top)
                        self.data[k][self.top:next_top] = other_rb[k]
                elif add_num_step <= self.max_steps:
                    for k in other_rb.keys():
                        self._update_allocate_buffer(
                            k, self.max_steps)
                        if self.top == 0:
                            self._update_allocate_buffer(k, add_num_step)
                            self.data[k][:add_num_step] = other_rb[k]
                        else:
                            other_data = other_rb[k]
                            self._update_allocate_buffer(k, self.max_steps)
                            self.data[k][self.top:
                                         self.max_steps] = \
                                other_data[:self.max_steps - self.top]
                            self.data[k][:next_top] = \
                                other_data[self.max_steps - self.top:]
                else:
                    # case of
                    # total_step >= self.max_steps
                    # add_num_step > self.max_steps
                    for k in other_rb.keys():
                        self._update_allocate_buffer(
                            k, self.max_steps)
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
            key_type = type(key)
            if key is None:
                key = list(range(len(data)))
            elif not (key_type is tuple or t is list):
                key = ['{}-{}'.format(key, index)
                       for index in range(len(data))]
            for (k, d) in zip(key, data):
                self._update_allocate_buffer(k, self.top, d.shape)
                self._append(d, k)
            self.top += 1
            self.num_step += 1
        else:
            self._update_allocate_buffer(key, self.top, data.shape)
            self._append(data, key)
            self.top += 1
            self.num_step += 1
        if self.top == self.max_steps:
            self.top = 0
        self.num_step = min(self.num_step, self.max_steps)
