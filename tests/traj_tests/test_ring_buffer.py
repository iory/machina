import unittest

import numpy as np
from numpy import testing
import torch

from machina.traj.ring_buffer import RingBuffer


class TestRingBuffer(unittest.TestCase):

    def test_ring_buffer(self):
        # append
        rb = RingBuffer(5)
        for i in range(11):
            rb.append(torch.ones(1) * i)

        testing.assert_almost_equal(
            rb[None].numpy(),
            np.array([[6], [7], [8], [9], [10]], 'f'))

        # append with key
        rb = RingBuffer(5)
        for i in range(11):
            rb.append(torch.ones(1) * i, key='data')

        testing.assert_almost_equal(
            rb['data'].numpy(),
            np.array([[6], [7], [8], [9], [10]], 'f'))

        # append tuple data
        rb = RingBuffer(5)
        for i in range(11):
            rb.append((torch.ones(1) * i, torch.ones(1, 1, 1) * i))

        testing.assert_almost_equal(
            rb[0].numpy(),
            np.array([[6], [7], [8], [9], [10]], 'f'))

        testing.assert_almost_equal(
            rb[1].numpy(),
            np.array([[[[6]]], [[[7]]], [[[8]]], [[[9]]], [[[10]]]], 'f'))

        # append tuple data with key
        rb = RingBuffer(5)
        for i in range(11):
            rb.append((torch.ones(1) * i, torch.ones(1, 1, 1) * i),
                      key=('vec', 'image'))

        testing.assert_almost_equal(
            rb['vec'].numpy(),
            np.array([[6], [7], [8], [9], [10]], 'f'))

        testing.assert_almost_equal(
            rb['image'].numpy(),
            np.array([[[[6]]], [[[7]]], [[[8]]], [[[9]]], [[[10]]]], 'f'))

        # key
        rb = RingBuffer(4)
        for i in range(2):
            rb.append(torch.ones(1) * i)

        testing.assert_almost_equal(
            rb[None].numpy(),
            np.array([[0], [1]], 'f'))

        rb = RingBuffer(5)
        for i in range(11):
            rb.append(torch.ones(1) * i)

        testing.assert_almost_equal(
            rb[None].numpy(),
            np.array([[6], [7], [8], [9], [10]], 'f'))

        # append RingBuffer case
        rb_a = RingBuffer(10)
        for i in range(11):
            rb_a.append((torch.ones(1) * i, torch.ones(1, 1, 1)),
                        key=('vec', 'image'))

        # other RingBuffer not initialized case
        rb_b = RingBuffer(5)
        rb_a.append(rb_b)

        testing.assert_almost_equal(
            rb_a['vec'].numpy(),
            np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]], 'f'))

        # RingBuffer not initialized case (self.max_steps < other_rb.num_step)
        rb_b = RingBuffer(5)
        rb_b.append(rb_a)

        testing.assert_almost_equal(
            rb_b['vec'].numpy(),
            np.array([[6], [7], [8], [9], [10]], 'f'))

        rb_b.append((torch.ones(1) * 11, torch.ones(1, 1, 1)),
                    key=('vec', 'image'))

        testing.assert_almost_equal(
            rb_b['vec'].numpy(),
            np.array([[7], [8], [9], [10], [11]], 'f'))

        # RingBuffer not initialized case (self.max_steps == other_rb.num_step)
        rb_b = RingBuffer(10)
        rb_b.append(rb_a)

        testing.assert_almost_equal(
            rb_b['vec'].numpy(),
            np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]], 'f'))

        rb_b.append((torch.ones(1) * 11, torch.ones(1, 1, 1)),
                    key=('vec', 'image'))

        testing.assert_almost_equal(
            rb_b['vec'].numpy(),
            np.array([[2], [3], [4], [5], [6],
                      [7], [8], [9], [10], [11]], 'f'))

        # RingBuffer not initialized case (self.max_steps > other_rb.num_step)
        rb_b = RingBuffer(20)
        rb_b.append(rb_a)

        testing.assert_almost_equal(
            rb_b['vec'].numpy(),
            np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]], 'f'))

        rb_b.append((torch.ones(1) * 11, torch.ones(1, 1, 1)),
                    key=('vec', 'image'))

        testing.assert_almost_equal(
            rb_b['vec'].numpy(),
            np.array([[1], [2], [3], [4], [5], [6],
                      [7], [8], [9], [10], [11]], 'f'))

        # elif add_num_step <= self.max_steps:
        rb_a = RingBuffer(2)
        rb_a.append((torch.ones(1) * 9, torch.ones(1, 1, 1)),
                    key=('vec', 'image'))
        rb_a.append((torch.ones(1) * 10, torch.ones(1, 1, 1)),
                    key=('vec', 'image'))
        rb_c = RingBuffer(2)
        rb_c.append((torch.ones(1) * 11, torch.ones(1, 1, 1)),
                    key=('vec', 'image'))
        rb_c.append((torch.ones(1) * 12, torch.ones(1, 1, 1)),
                    key=('vec', 'image'))

        rb_b = RingBuffer(6)
        rb_b.append(rb_a)
        testing.assert_almost_equal(
            rb_b['vec'].numpy(),
            np.array([[9], [10]], 'f'))
        rb_b.append(rb_a)
        testing.assert_almost_equal(
            rb_b['vec'].numpy(),
            np.array([[9], [10], [9], [10]], 'f'))
        rb_b.append(rb_a)
        testing.assert_almost_equal(
            rb_b['vec'].numpy(),
            np.array([[9], [10], [9], [10], [9], [10]], 'f'))
        rb_b.append(rb_c)
        testing.assert_almost_equal(
            rb_b['vec'].numpy(),
            np.array([[9], [10], [9], [10], [11], [12]], 'f'))

        rb_b.append((torch.ones(1) * 0, torch.ones(1, 1, 1)),
                    key=('vec', 'image'))

        testing.assert_almost_equal(
            rb_b['vec'].numpy(),
            np.array([[10], [9], [10], [11], [12], [0]], 'f'))

        # test else case
        rb_a = RingBuffer(2)
        rb_a.append(torch.ones(1) * 9, key='vec')
        rb_a.append(torch.ones(1) * 10, key='vec')
        rb_b = RingBuffer(6)
        for i in range(5):
            rb_b.append(torch.ones(1) * i, key='vec')

        rb_c = RingBuffer(2)
        rb_c.append(torch.ones(1) * 9, key='vec')

        rb_a.append(rb_b)

        testing.assert_almost_equal(
            rb_a['vec'].numpy(),
            np.array([[3], [4]], 'f'))

        rb_a.append(torch.ones(1) * 0, key='vec')

        testing.assert_almost_equal(
            rb_a['vec'].numpy(),
            np.array([[4], [0]], 'f'))

        rb_a.append(rb_c)

        testing.assert_almost_equal(
            rb_a['vec'].numpy(),
            np.array([[0], [9]], 'f'))

        # next_top == 0 case
        rb_a = RingBuffer(1)
        rb_a.append(torch.ones(1) * 0, key='vec')
        rb_b = RingBuffer(6)
        for i in range(6):
            rb_b.append(torch.ones(1) * i, key='vec')
        rb_a.append(rb_b)

        # memory allocation test
        rb = RingBuffer(2056, default_buffer_length=8)
        for i in range(3000):
            rb.append(torch.ones(1) * i, key='data')
        testing.assert_almost_equal(
            rb['data'].numpy(),
            np.arange(3000 - 2056, 3000).reshape(-1, 1))

        rb_a = RingBuffer(2056, default_buffer_length=8)
        for i in range(3000):
            rb_a.append((torch.ones(1) * i, torch.ones(1, 1, 1)),
                        key='obs')
        testing.assert_almost_equal(
            rb_a['obs-0'].numpy(),
            np.arange(3000 - 2056, 3000).reshape(-1, 1))

        rb_b = RingBuffer(1024, default_buffer_length=8)
        rb_b.append(rb_a)
        testing.assert_almost_equal(
            rb_b['obs-0'].numpy(),
            np.arange(3000 - 1024, 3000).reshape(-1, 1))

        rb_b = RingBuffer(1024, default_buffer_length=8)
        for i in range(13):
            rb_b.append((torch.ones(1) * i, torch.ones(1, 1, 1)), key='obs')
        rb_b.append(rb_a)
        testing.assert_almost_equal(
            rb_b['obs-0'].numpy(),
            np.arange(3000 - 1024, 3000).reshape(-1, 1))
