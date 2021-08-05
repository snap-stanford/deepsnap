import math
import torch
import unittest

from torch import nn
from deepsnap.hetero_gnn import forward_op

class TestHeteroGNN(unittest.TestCase):

    def test_hetero_gnn_forward(self):
        xs = {}
        layers = nn.ModuleDict()
        emb_dim = 5
        feat_dim = 10
        num_samples = 8
        keys = ['a', 'b', 'c']

        for key in keys:
            layers[key] = nn.Linear(feat_dim, emb_dim)
            xs[key] = torch.ones(num_samples, feat_dim)

        ys = forward_op(xs, layers)
        for key in keys:
            self.assertEqual(ys[key].shape[0], num_samples)
            self.assertEqual(ys[key].shape[1], emb_dim)

if __name__ == "__main__":
    unittest.main()