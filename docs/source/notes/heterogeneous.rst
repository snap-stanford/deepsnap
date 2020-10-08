Heterogeneous Graph
===================

Heterogeneous Graph in DeepSNAP.

.. contents::
    :local:

Heterogeneous Graph Data Format
-------------------------------

Features:

* `node_feature`: The feature of each node (`torch.tensor`)
* `node_type`: The node type of each node (`string`)
* `edge_type`: The edge type of each edge (`string`)
* `node_label` (optional): The label of each node (`int`)

Here is an example heterogeneous graph (in NetworkX multigraph) that is supported by DeepSNAP:

.. code-block:: shell

	wget https://www.dropbox.com/s/gb6b3fixmaadltu/WN18.gpickle

.. code-block:: python

	import networkx as nx

	G = nx.read_gpickle("WN18.gpickle")
	for node in G.nodes(data=True):
	  print(node)
	  break
	for edge in G.edges(data=True):
	  print(edge)
	  break
	print("Number of edges is {}".format(G.number_of_edges()))
	print("Number of nodes is {}".format(G.number_of_nodes()))
	>>> (0, {'node_type': 'n1', 'node_feature': tensor([1., 1., 1., 1., 1.])})
	>>> (0, 1, {'edge_feature': tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), 'edge_type': '0'})
	>>> Number of edges is 141442
	>>> Number of nodes is 41105

Use DeepSNAP with the Heterogeneous Graph
-----------------------------------------

* Generate internal heterogeneous GNN layers. Each GNN layer is a dictionary that the key is the message type which is in a tuple of (node_type_1, edge_type, node_type_2). This means that the information will be passed through each message type. The value of the dictionary is the internal heterogeneous GNN layers such as :class:`deepsnap.hetero_gnn.HeteroSAGEConv`.

Here is an example function to generate the internal heterogeneous GNN layers:

.. code-block:: python

	def generate_2convs_link_pred(hete, conv, hidden_size):
		convs1 = {}
		convs2 = {}
		for message_type in hete.message_types:
			n_type = message_type[0]
			s_type = message_type[2]
			n_feat_dim = hete.get_num_node_features(n_type)
			s_feat_dim = hete.get_num_node_features(s_type)
			label_dim = 2 # output size
			convs1[message_type] = conv(n_feat_dim, hidden_size, s_feat_dim)
			convs2[message_type] = conv(hidden_size, label_dim, hidden_size)
		return convs1, convs2

Call the function:

.. code-block:: python

	from deepsnap.hetero_graph import HeteroGraph
	from deepsnap.dataset import GraphDataset
	from deepsnap.batch import Batch
	from deepsnap.hetero_gnn import *
	from torch.utils.data import DataLoader

	hete = HeteroGraph(G)
	hidden_size = 32

	# Generate two internal GNN layers for link prediction
	conv1, conv2 = generate_2convs_link_pred(hete, HeteroSAGEConv, hidden_size)

	dataset = GraphDataset([hete], task='link_pred')
	dataset_train, dataset_val, dataset_test = dataset.split(transductive=True,
	                                                        split_ratio=[0.8, 0.1, 0.1])
	train_loader = DataLoader(dataset_train, collate_fn=Batch.collate(),
	                    batch_size=1)
	val_loader = DataLoader(dataset_val, collate_fn=Batch.collate(),
	                    batch_size=1)
	test_loader = DataLoader(dataset_test, collate_fn=Batch.collate(),
	                    batch_size=1)
	dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

* Pass each of the internal GNN layers to `HeteroConv` and generate wrapper heterogeneous layers.

* Write the `forward` function for the heterogeneous graph network. Notice that the `node_feature` and the returned value of the wrapper GNN layers is in format of dictionary that the key is the `node_type` and value is its corresponding embedding tensor. So, adding the nonlinear activation or apply `dropout` layer also need to loop through the dictionary. We provide helper function such as the :func:`deepsnap.hetero_gnn.forward_op` will automatically loop through the dictionary.

Here is an example network (link prediction task) constructed by aforementioned DeepSNAP functionalities:

.. code-block:: python

	import torch
	import torch.nn as nn
	import torch.nn.functional as F
	from deepsnap.hetero_gnn import *

	# Define the heterogeneous GNN
	class HeteroNet(torch.nn.Module):
	    def __init__(self, conv1, conv2, dropout):
	        super(HeteroNet, self).__init__()
	        
	        self.conv1 = HeteroConv(conv1) # wrap the internal GNN layer
	        self.conv2 = HeteroConv(conv2)
	        self.loss_fn = torch.nn.BCEWithLogitsLoss()
	        self.dropout = dropout

	    def forward(self, data):
	        x = forward_op(data.node_feature, F.dropout, p=self.dropout, training=self.training)
	        x = forward_op(x, F.relu)
	        x = self.conv1(x, data.edge_index)
	        x = forward_op(x, F.dropout, p=self.dropout, training=self.training)
	        x = forward_op(x, F.relu)
	        x = self.conv2(x, data.edge_index)

	        pred = {}
	        for message_type in data.edge_label_index:
	            nodes_first = torch.index_select(x['n1'], 0, data.edge_label_index[message_type][0,:].long())
	            nodes_second = torch.index_select(x['n1'], 0, data.edge_label_index[message_type][1,:].long())
	            pred[message_type] = torch.sum(nodes_first * nodes_second, dim=-1)
	        return pred

	    def loss(self, pred, y, edge_label_index):
	        loss = 0
	        for key in pred:
	            p = torch.sigmoid(pred[key])
	            loss += self.loss_fn(p, y[key].type(pred[key].dtype))
	        return loss

* Define the `train` and `test` functions.

.. code-block:: python

	import numpy as np
	import copy

	# Train function
	def train(model, dataloaders, optimizer, args):
	    val_max = 0
	    best_model = model
	    t_accu = []
	    v_accu = []
	    e_accu = []
	    for epoch in range(1, args["epochs"]):
	        for iter_i, batch in enumerate(dataloaders['train']):
	            batch.to(args["device"])
	            model.train()
	            optimizer.zero_grad()
	            pred = model(batch)
	            loss = model.loss(pred, batch.edge_label, batch.edge_label_index)
	            loss.backward()
	            optimizer.step()

	            log = 'Epoch: {:03d}, Train loss: {:.4f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
	            accs = test(model, dataloaders, args)
	            t_accu.append(accs['train'])
	            v_accu.append(accs['val'])
	            e_accu.append(accs['test'])

	            print(log.format(epoch, loss.item(), accs['train'], accs['val'], accs['test']))
	            if val_max < accs['val']:
	                val_max = accs['val']
	                best_model = copy.deepcopy(model)

	    log = 'Best: Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
	    accs = test(best_model, dataloaders, args)
	    print(log.format(accs['train'], accs['val'], accs['test']))

	    return t_accu, v_accu, e_accu

	# Test function
	def test(model, dataloaders, args):
	    model.eval()
	    accs = {}
	    for mode, dataloader in dataloaders.items():
	        acc = 0
	        for i, batch in enumerate(dataloader):
	            num = 0
	            batch.to(args["device"])
	            pred = model(batch)
	            for key in pred:
	                p = torch.sigmoid(pred[key]).cpu().detach().numpy()
	                pred_label = np.zeros_like(p, dtype=np.int64)
	                pred_label[np.where(p > 0.5)[0]] = 1
	                pred_label[np.where(p <= 0.5)[0]] = 0
	                acc += np.sum(pred_label == batch.edge_label[key].cpu().numpy())
	                num += len(pred_label)
	        accs[mode] = acc / num
	    return accs

* Specify parameters and start trainning!

.. code-block:: python

	args = {
	    "device": "cuda",
	    "epochs": 300
	}

	# Build the model and train
	model = HeteroNet(conv1, conv2, 0.2).to(args["device"])
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
	t_accu, v_accu, e_accu = train(model, dataloaders, optimizer, args)

* You can run this `DeepSNAP Heterogeneous Graph Guide <https://colab.research.google.com/drive/1H8X6z1R_3RsL_vCvRabKY7BF1b6OSuYr?usp=sharing>`_ on Colab directly.
* 