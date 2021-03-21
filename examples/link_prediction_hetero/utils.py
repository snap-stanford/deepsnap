import networkx as nx
from copy import deepcopy
from deepsnap.graph import Graph

def generate_convs(hete, conv, hidden_size, task='node'):
    convs1 = {}
    convs2 = {}
    for message_type in hete.message_types:
        n_type = message_type[0]
        s_type = message_type[2]
        n_feat_dim = hete.num_node_features(n_type)
        s_feat_dim = hete.num_node_features(s_type)
        if task == 'node':
            label_dim = hete.num_node_labels(s_type)
        elif task == 'link_pred':
            label_dim = 2
        convs1[message_type] = conv(n_feat_dim, hidden_size, s_feat_dim)
        convs2[message_type] = conv(hidden_size, label_dim, hidden_size)
    return convs1, convs2


def concatenate_citeseer_cora(cora_pyg, citeseer_pyg):
    cora = Graph.pyg_to_graph(cora_pyg)
    citeseer = Graph.pyg_to_graph(citeseer_pyg)
    cora_g = cora.G
    citeseer_g = citeseer.G
    nx.set_node_attributes(cora_g, 'cora_node', name='node_type')
    nx.set_edge_attributes(cora_g, 'cora_edge', name='edge_type')
    nx.set_node_attributes(citeseer_g, 'citeseer_node', name='node_type')
    nx.set_edge_attributes(citeseer_g, 'citeseer_edge', name='edge_type')

    G = deepcopy(cora_g)
    num_nodes_cora = cora_g.number_of_nodes()
    num_edges_cora = cora_g.number_of_edges()
    for i, node in enumerate(citeseer_g.nodes(data=True)):
        G.add_node(node[0] + num_nodes_cora, **node[1])
        assert (
            G.nodes[num_nodes_cora + i]['node_label']
            == citeseer_g.nodes[i]['node_label']
        )
        assert (
            G.nodes[num_nodes_cora + i]['node_type']
            == citeseer_g.nodes[i]['node_type']
        )
    assert (
        G.number_of_nodes()
        == cora_g.number_of_nodes() + citeseer_g.number_of_nodes()
    )

    for i, edge in enumerate(citeseer_g.edges(data=True)):
        u = edge[0] + num_nodes_cora
        v = edge[1] + num_nodes_cora
        G.add_edge(u, v, **edge[2])
        assert (
            G.edges[(u, v)]['edge_type']
            == citeseer_g.edges[(edge[0], edge[1])]['edge_type']
        )
    assert (
        G.number_of_edges()
        == cora_g.number_of_edges() + citeseer_g.number_of_edges()
    )
    return G
