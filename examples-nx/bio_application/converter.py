import pandas as pd
import networkx as nx
import torch
import torch_geometric.data
import torch_geometric.utils

#read BioSNAP File into Pandas
def readFilePD(fn,features = []):
    df = pd.read_csv(fn, sep='\t')
    #print(df.columns)
    for col in features: 
        df[col] = df[col].astype('category').cat.codes
    return df

#converts the bioSnap function-function dataset to a NetworkX graph
def bioSnapFFToNx(nodef,naIdx,edgef,eaIdx):
    dicty = {'cellular_component' : 0, 'biological_process': 1.0, 'molecular_function':2.0}
    f = open(nodef, "r")
    nodetups = []
    nodes = f.readlines()
    lbls = nodes[0].split('\t')
    nodes = nodes[1:]
    nodeAttr = lbls[naIdx]
    for node in nodes:
        node = node.split('\t')
        if node[naIdx] in dicty.keys() :
            nodetups.append((node[0],{nodeAttr : dicty[node[naIdx]], 'sample':1}))
    f.close()
    G = nx.DiGraph()
    G.add_nodes_from(nodetups)
    f = open(edgef,"r")
    edgetups = []
    edges = f.readlines()
    lbls = edges[0].replace('\n','').split('\t')
    edges = edges[1:]
    dictt = {'is_a' : 0, 'alt_id' : 1, 'disjoint_from' : 2, 'consider' : 3, 'intersection_of' : 4, 'relationship' : 5}
    edgeAttr = lbls[eaIdx]
    for edge in edges:
        edge = edge.replace('\n','').split('\t')
        if G.has_node(edge[0]) and G.has_node(edge[1]) and edge[eaIdx] in dictt.keys():
            edgetups.append((edge[0],edge[1],{edgeAttr : dictt[edge[eaIdx]]}))
    G.add_edges_from(edgetups) 
    return G

def testBSTN():
    g = bioSnapToNx('examples/minerf.tsv',3,'examples/minerff.tsv',2)
    print(nx.info(g))
    print(g['GO:0000001']['GO:0048308'])
    print(nx.get_node_attributes(g,'namespace'))
    print(nx.is_directed(g))

#faster function, converts PD DataFrame to NX graph
#only adds edges, so only works with 0 node attrs
def pdToNxSimple(d,c1,c2,attr):
    return nx.convert_matrix.from_pandas_edgelist(d,c1,c2,attr)

#slower function
def pdToNx2(d,d2,c1,c2,eattr,n,nattr):
    G = nx.Graph()
    nds = d2[n].tolist()
    nattrs = d2[nattr].tolist()
    nodes = []
    for i in range(len(nds)):
        dicty = {'node_label' : int(nattrs[i]),'node_feature' : float(1)}
        t = (nds[i],dicty)
        nodes.append(t)
    G.add_nodes_from(nodes)
    src = d[c1].tolist()
    dst = d[c2].tolist()
    attr = d[eattr].tolist()
    edges = []
    for i in range(len(src)):
        currData = attr[i]
        if G.has_node(src[i]) and G.has_node(dst[i]):
            dicty = {'edge_feature' : float(currData)}
            t = (src[i],dst[i],dicty)
            edges.append(t)
    G.add_edges_from(edges)
    return G

def pdToNxCC(d,d2,lbl = 'type',mask = []):
    listy = ['type','approved','nutraceutical','illicit','investigational','withdrawn','experimental']
    G = nx.Graph()
    nds = d2['id'].tolist()
    nattrs = {}
    for attr in listy:
        nattrs[attr] = d2[attr].tolist()
    nodes = []
    otherattrs = []
    for a in listy:
        if a != lbl and a not in mask:
            otherattrs.append(a)
    for i in range(len(nds)):
        dicty = {'node_label' : int(nattrs[lbl][i]),'node_feature' : torch.FloatTensor([float(nattrs[attr][i]) for attr in otherattrs])}
        t = (nds[i],dicty)
        nodes.append(t)
    G.add_nodes_from(nodes)
    src = d['srcid'].tolist()
    dst = d['dstid'].tolist()
    edges = []
    for i in range(len(src)):
        if G.has_node(src[i]) and G.has_node(dst[i]):
            t = (src[i],dst[i],{'edge_feature' : float(0)})
            edges.append(t)
    G.add_edges_from(edges)
    return G

def pdToNx3(d,d2,c1,c2,eattr,n,nattr):
    G = nx.Graph()
    nds = d2[n].tolist()
    nattrs = d2[nattr].tolist()
    nodes = []
    for i in range(len(nds)):
        dicty = {'node_feature' : float(nattrs[i])}
        t = (nds[i],dicty)
        nodes.append(t)
    G.add_nodes_from(nodes)
    src = d[c1].tolist()
    dst = d[c2].tolist()
    attr = d[eattr].tolist()
    edges = []
    for i in range(len(src)):
        currData = attr[i]
        if G.has_node(src[i]) and G.has_node(dst[i]):
            dicty = {'edge_label' : int(currData), 'edge_feature' : float(1)}
            t = (src[i],dst[i],dicty)
            edges.append(t)
    G.add_edges_from(edges)
    return G

def NxToPyG(x):
    return torch_geometric.utils.from_networkx(x)

def tests():
    f = 'miner/minerff.tsv'
    f2 = 'miner/minerf.tsv'
    d = readFilePD(f,['relation'])
    d2 = readFilePD(f2,['namespace'])
    n = pdToNx2(d,d2,'GO_id0','GO_id2','relation','GO_id1','namespace')
    #print(nx.get_node_attributes(n,'namespace'))
    #print(nx.get_edge_attributes(n,'relation'))
    #print(nx.info(n))
#tests()
