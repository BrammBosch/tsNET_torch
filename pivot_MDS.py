from tulip import tlp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def Pivot_MDS(file,pivots=250,use_edge_cost = False,edge_cost = 100):

    graph = tlp.loadGraph(file)
    print(graph)
    params = tlp.getDefaultPluginParameters('Pivot MDS (OGDF)', graph)
    params['number of pivots'] = pivots
    params['use edge costs'] = use_edge_cost
    params['edge costs'] = edge_cost
    resultLayout = graph.getLayoutProperty('resultLayout')

    graph.applyLayoutAlgorithm('Pivot MDS (OGDF)', resultLayout, params)

    nodes = []

    for i,n in enumerate(graph.getNodes()):
        nodes.append([resultLayout[n][0],resultLayout[n][1]])


    return nodes

