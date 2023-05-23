#!/usr/bin/env python3

import os
import time
import networkx as nx
import modules.layout_io as layout_io
import modules.graph_io as graph_io
import modules.thesne as thesne
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pivot_MDS import Pivot_MDS

if __name__ == '__main__':
    parser = ArgumentParser(description='Read a graph, and produce a layout with tsNET(*).')

    # Input
    parser.add_argument('input_graph')
    parser.add_argument('--star', action='store_true', help='Use the tsNET* scheme. (Requires PivotMDS layout in ./pivotmds_layouts/ as initialization.)\nNote: Use higher learning rates for larger graphs, for faster convergence.')
    parser.add_argument('--perplexity', '-p', type=float, default=40, help='Perplexity parameter.')
    parser.add_argument('--learning_rate', '-l', type=float, default=50, help='Learning rate (hyper)parameter for optimization.')
    parser.add_argument('--output', '-o', type=str, help='Save layout to the specified file.')


    args = parser.parse_args()




    # Check for valid input
    assert(os.path.isfile(args.input_graph))
    graph_name = os.path.splitext(os.path.basename(args.input_graph))[0]


    # Global hyperparameters
    n = 5000  # Maximum #iterations before giving up
    momentum = 0.5
    tolerance = 1e-7
    window_size = 40

    # Cost function parameters
    r_eps = 0.05

    # Phase 2 cost function parameters
    lambdas_2 = [1, 1.2, 0]
    if args.star:
        lambdas_2[1] = 0.1

    # Phase 3 cost function parameters
    lambdas_3 = [1, 0.01, 0.6]

    # Read input graph
    extension = os.path.splitext(args.input_graph)[1]
    if extension == '.vna':
        g = graph_io.load_graph(args.input_graph)
    elif extension == '.graphml':
        g = nx.read_graphml(args.input_graph)
        g = nx.convert_node_labels_to_integers(g)


    print('Input graph: {0}, (|V|, |E|) = ({1}, {2})'.format(graph_name, g.number_of_nodes(), g.number_of_edges()))

    # # Load the PivotMDS layout for initial placement
    if args.star:
        if not os.path.isfile('./pivotmds_layouts/{0}.vna'.format(graph_name)):
            try:
                Y_init = Pivot_MDS(args.input_graph)
            except Exception as e:
                print("Make sure either a VNA pivot_mds layout is provided or the graph file is graphml")
                quit()
        else:
            _, Y_init = layout_io.load_layout('./pivotmds_layouts/{0}.vna'.format(graph_name))
    else:
        Y_init = None

    # Time the method including SPDM calculations
    start_time = time.time()

    # Compute the shortest-path distance matrix.
    X = nx.floyd_warshall_numpy(g)
    # if there are disconnected components then some distances are inf, resolve this

    X[X == np.inf] = 10 * np.max(X)

    # normalize gtds
    X /= np.max(X)

    # The actual optimization is done in the thesne module.
    Y = thesne.tsnet(
        X, output_dims=2, random_state=1, perplexity=args.perplexity, n_epochs=n,
        Y=Y_init,
        initial_lr=args.learning_rate, final_lr=args.learning_rate, lr_switch=n // 2,
        initial_momentum=momentum, final_momentum=momentum, momentum_switch=n // 2,
        initial_l_kl=lambdas_2[0], final_l_kl=lambdas_3[0], l_kl_switch=n // 2,
        initial_l_c=lambdas_2[1], final_l_c=lambdas_3[1], l_c_switch=n // 2,
        initial_l_r=lambdas_2[2], final_l_r=lambdas_3[2], l_r_switch=n // 2,
        r_eps=r_eps, autostop=tolerance, window_size=window_size,
        verbose=True
    )

    Y = layout_io.normalize_layout(Y)

    end_time = time.time()
    comp_time = end_time - start_time
    print('tsNET took {0:.2f} s.'.format(comp_time))


    # Show layout on the screen
    pos = {}
    for i,coords in enumerate(Y):
        pos[i] = Y[i]
    nx.draw(g,pos=pos)
    plt.show()

    if args.output is not None:
        layout_io.save_vna(args.output, g, Y)
        print('Saved layout data in "{}"'.format(args.output))
