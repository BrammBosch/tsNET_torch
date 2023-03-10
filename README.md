# tsNET

This torch implementation is direct translation of the original theano version (https://github.com/HanKruiger/tsNET)

Graph Layouts by t-SNE

```
usage: tsnet.py [-h] [--star] [--perplexity PERPLEXITY]
                [--learning_rate LEARNING_RATE] [--output OUTPUT]
                input_graph

Read a graph, and produce a layout with tsNET(*).

positional arguments:
  input_graph

optional arguments:
  -h, --help            show this help message and exit
  --star                Use the tsNET* scheme. (Requires PivotMDS layout in
                        ./pivotmds_layouts/ as initialization.) Note: Use
                        higher learning rates for larger graphs, for faster
                        convergence.
  --perplexity PERPLEXITY, -p PERPLEXITY
                        Perplexity parameter.
  --learning_rate LEARNING_RATE, -l LEARNING_RATE
                        Learning rate (hyper)parameter for optimization.
  --output OUTPUT, -o OUTPUT
                        Save layout to the specified file.
```

Example:
```bash
# Read the input graph dwt_72, and save the output in ./output.vna
./tsnet.py graphs/dwt_72.vna --output ./output.vna
```

# Dependencies

* `python3`
* [`numpy`](http://www.numpy.org/)
* [`matplotlib`](https://matplotlib.org/)
* [`networkx`](https://networkx.org/)
* [`torch`](https://pytorch.org/)
* [`tulip`](https://tulip.labri.fr/Documentation/current/tulip-python/html/index.html)
