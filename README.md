# The Next Generation Sequencing-Chess Problem
The development of Next Generation Sequencing (NGS) technologies paved the way for studying the spatiotemporal coordination of cellular processes along the genome. However, data sets are commonly limited to a few time points, and missing information needs to be interpolated. Most models assume that the studied dynamics are similar between individual cells, so that a homogeneous cell culture can be represented by a population-wide average. Here, we demonstrate that this understanding can be inappropriate. We developed a thought experiment--which we call the NGS chess problem--in which we compare the temporal sequencing data analysis to observing a superimposed picture of many independent games of chess at a time. The analysis of the spatiotemporal kinetics advocates for a new methodology that considers DNA-particle interactions in each cell independently even for a homogeneous cell population.

## Requirements and Installation
The code was run on `Python3.8` and `Python3.9`. Install the necessary packages via `pip` by running

```console
python3 -m pip install -r requirements.txt
```

We also provide the definition of a singularity container in the file `singularity_container.def`

## Exectuion
Run the top-down simulations by
```console
python3 experimentsChess.py
```

and the bottom-up simulations with
```console
python3 experimentsChessNN.py
```

Feel free to test your own parameters. The top-down models are implemented in `chess.py`. To know more about the parameters that are needed, run
```console
python3 chess.py --help
```

Similarly, the bottom-up model is implemented in the file `chessNN.py`. Check the parameters that you can set bu running
```console
python3 chess.NN.py --help
```

## Jupyter notebooks
We added two Jupyter notebooks that visualise some additional concepts. `Distributions.ipynb` provides an overview how different parameters influence the beta distribution, as well as distributions in the game graph.  `Performance Analysis.ipynb` loads and visualise the results of the simulations. Please note that this notebook is not stand-alone, and you need to produce the simulation results yourself in the first place. To do so, run the `experimentsChessNN.py` and the `experimentsChess.py` scripts.