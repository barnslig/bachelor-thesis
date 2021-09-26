# Evaluation

This folder contains protocols and raw data of our series of experiments.

The first series of experiments empirically evaluates the correctness of our implementation and compares our implementation it to the paper.
The second series tries to empirically observe key difference of low-connectivity state space exploration.

The folder structure is as follows:

- [`data/`](./data/) contains raw data of the experiments.
- [`assets/`](./assets/) contains external assets, for example, a screenshotted plot from the Grapple paper.
- [`output-assets/`](./output-assets/) contains the PDFs of all plots and is used by the writeup.

## General evaluation of the implementation and comparison with the paper

- [EXP-00 Comparison with the Grapple Paper using the Waypoints Model](EXP-00-comparison-with-paper.ipynb)
- [EXP-01 Comparison of Waypoints and HLL as Indicators of State Space Coverage](EXP-01-comparison-waypoints-hll.ipynb)
- [EXP-02 Maximum Size Hash Table](EXP-02-large-hash-table.ipynb)
- [EXP-03 Multiple VTs Sharing a Kernel](EXP-03-shared-kernel.ipynb)
- [EXP-04 Improving State Space Coverage Growth using Start Overs](EXP-04-start-overs.ipynb)

## Low-Connectivity model evaluation

- [EXP-10 Comparison of Exploration of the Dining Philosophers Problem with Different Number of Processes](EXP-10-dining-philosophers-processes.ipynb)
- [EXP-11 Comparing Models by their BFS Frontiers: Relative and Absolute](EXP-11-bfs-frontiers.ipynb)
- [EXP-12 Comparing Models by Percentage of VTs Finding a Violation, Unique States Visited, Total States Visited, and State Space Coverage](EXP-12-model-comparison.ipynb)
- [EXP-13 Comparing the Start Over Strategy on Low-Connectivity Models](EXP-13-start-overs-low-connectivity.ipynb)

## Jupyter Environment

The experiments are evaluated in a [VSCode development container](https://code.visualstudio.com/docs/remote/create-dev-container) using [VSCode Jupyter Notebooks](https://code.visualstudio.com/docs/datascience/jupyter-notebooks).
They should, however, work in any miniconda environment.
