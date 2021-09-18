# Evaluation

This folder contains protocols and raw data of our series of experiments.

Our experiments serve two goals:
First, to empirically evaluate our implementation and compare it to the paper.
Then, to empirically observe key difference of low-connectivity state space exploration.

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

### TODO

- EXP-14 Exploring DP/Anderson/Peterson using Depth-Limited process-PDS
