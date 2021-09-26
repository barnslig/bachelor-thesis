# Low-Connectivity State Space Exploration using Swarm Model Checking on the GPU

This repository contains the source of my bachelor's thesis.

[Click here to download the writeup (PDF)](./writeup/main.pdf).

## Folder structure

- [writeup/](./writeup/) The LaTeX source of the writeup and intermediate presentation
- [implementation/](./implementation/) The source code of our Grapple model checker implementation
- [evaluation/](./evaluation/) Protocols and raw data of our experiments

## Abstract

Explicit verification of all states in a software or hardware system through model checking can be time-consuming due to the state explosion problem.
Swarm verification is an approach to tackle the state explosion problem by splitting the verification into many small, independent verification tests that can be massively parallelized, for example on the GPU.
Past work on the Grapple model checking framework has shown that low-connectivity models with only a single edge between each pair of states, or large portions of the state space hidden behind bottleneck structures, can cause a swarm model checker to significantly slow down in terms of unique states visited per verification test.
In this thesis, we provide an implementation of a Grapple model checker, a method for estimating unique states visited, an extension to the default Grapple search strategy called _start overs_ that can reach deeper states, and a visualization technique for breadth-first search frontiers.
Our experimental results show that our start over strategy can achieve a significant increase in state space coverage.
Our experiments on low-connectivity models shows promising characteristics that can potentially be used to identify low-connectivity models.

## License

See [LICENSE](./LICENSE).
