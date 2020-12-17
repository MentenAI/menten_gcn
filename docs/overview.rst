=====
Overview
=====

The goal of Menten GCN is to create GCN tensors from protein models (poses).
We are aligning with Spektral's vocabulary style when talking about GCNs
and Rosetta's vocabulary when talking about poses.

Prerequisite Knowledge
######################

This overview assumes general familiarity with
protein structure,
machine learning,
graph data structures,
and graph neural networks (to a small extent).


This is a very niche intersections of fields
so do not feel bad if you are not up to speed on all of these topics.
If you find yourself in that position, here are some links that might help:

Protein Structure:

- `Wikipedia <https://en.wikipedia.org/wiki/Protein_structure>`_
- `Amino Acids <https://en.wikipedia.org/wiki/Amino_acid>`_
- `Phi and Psi <https://proteopedia.org/wiki/index.php/Phi_and_Psi_Angles>`_
- `Rosetta <https://www.rosettacommons.org/support/overview>`_

Machine Learning:

- `Machine Learning <https://en.wikipedia.org/wiki/Machine_learning>`_
- `Neural Networks <https://en.wikipedia.org/wiki/Artificial_neural_network>`_
- `Graph Convolutional Neural Networks <https://tkipf.github.io/graph-convolutional-networks/>`_
- `Spektral <https://graphneural.network/>`_


Graph Layout
###############

Each node (vertex) in our graph represents a single residue position.
Edges connect nodes that are close in 3D space.

