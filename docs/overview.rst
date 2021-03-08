========
Overview
========

The goal of Menten GCN is to create GCN tensors from protein models (poses).
We are aligning with Spektral's vocabulary style when talking about GCNs
and Rosetta's vocabulary when talking about poses.


Graph Layout
###############

Each node (vertex) in our graph represents a single residue position.
Edges connect nodes that are close in 3D space.
Our goal in Menten GCN is to analyze small pockets of residues at a time,
though the size of each pocket is entirely up to the user and can encompass the entire protein if you wish.

We generate a graph by first declaring one or more "focus" residues.
These residues will be at the center of our pocket.
Menten GCN will automatically select the residue positions closest in space
to the focus residues and will use them to build neighbor nodes.
Menten GCN will also automatically add edges between any two nodes that are close in space.

.. image:: _images/MentenGCN1.png

Graph Tensors
#############

We have 3 primary parameters in this system:

- "N" is maximum the number of nodes in any graph.
  This includes focus nodes and neighbor nodes
- "F" is the number of features per node
- "S" is the number of features per edge  
  
These parameters are used to define 3 input tensors:

- Tensor "X" holds the node features and is of shape (N,F)
- Tensor "A" holds the adjacency matrix and is of shape (N,N)
- Tensor "E" holds the edge features and is of shape (N,N,S)

One nuance of the "E" tensor is that edges can have direction.
Every pair of residues has room for two edge tensors in our system.
Some of our edge features are symmetric (like distance) so they will
have the same value going in both directions.
Other edge tensors are asymmetric (like relative geometries) so they
will have different values for each of the two slots in "E".

.. image:: _images/MentenGCNXEij.png

Usage
#####

.. image:: _images/MentenGCNOverview.png

1. Start by loading your pose in python using any of our supported packages.

   - Just Rosetta and MDTraj right now. Get in touch if you want more!
    
2. Wrap your pose using the appropriate wrapper for your package.

   - See Classes -> Pose Wrappers
    
3. Define a list of decorators to use to represent your pose.

   - See Classes -> Decorators
   - An example decorator would be PhiPsiRadians,
     which decorates each node with its Phi and Psi value
    
4. Use this list of decorators to build a DataMaker
   
5. The DataMaker will then take your wrapped pose, ask for the focus residues, and return the X, A, and E tensors
   
6. From here you have a few choices.

   - You can train on these tensors directly
   - You can utilize Spektral's Dataset interface to make training easier with large amounts of data
   - Or you can save these for later. Stick them on disk and come back to them when you're ready to train


See the DataMaker class and examples for more details.
