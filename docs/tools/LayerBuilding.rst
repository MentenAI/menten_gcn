Model Building
==============

We recommend using tools like Spektral to build your model.
That said, here are some utilities that we found useful

Masks
#####

.. autofunction:: menten_gcn.util.make_and_apply_node_mask

.. autofunction:: menten_gcn.util.make_and_apply_edge_mask

Example:

.. code-block:: python

   # Setup
   X_in, A_in, E_in = data_maker.generate_XAE_input_tensors()
   X = X_in
   A = A_in
   E = E_in

   # Phase 1 - preprocess
   for i in [128,64]:
       # Conv1D and Conv2D are keras layers
       X = Conv1D( i, kernel_size=1, activation='relu' )( X )
       E = Conv2D( i, kernel_size=1, activation='relu' )( E )
   X = menten_gcn.util.make_and_apply_node_mask( X, A )
   E = menten_gcn.util.make_and_apply_edge_mask( E, A )

   # continue with more layers...

.. autofunction:: menten_gcn.util.make_node_mask

.. autofunction:: menten_gcn.util.apply_node_mask		

.. autofunction:: menten_gcn.util.make_edge_mask

.. autofunction:: menten_gcn.util.apply_edge_mask		
		
Convolutions
############

.. autofunction:: menten_gcn.playground.make_NENE_XE_conv

.. autofunction:: menten_gcn.playground.make_NEENEENEE_XE_conv
