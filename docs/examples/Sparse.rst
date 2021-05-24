Sparse Mode
===========

This modification of the "Simple Train" example
utilizes Spektral's disjoint mode to
model a sparse representation of the graph.

This can result in lower memory usage depending on the connectivity of your graph.

The key differences are:

- data_maker.generate_graph_for_resid has sparse=True

- data_maker.generate_XAE_input_layers has sparse=True and returns a 4th input

  - inputs=[X_in,A_in,E_in,I_in] when building the model

- We are making a Spektral Dataset and feeding it into the DisjointLoader

- We are using a Spektral Graph instead of freefloating lists. This change can be done with dense mode too.

  - 'y' is the output value in spektral graphs
  
    - Please read Spektral's documentation for options regarding 'y'
  
  
.. code-block:: python

   import pyrosetta
   pyrosetta.init()
   
   import menten_gcn as mg
   import menten_gcn.decorators as decs

   from spektral.layers import *
   from tensorflow.keras.layers import *
   from tensorflow.keras.models import Model
   
   import numpy as np
      
   decorators = [ decs.StandardBBGeometry(), decs.Sequence() ]

   data_maker = mg.DataMaker( decorators=decorators,
                              edge_distance_cutoff_A=10.0, # Create edges between all residues within 10 Angstroms of each other
			      max_residues=20,             # Do not include more than 20 residues total in this network
			      nbr_distance_cutoff_A=25.0 ) # Do not include any residue that is more than 25 Angstroms from the focus residue(s)

   data_maker.summary()

   class MyDataset(spektral.data.dataset.Dataset):
       def __init__(self, **kwargs):
           self.graphs = []
	   spektral.data.dataset.Dataset.__init__(self, **kwargs)

       def read(self):
           return self.graphs
	   
   dataset = MyDataset()
      
   for pdb in [ "test1.pdb", "test2.pdb", "test3.pdb", "test4.pdb", "test5.pdb" ]:
   
       pose = pyrosetta.pose_from_pdb( pdb )
       wrapped_pose = mg.RosettaPoseWrapper( pose )
       cache = data_maker.make_data_cache( wrapped_pose )
       
       for resid in range( 1, pose.size() + 1 ):
           g, resids = data_maker.generate_graph_for_resid( wrapped_pose, resid, data_cache=cache, sparse=True )

	   # for the sake of keeping this simple, let's have this model predict if this residue is an N-term
	   if wrapped_pose.resid_is_N_term( resid ):
		g.y = [1.0,]
	   else:
		g.y = [0.0,]
	
	   dataset.graphs.append( g )
	   
   # Note we have a 4th input now
   X_in, A_in, E_in, I_in = data_maker.generate_XAE_input_layers( sparse=True )

   # ECCConv is called EdgeConditionedConv in older versions of spektral
   L1 = ECCConv( 30, activation='relu' )([X_in, A_in, E_in])
   # Try this if the first one fails:
   #L1 = EdgeConditionedConv( 30, activation='relu' )([X_in, A_in, E_in])
   
   L2 = GlobalSumPool()(L1)
   L3 = Flatten()(L2)
   output = Dense( 1, name="out" )(L3)

   # Make sure to include the 4th input because the DisjointLoader will pass it
   model = Model(inputs=[X_in,A_in,E_in,I_in], outputs=output)
   model.compile(optimizer='adam', loss='binary_crossentropy' )
   model.summary()

   loader = spektral.data.loaders.DisjointLoader(dataset)
   model.fit(loader.load(), steps_per_epoch=loader.steps_per_epoch)
   # This part can sometimes fail due to tensorflow / numpy versioning.
   # See the troubleshooting page of our documentation for details
