Simple Train
============

This model builds off of the hello world
but has some extra complexity and takes us all the way to training

.. code-block:: python

   import pyrosetta
   pyrosetta.init()
   
   import menten_gcn as mg
   import menten_gcn.decorators as decs

   from spektral.layers import *
   from tensorflow.keras.layers import *
   from tensorflow.keras.models import Model
   
   import numpy as np
      
   # Pick some decorators to add to your network
   decorators = [ decs.StandardBBGeometry(), decs.Sequence() ]

   data_maker = mg.DataMaker( decorators=decorators,
                              edge_distance_cutoff_A=10.0, # Create edges between all residues within 10 Angstroms of each other
			      max_residues=20,             # Do not include more than 20 residues total in this network
			      nbr_distance_cutoff_A=25.0 ) # Do not include any residue that is more than 25 Angstroms from the focus residue(s)

   data_maker.summary()

   Xs = []
   As = []
   Es = []
   outs = []

   # This part is all very hand-wavy
   for pdb in [ "test1.pdb", "test2.pdb", "test3.pdb", "test4.pdb", "test5.pdb" ]:
   
       pose = pyrosetta.pose_from_pdb( pdb )
       wrapped_pose = mg.RosettaPoseWrapper( pose )
       cache = data_maker.make_data_cache( wrapped_pose )
       
       for resid in range( 1, pose.size() + 1 ):
           X, A, E, resids = data_maker.generate_input_for_resid( wrapped_pose, resid, data_cache=cache )
	   Xs.append( X )
	   As.append( A )
	   Es.append( E )

	   # for the sake of keeping this simple, let's have this model predict if this residue is an N-term
	   if wrapped_pose.resid_is_N_term( resid ):
		outs.append( [1.0,] )
	   else:
		outs.append( [0.0,] )

   # Okay now we need to define a model.
   # The data_maker can tell use the right sizes to use.
   # Better yet, the data_maker can simply create the input layers for us:
   X_in, A_in, E_in = data_maker.generate_XAE_input_layers()

   # GCN model architectures are tricky
   # Here's just a very simple one to get us off the ground

   # ECCConv is called EdgeConditionedConv in older versions of spektral
   L1 = ECCConv( 30, activation='relu' )([X_in, A_in, E_in])
   # Try this if the first one fails:
   #L1 = EdgeConditionedConv( 30, activation='relu' )([X_in, A_in, E_in])
   
   L2 = GlobalSumPool()(L1)
   L3 = Flatten()(L2)
   output = Dense( 1, name="out" )(L3)

   model = Model(inputs=[X_in,A_in,E_in], outputs=output)
   model.compile(optimizer='adam', loss='binary_crossentropy' )
   model.summary()
   
   Xs = np.asarray( Xs )
   As = np.asarray( As )
   Es = np.asarray( Es )
   outs = np.asarray( outs )

   print( Xs.shape )
   print( As.shape )
   print( Es.shape )
   print( outs.shape )

   model.fit( x=[Xs,As,Es], y=outs, batch_size=32, epochs=10, validation_split=0.2 )
