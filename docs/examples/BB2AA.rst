Backbone-to-Seq
===============

Okay here our goal is to predict the native amino acid for a given position based solely on the backbone.
A single pass of the network will only predict one residue (the focus residue) and will include up to 29 neighbor nodes.

Like the other example, let's say
we have 10000 pdb files on disk for training

>>> ls inputs/*
inputs/00001.pdb inputs/00002.pdb ... inputs/10000.pdb

What's different about this example is that we will attempt to fit all of them in memory
Let's make two pdblists, one for training and one for testing

>>> ls ./*_pdblist
train_pdblist validation_pdblist

And now it's just a one-stop-shop to get everything done

.. code-block:: python

   # train.py
		
   import pyrosetta
   pyrosetta.init()

   import menten_gcn
   import menten_gcn.decorators as decs

   from spektral.layers import *
   from tensorflow.keras.layers import *
   from tensorflow.keras.models import Model
   
   import numpy as np

   def make_data( data_maker, pdblist: str ):
       #listname is list001 or list002 or so on

       dataholder = menten_gcn.DataHolder()
       
       #we're going to use a decorator for output too
       seq = decs.Sequence()

       pdblist = open( listname, "r" )
       for line in pdblist:

           pose = pose_from_pdb( filename.rstrip() )
	   wrapped_pose = RosettaPoseWrapper( pose )
	   cache = data_maker.make_data_cache( wrapped_pose )

	   sasa_results = sasa_calc.calculate( pose )
	   
	   for resid in range( 1, pose.size() + 1 ):
		X, A, E, meta = data_maker.generate_input_for_resid( wrapped_pose, resid, data_cache=cache )
		# useful tip, `len(meta)-1` is the number of neighbors
		out = seq.calc_node_features( wrapped_pose, resid )
                dataholder.append( X=X, A=A, E=E, out=out )

       return dataholder

   def make_model( data_maker ):
   
       """
       This is just a simple model
       Model building is not the point of this example
       """
       
       X_in, A_in, E_in = data_maker.generate_XAE_input_layers()
       X1 = EdgeConditionedConv( 30, activation='relu' )([X_in, A_in, E_in])
       X2 = EdgeConditionedConv( 30, activation='relu' )([X1, A_in, E_in])
       FinalPool = GlobalSumPool()(X2)
       output = Dense( 1, name="out" )(FinalPool)

       model = Model(inputs=[X_in,A_in,E_in], outputs=output)
       model.compile(optimizer='adam', loss='mean_squared_error' )
       model.summary()

       return model
       
   if __name__ == '__main__':

       decorators = [ decs.AdvancedBBGeometry(), decs.SequenceSeperation() ]
       data_maker = menten_gcn.DataMaker( decorators=decorators, edge_distance_cutoff_A=10.0, max_residues=30 )
       data_maker.summary()
   
       train_data = make_data( data_maker, "./train_pdblist" )
       val_data = make_data( data_maker, "./validation_pdblist" )

       train_generator = mg.DataHolderInputGenerator( train_data )
       val_generator = mg.DataHolderInputGenerator( val_data )
       
       model = make_model( data_maker )
       model.fit( train_generator, validation_data=validation_generator, epochs=1000, shuffle=True )
       #Note shuffle=True when we're using the DataHolderInputGenerator instead of the CachedDataHolderInputGenerator
       
       model.save( "my_model.h5" )

>>> python3 train.py
>>> ls *.h5
my_model.h5
