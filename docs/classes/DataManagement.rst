Data Management
==========

.. autoclass:: menten_gcn.DataHolder

   Example:
	       
   .. code-block:: python

      def get_data_from_poses( pose_filenames, data_maker: menten_gcn.DataMaker ):
      
          """
	  This function will load in many poses from disk and store their GCN tensors.
	  Some parts of this will look different for you.
	  In this case, we are making N graphs per pose where N is the number of residues in that pose.
	  Each residue is the center of one graph.
	  This is just an example, yours may look different.
	  The point is that we are making many X, A, E, and out tensors and storing them in the DataHolder.
	  """

      
          dataholder = menten_gcn.DataHolder()
          for filename in pose_filenames:
	      pose = pose_from_pdb( filename ) #Rosetta
	      wrapped_pose = RosettaPoseWrapper( pose=pose )
	      for resid in range( 1, pose.size() + 1 ):
		   X, A, E, meta = data_maker.generate_input_for_resid( wrapped_pose, resid )
		   out = foo() #what should the output of the network be?
                   dataholder.append( X=X, A=A, E=E, out=out )

	  #optionally create an npz
	  dataholder.save_to_file( "gcn_data" ) #creates gcn_data.npz
		   
	  return dataholder

   See below for a full example


.. autoclass:: menten_gcn.DecoratorDataCache

   Example:
	       
   .. code-block:: python

      def get_data_from_poses( pose_filenames, data_maker: menten_gcn.DataMaker ):
          """
	  This function will load in many poses from disk and store their GCN tensors.
	  Some parts of this will look different for you.
	  In this case, we are making N graphs per pose where N is the number of residues in that pose.
	  Each residue is the center of one graph.
	  This is just an example, yours may look different.
	  The point is that we are making many X, A, E, and out tensors and storing them in the DataHolder.
	  """
	  
          dataholder = menten_gcn.DataHolder()
          for filename in pose_filenames:
	      pose = pose_from_pdb( filename ) #Rosetta
	      wrapped_pose = RosettaPoseWrapper( pose=pose )
	      cache = data_maker.make_data_cache( wrapped_pose )
	      
	      for resid in range( 1, pose.size() + 1 ):
		   X, A, E, meta = data_maker.generate_input_for_resid( wrapped_pose, resid, data_cache = cache )
		   out = foo() #what should the output of the network be?
                   dataholder.append( X=X, A=A, E=E, out=out )

	  #optionally create an npz
	  dataholder.save_to_file( "gcn_data" ) #creates gcn_data.npz
		   
	  return dataholder

   See below for a full example

	       
.. autoclass:: menten_gcn.CachedDataHolderInputGenerator

   Example:
	       
   .. code-block:: python

      training_generator = CachedDataHolderInputGenerator( training_data_filenames, cache=False, batch_size=64 )
      validation_generator = CachedDataHolderInputGenerator( validation_data_filenames, cache=False, batch_size=64 )
      model.fit( training_generator, validation_data=validation_generator, epochs=1000, shuffle=False, use_multiprocessing=False, callbacks=callbacks )
      # Note shuffle=False
      # CachedDataHolderInputGenerator does all shuffling internally to minimize disk access

   See below for a full example


Full Example
############

Let's say we want to create a model that predicts the solvent accessible surface area of a residue, given the residue and its surroundings.

We have tons of data (the entire PDB, for example) local on disk:

>>> ls inputs/*
inputs/A00001.pdb inputs/A00002.pdb ... inputs/A99999.pdb

This will take a lot of memory to hold.
We should group this into, say, batches of 100 poses each

>>> ls inputs/* | shuf | split -dl 100 - list
>>> ls ./list*
list001 list002 ... list100

We're then going to feed each list into:

.. code-block:: python

   # Let's call this make_data.py
		
   import pyrosetta
   pyrosetta.init()

   import menten_gcn
   import menten_gcn.decorators as decs

   import numpy as np

   def run( listname: str ):
       #listname is list001 or list002 or so on

       dataholder = menten_gcn.DataHolder()
       
       decorators = [ decs.StandardBBGeometry(), decs.Sequence() ]
       data_maker = menten_gcn.DataMaker( decorators=decorators, edge_distance_cutoff_A=10.0, max_residues=20 )
       data_maker.summary()
       
       sasa_calc = pyrosetta.rosetta.core.simple_metrics.per_residue_metrics.PerResidueSasaMetric()
       
       listfile = open( listname, "r" )
       for line in listfile:

           pose = pose_from_pdb( filename.rstrip() )
	   wrapped_pose = RosettaPoseWrapper( pose )
	   cache = data_maker.make_data_cache( wrapped_pose )

	   sasa_results = sasa_calc.calculate( pose )
	   
	   for resid in range( 1, pose.size() + 1 ):
		X, A, E, meta = data_maker.generate_input_for_resid( wrapped_pose, resid, data_cache=cache )
		out = np.asarray( [ sasa_results[ resid ] ] )
                dataholder.append( X=X, A=A, E=E, out=out )

       dataholder.save_to_file( listname ) #creates list001.npz, for example

       # this is a good time for "del dataholder" and garbage collection

   if __name__ == '__main__':
       assert len( sys.argv ) == 2, "Please pass the list file name as the one and only argument"
       listname = sys.argv[ 1 ]
       run( listname )


>>> ls ./list* | xargs -n1 python3 make_data.py
>>> # ^ run xargs -n1 -P N python3 make_data.py to run this in parallel on N processors
>>> ls ./list*.npz
list001.npz list002.npz ... list100.npz

Okay now we have all of our training data on disk.
Let's train

.. code-block:: python

   # train.py

   from spektral.layers import *
   from tensorflow.keras.layers import *
   from tensorflow.keras.models import Model
   
   import menten_gcn
   import menten_gcn.decorators as decs

   import numpy as np
   
   def make_model( data_maker ):
   
       """
       This is just a simple model
       Model building is not the point of this example
       """
       
       X_in, A_in, E_in = data_maker.generate_XAE_input_tensors()
       X1 = EdgeConditionedConv( 30, activation='relu' )([X_in, A_in, E_in])
       X2 = EdgeConditionedConv( 30, activation='relu' )([X1, A_in, E_in])
       FinalPool = GlobalSumPool()(X2)
       output = Dense( 1, name="out" )(FinalPool)

       model = Model(inputs=[X_in,A_in,E_in], outputs=output)
       model.compile(optimizer='adam', loss='mean_squared_error' )
       model.summary()

       return model

   if __name__ == '__main__':
       assert len( sys.argv ) > 1, "Please pass the npz files as arguments"
       npznames = sys.argv[1:]

       # use 20% for validation
       fifth = int(len(data_list_lines)/5)
       training_data_filenames = npznames[fifth:]
       validation_data_filenames = npznames[:fifth]
       
       training_generator = menten_gcn.CachedDataHolderInputGenerator( training_data_filenames, cache=False, batch_size=64 )
       validation_generator = menten_gcn.CachedDataHolderInputGenerator( validation_data_filenames, cache=False, batch_size=64 )
       model.fit( training_generator, validation_data=validation_generator, epochs=1000, shuffle=False, use_multiprocessing=False, callbacks=callbacks )

      
       
>>> python3 train.py ./list*.npz
