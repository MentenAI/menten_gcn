SASA Example
============

Let's say we want to create a model that predicts the solvent accessible surface area of a residue, given the residue and its surroundings.
A single pass of the network will only predict one residue (the focus residue) and will include up to 19 neighbor nodes.

We have tons of data (10000 pdb files, for example) local on disk:

>>> ls inputs/*
inputs/00001.pdb inputs/00002.pdb ... inputs/10000.pdb

Keep in mind that we're storing a single data point for every residue of these poses.
So if the average pose has 150 residues, we will end up with 10000 * 150 = 1.5 Million training points.
This will take a lot of memory to hold.
We should group this into, say, batches of 50 poses each

>>> ls inputs/* | shuf | split -dl 50 - list
>>> ls ./list*
list001 list002 ... list200
>>> # 200 makes sense right? 10000 / 50 = 200
>>> wc -l list001
50
>>> head list001
inputs/03863.pdb
inputs/00134.pdb
inputs/00953.pdb
inputs/02387.pdb
inputs/09452.pdb


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
list001.npz list002.npz ... list200.npz

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
   
   def make_model():
   
       """
       This is just a simple model
       Model building is not the point of this example
       """

       # Be sure to use the same data_maker configuration as before
       # Otherwise the tensor sizes may not be the same
       decorators = [ decs.StandardBBGeometry(), decs.Sequence() ]
       data_maker = menten_gcn.DataMaker( decorators=decorators, edge_distance_cutoff_A=10.0, max_residues=20 )

       
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
       assert len( sys.argv ) > 1, "Please pass the npz files as arguments"
       npznames = sys.argv[1:]

       # use 20% for validation
       fifth = int(len(data_list_lines)/5)
       training_data_filenames = npznames[fifth:]
       validation_data_filenames = npznames[:fifth]
       
       training_generator = menten_gcn.CachedDataHolderInputGenerator( training_data_filenames, cache=False, batch_size=64 )
       validation_generator = menten_gcn.CachedDataHolderInputGenerator( validation_data_filenames, cache=False, batch_size=64, autoshuffle=False ) #Note autoshuffle=False is recommended for validation data

       model = make_model()
       model.fit( training_generator, validation_data=validation_generator, epochs=1000, shuffle=False )
       model.save( "my_model.h5" )

      
       
>>> python3 train.py ./list*.npz
>>> ls *.h5
my_model.h5

Okay we're done!
So why did we deal with all that effort with caching on disk?

Your mileage may vary,
but I find that I end up with more data than can fit in my system's memory.
It's actually reasonably fast to just keep all of the data on disk and read it in each epoch, especially for you SSD users.

We were able to train this entire model with no more than two DataHolders loaded into memory at any given time. Given that we split our data into 200 DataHolders, this is a 100x decrease is memory usage!

