Data Management
===============

.. autoclass:: menten_gcn.DataHolder

   .. automethod:: append

   .. automethod:: assert_mode

   .. automethod:: save_to_file

   .. automethod:: load_from_file		   
		   
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

.. autoclass:: menten_gcn.DataHolderInputGenerator

   Example:
	       
   .. code-block:: python

      #Setup: (See above for get_data_from_poses)
      data_maker = make_datamaker() #Hand-wavy
      train_poses = [ "A.pdb", "B.pdb", "C.pdb" ]
      train_dataholder = get_data_from_poses( train_poses, data_maker )
      val_poses = [ "D.pdb", "E.pdb" ]
      val_dataholder = get_data_from_poses( val_poses, data_maker )

      #Important Part:
      train_generator = DataHolderInputGenerator( train_dataholder )
      val_generator = DataHolderInputGenerator( val_dataholder )
      model.fit( train_generator, validation_data=val_generator, epochs=100 )

   
.. autoclass:: menten_gcn.CachedDataHolderInputGenerator

   Example:
	       
   .. code-block:: python

      training_generator = CachedDataHolderInputGenerator( training_data_filenames, cache=False, batch_size=64 )
      validation_generator = CachedDataHolderInputGenerator( validation_data_filenames, cache=False, batch_size=64 )
      model.fit( training_generator, validation_data=validation_generator, epochs=1000, shuffle=False )
      # Note shuffle=False
      # CachedDataHolderInputGenerator does all shuffling internally to minimize disk access

   See below for a full example
   
	       

Full Example
############

.. include:: ../examples/SASA.rst
   :start-line: 3
	     
