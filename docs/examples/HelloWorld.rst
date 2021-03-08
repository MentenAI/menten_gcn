Hello World
============

Let's start simple and just generate a single set of X, A, and E tensors

With PyRosetta
############

.. code-block:: python

   import pyrosetta
   pyrosetta.init()
   
   import menten_gcn as mg
   import menten_gcn.decorators as decs

   import numpy as np
      
   # Pick some decorators to add to your network
   decorators = [ decs.StandardBBGeometry(), decs.Sequence() ]

   data_maker = mg.DataMaker( decorators=decorators,
                              edge_distance_cutoff_A=10.0, # Create edges between all residues within 10 Angstroms of each other
			      max_residues=20,             # Do not include more than 20 residues total in this network
			      nbr_distance_cutoff_A=25.0 ) # Do not include any residue that is more than 25 Angstroms from the focus residue(s)

   data_maker.summary()
				      
   pose = pyrosetta.pose_from_pdb( "test.pdb" )
   wrapped_pose = mg.RosettaPoseWrapper( pose )

   #picking an arbitrary resid to be interested in
   resid_of_interest = 10
   
   X, A, E, resids = data_maker.generate_input_for_resid( wrapped_pose, resid_of_interest )

   # Sanity check:
   print( "X shape:", X.shape )
   print( "A shape:", A.shape )
   print( "E shape:", E.shape )
   print( "Resids in network:", resids )



With MDTraj
############

.. code-block:: python

   import mdtraj as md
   
   import menten_gcn as mg
   import menten_gcn.decorators as decs

   import numpy as np
      
   # Pick some decorators to add to your network
   decorators = [ decs.StandardBBGeometry(), decs.Sequence() ]

   data_maker = mg.DataMaker( decorators=decorators,
		              edge_distance_cutoff_A=10.0, # Create edges between all residues within 10 Angstroms of each other
			      max_residues=20,             # Do not include more than 20 residues total in this network
			      nbr_distance_cutoff_A=25.0 ) # Do not include any residue that is more than 25 Angstroms from the focus residue(s)

   data_maker.summary()
				      
   pose = md.load_pdb( "test.pdb" )
   wrapped_pose = mg.MDTrajPoseWrapper( pose )   

   #picking an arbitrary resid to be interested in
   resid_of_interest = 10
   
   X, A, E, resids = data_maker.generate_input_for_resid( wrapped_pose, resid_of_interest )

   # Sanity check:
   print( "X shape:", X.shape )
   print( "A shape:", A.shape )
   print( "E shape:", E.shape )
   print( "Resids in network:", resids )
   
