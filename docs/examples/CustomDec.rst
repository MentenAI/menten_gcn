Custom Decorator
================

We try and make it relatively easy to create your own decorator.
All you need to do is inherit a these seven methods from the base class.

(see full base class description at the bottom)

.. code-block:: python

   import menten_gcn as mg
   import menten_gcn.decorators as decs

    class TestDec( decs.Decorator ):

        def get_version_name( self ):
            return "TestDec"


        # NODES #

        def n_node_features( self ):
            return 1

        def calc_node_features( self, wrapped_protein, resid, dict_cache=None ):
            if wrapped_protein.get_name1( resid ) == "G":
                return [ 1.0 ]
            else:
                return [ 0.0 ]

        def describe_node_features( self ):
            return [ "1 if the residue is GLY, 0 otherwise" ]


        # EDGES #

        def n_edge_features( self ):
            return 2

        def calc_edge_features( self, wrapped_protein, resid1, resid2, dict_cache=None ):
            diff = resid2 - resid1
            same = 1.0 if wrapped_protein.get_name1( resid1 ) == wrapped_protein.get_name1( resid2 ) else 0.0
            return [ diff, same ], [ -diff, same ]

        def describe_edge_features( self ):
            return [ "Measures the distance in sequence space between the two residues", "1 if the two residues have the same amino acid, 0 otherwise" ]



    decorators=[ decs.SimpleBBGeometry(), TestDec(), ]
    data_maker = mg.DataMaker( decorators=decorators, edge_distance_cutoff_A=10.0, max_residues=5 )
    data_maker.summary()


.. code-block::

   Summary:

   4 Node Features:
   1 : 1 if the node is a focus residue, 0 otherwise
   2 : Phi of the given residue, measured in radians. Spans from -pi to pi
   3 : Psi of the given residue, measured in radians. Spans from -pi to pi
   4 : 1 if the residue is GLY, 0 otherwise

   4 Edge Features:
   1 : 1.0 if the two residues are polymer-bonded, 0.0 otherwise
   2 : Euclidean distance between the two CB atoms of each residue, measured in Angstroms. In the case of GLY, use an estimate of ALA's CB position
   3 : Measures the distance in sequence space between the two residues
   4 : 1 if the two residues have the same amino acid, 0 otherwise

.. autoclass:: menten_gcn.decorators.Decorator

   .. automethod:: n_node_features

   .. automethod:: describe_node_features

   .. automethod:: calc_node_features
		   
   .. automethod:: n_edge_features

   .. automethod:: describe_edge_features

   .. automethod:: calc_edge_features

   .. automethod:: get_version_name

   .. automethod:: cache_data
	       

For reference, here are the methods for the WrappedPose that will be passed into your decorator
		   
.. autoclass:: menten_gcn.WrappedPose
   :members:
