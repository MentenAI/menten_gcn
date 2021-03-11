==============
Decorator Menu
==============

********
Geometry
********
      
.. autoclass:: menten_gcn.decorators.CACA_dist
   
   
.. autoclass:: menten_gcn.decorators.CBCB_dist
   
   
.. autoclass:: menten_gcn.decorators.PhiPsiRadians
   
   
.. autoclass:: menten_gcn.decorators.ChiAngleDecorator
   

.. autoclass:: menten_gcn.decorators.trRosettaEdges
   
   .. image:: https://www.pnas.org/content/pnas/117/3/1496/F1.large.jpg
      
.. autoclass:: menten_gcn.decorators.SimpleBBGeometry
   
   
.. autoclass:: menten_gcn.decorators.StandardBBGeometry
   

.. autoclass:: menten_gcn.decorators.AdvancedBBGeometry
   

********
Sequence
********

.. autoclass:: menten_gcn.decorators.Sequence
   

.. autoclass:: menten_gcn.decorators.DesignableSequence
   

.. autoclass:: menten_gcn.decorators.SequenceSeparation
   

.. autoclass:: menten_gcn.decorators.SameChain
   

*******
Rosetta
*******
      
.. autoclass:: menten_gcn.decorators.RosettaResidueSelectorDecorator


   Example:
	       
   .. code-block:: python
		   
      import menten_gcn as mg
      import menten_gcn.decorators as decs
      import pyrosetta

      pyrosetta.init()

      buried = pyrosetta.rosetta.core.select.residue_selector.LayerSelector()
      buried.set_layers( True, False, False )
      buried_dec = decs.RosettaResidueSelectorDecorator( selector=buried, description='<Layer select_core="true" />' )

      data_maker = mg.DataMaker( decorators=[ buried_dec ], edge_distance_cutoff_A=10.0, max_residues=30 )
      data_maker.summary()    

   Gives:
      
   .. code-block::
      
      Summary:

      2 Node Features:
      1 : 1 if the node is a focus residue, 0 otherwise
      2 : 1.0 if the residue is selected by the residue selector, 0.0 otherwise. User defined definition of the residue selector and how to reproduce it: <Layer select_core="true" />

      1 Edge Features:
      1 : 1.0 if the two residues are polymer-bonded, 0.0 otherwise

   Note that the additional features are due to the BareBonesDecorator, which is included by default 

.. autoclass:: menten_gcn.decorators.RosettaResidueSelectorFromXML
   

   Example:
	       
   .. code-block:: python

      import menten_gcn as mg
      import menten_gcn.decorators as decs
      import pyrosetta

      pyrosetta.init()
      xml = '''
      <RESIDUE_SELECTORS>
      <Layer name="surface" select_surface="true" />
      </RESIDUE_SELECTORS>
      '''
      surface_dec = decs.RosettaResidueSelectorFromXML( xml, "surface" )

      max_res=30
      data_maker = mg.DataMaker( decorators=[ surface_dec ], edge_distance_cutoff_A=10.0, max_residues=max_res )
      data_maker.summary()
      
   Gives:
      
   .. code-block::
      
      Summary:

      2 Node Features:
      1 : 1 if the node is a focus residue, 0 otherwise
      2 : 1.0 if the residue is selected by the residue selector, 0.0 otherwise. User defined definition of the residue selector and how to reproduce it: Took the residue selector named surface from this XML: 
      <RESIDUE_SELECTORS>
      <Layer name="surface" select_surface="true" />
      </RESIDUE_SELECTORS>


      1 Edge Features:
      1 : 1.0 if the two residues are polymer-bonded, 0.0 otherwise

   Note that the additional features are due to the BareBonesDecorator, which is included by default 

      
.. autoclass:: menten_gcn.decorators.RosettaJumpDecorator
   

.. autoclass:: menten_gcn.decorators.RosettaHBondDecorator
   

.. autoclass:: menten_gcn.decorators.Rosetta_Ref2015_OneBodyEneriges
   

.. autoclass:: menten_gcn.decorators.Rosetta_Ref2015_TwoBodyEneriges
   

.. autoclass:: menten_gcn.decorators.Ref2015Decorator
   
