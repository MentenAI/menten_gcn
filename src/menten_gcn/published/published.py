from menten_gcn import DataMaker
import menten_gcn.decorators as decs

import numpy as np

class Maguire_Grattarola_2021( DataMaker ):
    def __init__( self ):
        decorators = [ decs.Sequence(), decs.CACA_dist(), decs.PhiPsiRadians(),
               decs.ChiAngleDecorator(), decs.trRosettaEdges(),
               decs.SequenceSeparation(), decs.RosettaJumpDecorator(rottype="euler" ),
               decs.RosettaHBondDecorator(),
               decs.AbbreviatedRef2015Decorator_v0() ]
        DataMaker.__init__( self, decorators=decorators,
                            edge_distance_cutoff_A=15.0,
                            max_residues=30,
                            exclude_bbdec=False,
                            nbr_distance_cutoff_A=100,
                            dtype=np.float32 )

    def run_consistency_check( self ):
        N, F, S = self.get_N_F_S()
        print( N, F, S )
        
        try:
            import pyrosetta
            from pyrosetta import pose_from_sequence
            from menten_gcn import RosettaPoseWrapper
            
            pose = pose_from_sequence( "MENTENAI" )
            wrapped_pose = RosettaPoseWrapper( pose )
            X, A, E, meta = self.generate_input_for_resid( wrapped_pose, 1 )

            print( repr( X ) )
            print( repr( A ) )
            print( repr( E ) )
            print( repr( meta ) )            
        except:
            print("Unable to use pyrosetta. Did you call init?")
