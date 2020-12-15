import numpy as np
import math

try:
    from pyrosetta import rosetta
except:
    rosetta = None

try:
    import mdtraj as md
except:
    md = None

'''
try:
    import Bio as bp
except:
    bp = None
'''
    
#TODO just do this all the time for CB
def estimate_CB_xyz( C_xyz, N_xyz, CA_xyz ):
    #ICOOR_INTERNAL    CB  -122.800000   69.625412    1.521736   CA    N     C
    '''
    That is, CB is found 1.52 Ang from CA, at an angle of 70 degrees from the CA-N line, and a dihedral of -123 degrees for CB-CA-N-C.
    -ROCCO
    '''

    #Credit to Rohit Bhattacharya from https://github.com/rbhatta8/protein-design/blob/master/nerf.py
    '''
    Nerf method of finding 4th coord (d)
    in cartesian space
    Params:
    a, b, c : coords of 3 points
    l : bond length between c and d
    theta : bond angle between b, c, d (in degrees)
    chi : dihedral using a, b, c, d (in degrees)
    Returns:
    d : tuple of (x, y, z) in cartesian space
    '''

    l = 1.521736
    theta = 69.625412
    chi = -122.800000

    a = C_xyz
    b = N_xyz
    c = CA_xyz
    
    # calculate unit vectors AB and BC
    ab_unit = (b-a)/np.linalg.norm(b-a)
    bc_unit = (c-b)/np.linalg.norm(c-b)

    # calculate unit normals n = AB x BC
    # and p = n x BC
    n_unit = np.cross(ab_unit, bc_unit)
    n_unit = n_unit/np.linalg.norm(n_unit)
    p_unit = np.cross(n_unit, bc_unit)

    # create rotation matrix [BC; p; n] (3x3)
    M = np.array([bc_unit, p_unit, n_unit]).T

    # convert degrees to radians
    theta = np.pi/180 * theta
    chi = np.pi/180 * chi

    # calculate coord pre rotation matrix
    d2 = [-l*np.cos(theta), l*np.sin(theta)*np.cos(chi), l*np.sin(theta)*np.sin(chi)]

    # calculate with rotation as our final output
    return c + np.dot(M, d2)

class WrappedPose:

    """
    TODO
    """
    
    def __init__( self ):
        #print( "HAHAHA" )
        #exit( 0 )
        self.CB_estimates = None
        self.legal_nbrs = None

    def get_legal_nbrs( self ):
        return self.legal_nbrs
        
    def get_atom_xyz( self, resid, atomname ):
        raise NotImplementedError

    def get_phi_psi( self, resid ):
        #radians
        raise NotImplementedError

    def get_chi( self, resid, chi_number ):
        #radians
        raise NotImplementedError
    
    def get_name1( self, resid ):
        raise NotImplementedError

    def residues_are_polymer_bonded( self, resid1, resid2 ):
        raise NotImplementedError

    def n_residues( self ):
        raise NotImplementedError

    def resid_is_N_term( self, resid ):
        raise NotImplementedError

    def resid_is_C_term( self, resid ):
        raise NotImplementedError

    def resids_are_same_chain( self, resid1, resid2 ):
        raise NotImplementedError

    def resid_is_designable( self, resid ):
        raise NotImplementedError

    def approximate_ALA_CB( self, resid ):
        assert hasattr( self, 'CB_estimates' )
        #if not hasattr( self, 'CB_estimates' ):
        if self.CB_estimates is None:
            #Lazy initialization
            self.CB_estimates = [ None for i in range( 0, self.n_residues() + 1 )]
        elif self.CB_estimates[ resid ] is not None:
            return self.CB_estimates[ resid ]
        self.CB_estimates[ resid ] = estimate_CB_xyz( self.get_atom_xyz( resid, "C" ), self.get_atom_xyz( resid, "N" ), self.get_atom_xyz( resid, "CA" ) )
        return self.CB_estimates[ resid ]

    
class RosettaPoseWrapper( WrappedPose ):
    
    def __init__( self, pose ):
        WrappedPose.__init__(self)
        if rosetta is None:
            print( "RosettaPoseWrapper requires the pyrosetta library to be installed" )
            raise ImportError
        assert isinstance( pose, rosetta.core.pose.Pose )
        self.pose = pose
    
    def get_atom_xyz( self, resid, atomname ):
        xyz = self.pose.residue( resid ).xyz( atomname )
        return np.asarray( [ xyz.x, xyz.y, xyz.z ] )
    
    def get_phi_psi( self, resid ):
        phipsi = np.asarray( [ self.pose.phi( resid ), self.pose.psi( resid ) ] )
        phipsi[0] = math.radians( phipsi[0] )
        phipsi[1] = math.radians( phipsi[1] )        
        return phipsi

    def get_chi( self, resid, chi_number ):
        if self.pose.residue(resid).nchi() < chi_number:
            return 0, False
        chi_deg = self.pose.chi( chi_number, resid )
        chi_rad = math.radians( chi_deg )
        return chi_rad, True
    
    def get_name1( self, resid ):
        return self.pose.residue( resid ).name1()

    def residues_are_polymer_bonded( self, resid1, resid2 ):
        return self.pose.residue( resid1 ).is_polymer_bonded( resid2 )

    def n_residues( self ):
        return self.pose.size()

    def approximate_ALA_CB_via_mutation( self, resid ):
        if not self.pose.residue( resid ).name1() == 'G':
            print( "RosettaPoseWrapper.approximate_ALA_CB is only setup for glycine right now" )
            print( self.pose.residue( resid ).name1() )
            assert False
        mutator = rosetta.protocols.simple_moves.MutateResidue(resid,'ALA')
        mutator.apply( self.pose )
        xyz = self.get_atom_xyz( resid, "CB" )
        mutator = rosetta.protocols.simple_moves.MutateResidue(resid,'GLY')
        mutator.apply( self.pose )
        #print( "GLY", xyz, estimate_CB_xyz( self.get_atom_xyz( resid, "CA" ), self.get_atom_xyz( resid, "N" ), self.get_atom_xyz( resid, "C" ) ), self.get_atom_xyz( resid, "CA" ) )
        #print( "GLY", xyz, estimate_CB_xyz( self.get_atom_xyz( resid, "C" ), self.get_atom_xyz( resid, "N" ), self.get_atom_xyz( resid, "CA" ) ) )        
        return xyz

    def resid_is_N_term( self, resid ):
        #TODO
        return False

    def resid_is_C_term( self, resid ):
        #TODO
        return False

    def resids_are_same_chain( self, resid1, resid2 ):
        return self.pose.chain( resid1 ) == self.pose.chain( resid2 )
    

class MDTrajPoseWrapper( WrappedPose ):
    
    def __init__( self, mdtraj_trajectory ):
        WrappedPose.__init__(self)        
        if md is None:
            print( "MDTrajPoseWrapper requires the mdtraj library to be installed" )
            raise ImportError

        assert isinstance( mdtraj_trajectory, md.Trajectory )
        #TODO assert type
        assert mdtraj_trajectory.n_frames == 1
        self.trajectory = mdtraj_trajectory
        self.phi_atoms, self.phis = md.compute_phi( self.trajectory ) #RADIANS
        self.psi_atoms, self.psis = md.compute_psi( self.trajectory ) #RADIANS
        
        self.chis = [ None, None, None, None, None ] #Adding zero element just to make indexing easier
        self.chi_atoms = [None,None,None,None,None]
        
        self.chi_atoms[1], self.chis[1] = md.compute_chi1( self.trajectory )
        self.chi_atoms[2], self.chis[2] = md.compute_chi2( self.trajectory )
        self.chi_atoms[3], self.chis[3] = md.compute_chi3( self.trajectory )
        self.chi_atoms[4], self.chis[4] = md.compute_chi4( self.trajectory )        
    
    def get_atom_xyz( self, resid, atomname ):
        atom = self.trajectory.topology.residue( resid-1 ).atom( atomname )
        all_xyzs = self.trajectory.xyz
        return all_xyzs[ 0 ][ atom.index ] * 10 #nm -> Å

    def _get_phi_or_psi_angle( self, resid, value_vec, atom_vec ):
        assert len( value_vec[0] ) == len( atom_vec )
        # value_vec.shape: (n_frames, n_phi)
        # value_vec.shape: (n_phi, 4)
        
        #Okay this is pretty inefficient
        top = self.trajectory.topology
        for i in range( 0, len( value_vec[0] ) ):
            atom_index = atom_vec[ i ][ 2 ] #Last-Middle atom
            atom = top.atom( atom_index )
            if atom.residue.index == resid-1:
                return value_vec[0][ i ]
            elif atom.residue.index > resid-1:
                return 0
        return 0
    
    def get_phi_psi( self, resid ):
        top = self.trajectory.topology
        #assuming chain.index is 0-indexed
        phi_rad = self._get_phi_or_psi_angle( resid, self.phis, self.phi_atoms )
        psi_rad = self._get_phi_or_psi_angle( resid, self.psis, self.psi_atoms )
        return [ phi_rad, psi_rad ]
        '''
        if self.resid_is_N_term( resid ):
            phi_deg = 0.0
        else:
            n_nterm_before_resid = top.residue( resid-1 ).chain.index+1
            phi_deg = self.phis[0,resid-1-n_nterm_before_resid]

        if self.resid_is_C_term( resid ):
            psi_deg = 0.0
        else:
            n_cterm_before_resid = top.residue( resid-1 ).chain.index+1
            psi_deg = self.psis[0,resid-1-n_cterm_before_resid]
        return [ math.radians(phi_deg), math.radians(psi_deg) ]
        '''

    
    def get_chi( self, resid, chi_number ):
        #DOESN'T GIVE PROTON CHIs
        assert chi_number > 0
        assert chi_number <= 4
        
        #Okay this is pretty inefficient
        top = self.trajectory.topology
        for i in range( 0, len( self.chi_atoms[ chi_number ] ) ):
            atom_index = self.chi_atoms[ chi_number ][ i ][ 3 ] #Last atom
            atom = top.atom( atom_index )
            if atom.residue.index == resid-1:
                return self.chis[chi_number][0][ i ], True
            elif atom.residue.index > resid-1:
                return 0, False
        return 0, False
        #needs to be in radians
        #TODO UNIT??
        #assert False #it's way more complicated than this
        #print( self.chis[chi_number][0].shape )
        #return self.chis[chi_number][0][ resid-1 ]

    
    def get_name1( self, resid ):
        return self.trajectory.topology.residue( resid-1 ).code

    def residues_are_polymer_bonded( self, resid1, resid2 ):
        if not self.resids_are_same_chain( resid1, resid2 ):
            return False
        if abs( resid1 - resid2 ) > 1:
            return False
        first_res = min( resid1, resid2 )
        second_res = max( resid1, resid2 )
        top = self.trajectory.topology
        C_atom_index = top.residue( first_res-1  ).atom( "C" )
        N_atom_index = top.residue( second_res-1 ).atom( "N" )
        for a1, a2 in top.bonds:
            if (a1 == C_atom_index and a2 == N_atom_index) or (a2 == C_atom_index and a1 == N_atom_index):
                return True
        return False

    def n_residues( self ):
        return self.trajectory.topology.n_residues

    '''
    def approximate_ALA_CB( self, resid ):
        #TODO
        CA_xyz = self.get_atom_xyz( resid, "CA" )        
        # for now just return CA I guess???
        return CA_xyz
        raise NotImplementedError
    '''
    
    def resid_is_N_term( self, resid ):
        top = self.trajectory.topology        
        chain = top.residue( resid-1 ).chain
        N_term_resid = chain.residue( 0 )
        return N_term_resid == resid-1

    def resid_is_C_term( self, resid ):
        top = self.trajectory.topology        
        chain = top.residue( resid-1 ).chain
        C_term_resid = chain.residue( chain.n_residues-1 )
        return C_term_resid == resid-1

    def resids_are_same_chain( self, resid1, resid2 ):
        top = self.trajectory.topology
        return top.residue( resid1-1 ).chain.index == top.residue( resid2-1 ).chain.index

    
