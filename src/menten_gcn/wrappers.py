import numpy as np
import math

try:
    from pyrosetta import rosetta
except BaseException:
    rosetta = None

try:
    import mdtraj as md
except BaseException:
    md = None

'''
try:
    import Bio as bp
except:
    bp = None
'''


def estimate_CB_xyz(C_xyz, N_xyz, CA_xyz):
    # ICOOR_INTERNAL   CB  -122.800000   69.625412    1.521736   CA    N   C
    '''
    That is, CB is found 1.52 Ang from CA, at an angle of 70 degrees from the CA-N line, and a dihedral of -123 degrees for CB-CA-N-C.
    -ROCCO
    '''

    # Credit to Rohit Bhattacharya from https://github.com/rbhatta8/protein-design/blob/master/nerf.py
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

    length = 1.521736
    theta = 69.625412
    chi = -122.800000

    a = C_xyz
    b = N_xyz
    c = CA_xyz

    # calculate unit vectors AB and BC
    ab_unit = (b - a) / np.linalg.norm(b - a)
    bc_unit = (c - b) / np.linalg.norm(c - b)

    # calculate unit normals n = AB x BC
    # and p = n x BC
    n_unit = np.cross(ab_unit, bc_unit)
    n_unit = n_unit / np.linalg.norm(n_unit)
    p_unit = np.cross(n_unit, bc_unit)

    # create rotation matrix [BC; p; n] (3x3)
    M = np.array([bc_unit, p_unit, n_unit]).T

    # convert degrees to radians
    theta = np.pi / 180 * theta
    chi = np.pi / 180 * chi

    # calculate coord pre rotation matrix
    d2 = [-length * np.cos(theta), length * np.sin(theta) * np.cos(chi), length * np.sin(theta) * np.sin(chi)]

    # calculate with rotation as our final output
    return c + np.dot(M, d2)


class WrappedPose:

    """
    This is the base class for all pose representations.
    The internal Menten GCN code will use API listed here
    """

    def __init__(self, designable_resids=None):
        self.CB_estimates = None
        self.legal_nbrs = None
        self.designable_resids = designable_resids

    def get_legal_nbrs(self):
        return self.legal_nbrs

    def get_atom_xyz(self, resid, atomname):
        raise NotImplementedError

    def get_phi_psi(self, resid):
        # radians
        raise NotImplementedError

    def get_chi(self, resid, chi_number):
        # radians
        raise NotImplementedError

    def get_name1(self, resid):
        raise NotImplementedError

    def residues_are_polymer_bonded(self, resid1, resid2):
        raise NotImplementedError

    def n_residues(self):
        raise NotImplementedError

    def resid_is_N_term(self, resid):
        raise NotImplementedError

    def resid_is_C_term(self, resid):
        raise NotImplementedError

    def resids_are_same_chain(self, resid1, resid2):
        raise NotImplementedError

    def set_designable_resids(self, resids):
        self.designable_resids = resids

    def resid_is_designable(self, resid):
        assert self.designable_resids is not None
        return resid in self.designable_resids

    def approximate_ALA_CB(self, resid):
        assert hasattr(self, 'CB_estimates')
        # if not hasattr( self, 'CB_estimates' ):
        if self.CB_estimates is None:
            # Lazy initialization
            self.CB_estimates = [None for i in range(0, self.n_residues() + 1)]
        elif self.CB_estimates[resid] is not None:
            return self.CB_estimates[resid]
        get_xyz = self.get_atom_xyz
        self.CB_estimates[resid] = estimate_CB_xyz(get_xyz(resid, "C"), get_xyz(resid, "N"), get_xyz(resid, "CA"))
        return self.CB_estimates[resid]


class RosettaPoseWrapper(WrappedPose):

    """
    This wrapper takes a rosetta pose and requires pyrosetta to be installed

    Parameters
    ---------
    pose: Pose
        Rosetta pose
    """

    def __init__(self, pose):
        WrappedPose.__init__(self)
        if rosetta is None:
            print("RosettaPoseWrapper requires the pyrosetta library to be installed")
            raise ImportError
        assert isinstance(pose, rosetta.core.pose.Pose)
        self.pose = pose

    def get_atom_xyz(self, resid, atomname):
        xyz = self.pose.residue(resid).xyz(atomname)
        return np.asarray([xyz.x, xyz.y, xyz.z])

    def get_phi_psi(self, resid):
        phipsi = np.asarray([self.pose.phi(resid), self.pose.psi(resid)])
        phipsi[0] = math.radians(phipsi[0])
        phipsi[1] = math.radians(phipsi[1])
        return phipsi

    def get_chi(self, resid, chi_number):
        if self.pose.residue(resid).nchi() < chi_number:
            return 0, False
        chi_deg = self.pose.chi(chi_number, resid)
        chi_rad = math.radians(chi_deg)
        return chi_rad, True

    def get_name1(self, resid):
        return self.pose.residue(resid).name1()

    def residues_are_polymer_bonded(self, resid1, resid2):
        return self.pose.residue(resid1).is_polymer_bonded(resid2)

    def n_residues(self):
        return self.pose.size()

    def approximate_ALA_CB_via_mutation(self, resid):
        if not self.pose.residue(resid).name1() == 'G':
            print("RosettaPoseWrapper.approximate_ALA_CB is only setup for glycine right now")
            print(self.pose.residue(resid).name1())
            assert False
        mutator = rosetta.protocols.simple_moves.MutateResidue(resid, 'ALA')
        mutator.apply(self.pose)
        xyz = self.get_atom_xyz(resid, "CB")
        mutator = rosetta.protocols.simple_moves.MutateResidue(resid, 'GLY')
        mutator.apply(self.pose)
        return xyz

    def resid_is_N_term(self, resid):
        return self.pose.residue(resid).is_lower_terminus()

    def resid_is_C_term(self, resid):
        return self.pose.residue(resid).is_upper_terminus()

    def resids_are_same_chain(self, resid1, resid2):
        return self.pose.chain(resid1) == self.pose.chain(resid2)


class MDTrajPoseWrapper(WrappedPose):

    """
    This wrapper takes a MDTraj trajectory and requires MDTraj to be installed

    Parameters
    ---------
    mdtraj_trajectory: Trajectory
        Pose in MDTraj trajectory format
    """

    def __init__(self, mdtraj_trajectory):
        WrappedPose.__init__(self)
        if md is None:
            print("MDTrajPoseWrapper requires the mdtraj library to be installed")
            raise ImportError

        assert isinstance(mdtraj_trajectory, md.Trajectory)
        assert mdtraj_trajectory.n_frames == 1
        self.trajectory = mdtraj_trajectory

        # RADIANS:
        self.phi_atoms, self.phis = md.compute_phi(self.trajectory)
        self.psi_atoms, self.psis = md.compute_psi(self.trajectory)

        self.chis = [None, None, None, None, None]  # Adding zero element just to make indexing easier
        self.chi_atoms = [None, None, None, None, None]

        self.chi_atoms[1], self.chis[1] = md.compute_chi1(self.trajectory)
        self.chi_atoms[2], self.chis[2] = md.compute_chi2(self.trajectory)
        self.chi_atoms[3], self.chis[3] = md.compute_chi3(self.trajectory)
        self.chi_atoms[4], self.chis[4] = md.compute_chi4(self.trajectory)

    def get_atom_xyz(self, resid, atomname):
        atom = self.trajectory.topology.residue(resid - 1).atom(atomname)
        all_xyzs = self.trajectory.xyz
        return all_xyzs[0][atom.index] * 10  # nm -> Å

    def _get_phi_or_psi_angle(self, resid, value_vec, atom_vec):
        assert len(value_vec[0]) == len(atom_vec)
        # value_vec.shape: (n_frames, n_phi)
        # value_vec.shape: (n_phi, 4)

        # Okay this is pretty inefficient
        top = self.trajectory.topology
        for i in range(0, len(value_vec[0])):
            atom_index = atom_vec[i][2]  # Last-Middle atom
            atom = top.atom(atom_index)
            if atom.residue.index == resid - 1:
                return value_vec[0][i]
            elif atom.residue.index > resid - 1:
                return 0
        return 0

    def get_phi_psi(self, resid):
        phi_rad = self._get_phi_or_psi_angle(resid, self.phis, self.phi_atoms)
        psi_rad = self._get_phi_or_psi_angle(resid, self.psis, self.psi_atoms)
        return [phi_rad, psi_rad]

    def get_chi(self, resid, chi_number):
        # DOESN'T GIVE PROTON CHIs
        assert chi_number > 0
        assert chi_number <= 4

        # Okay this is pretty inefficient
        top = self.trajectory.topology
        for i in range(0, len(self.chi_atoms[chi_number])):
            atom_index = self.chi_atoms[chi_number][i][3]  # Last atom
            atom = top.atom(atom_index)
            if atom.residue.index == resid - 1:
                return self.chis[chi_number][0][i], True
            elif atom.residue.index > resid - 1:
                return 0, False
        return 0, False

    def get_name1(self, resid):
        return self.trajectory.topology.residue(resid - 1).code

    def residues_are_polymer_bonded(self, resid1, resid2):
        if not self.resids_are_same_chain(resid1, resid2):
            return False
        if abs(resid1 - resid2) > 1:
            return False
        first_res = min(resid1, resid2)
        second_res = max(resid1, resid2)
        top = self.trajectory.topology
        C_atom_index = top.residue(first_res - 1).atom("C")
        N_atom_index = top.residue(second_res - 1).atom("N")
        for a1, a2 in top.bonds:
            if (a1 == C_atom_index and a2 == N_atom_index) or (a2 == C_atom_index and a1 == N_atom_index):
                return True
        return False

    def n_residues(self):
        return self.trajectory.topology.n_residues

    def resid_is_N_term(self, resid):
        top = self.trajectory.topology
        chain = top.residue(resid - 1).chain
        N_term_resid = chain.residue(0)
        return N_term_resid == resid - 1

    def resid_is_C_term(self, resid):
        top = self.trajectory.topology
        chain = top.residue(resid - 1).chain
        C_term_resid = chain.residue(chain.n_residues - 1)
        return C_term_resid == resid - 1

    def resids_are_same_chain(self, resid1, resid2):
        top = self.trajectory.topology
        return top.residue(resid1 - 1).chain.index == top.residue(resid2 - 1).chain.index
