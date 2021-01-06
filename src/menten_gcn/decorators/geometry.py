import numpy as np
import math

from menten_gcn.decorators.base import *


def angle_rad(a, b, c):
    # https://stackoverflow.com/questions/35176451/python-code-to-calculate-angle-between-three-point-using-their-3d-coordinates
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return angle


def dihedral_rad(p0, p1, p2, p3):
    # https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
    # Praxeolitic formula 1 sqrt, 1 cross product
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    # return np.degrees(np.arctan2(y, x))
    return np.arctan2(y, x)


class CACA_dist(Decorator):

    """
    Measures distance between the two C-Alpha atoms of each residue

    - 0 Node Features
    - 1 Edge Feature

    Parameters
    ---------
    use_nm: bool
        If true (default), measure distance in Angstroms.
        Otherwise use nanometers.
    """

    def __init__(self, use_nm: bool = False):
        self.use_nm = use_nm

    def get_version_name(self):
        return "CACA_dist"

    def n_edge_features(self):
        return 1

    def calc_edge_features(self, wrapped_pose, resid1: int, resid2: int, dict_cache=None):
        xyz1 = wrapped_pose.get_atom_xyz(resid1, "CA")
        xyz2 = wrapped_pose.get_atom_xyz(resid2, "CA")
        distance = [np.linalg.norm(xyz1 - xyz2)]
        if self.use_nm:
            distance[0] = distance[0] / 10.0

        # distance is symmetric, so return it twice
        return distance, distance

    def describe_edge_features(self):
        if self.use_nm:
            d_units = "nanometers"
        else:
            d_units = "Angstroms"
        return [
            "Euclidean distance between the CA atoms of each residue, measured in " + d_units,
        ]


class CBCB_dist(Decorator):

    """
    Measures distance between the two C-Beta atoms of each residue.
    Note: We will calculate the "ideal ALA" CB location even if this residue has a CB atom.
    This may sound silly but it is intended to prevents noise from different native amino acid types.

    - 0 Node Features
    - 1 Edge Feature

    Parameters
    ---------
    use_nm: bool
        If true (default), measure distance in Angstroms.
        Otherwise use nanometers.
    """

    def __init__(self, use_nm: bool = False):
        self.use_nm = use_nm

    def get_version_name(self):
        return "CBCB_dist"

    def n_edge_features(self):
        return 1

    def calc_edge_features(self, wrapped_pose, resid1: int, resid2: int, dict_cache=None):
        CB1_xyz = wrapped_pose.approximate_ALA_CB(resid1)
        CB2_xyz = wrapped_pose.approximate_ALA_CB(resid2)
        distance = [np.linalg.norm(CB1_xyz - CB2_xyz)]
        if self.use_nm:
            distance[0] = distance[0] / 10.0

        return distance, distance

    def describe_edge_features(self):
        if self.use_nm:
            d_units = "nanometers"
        else:
            d_units = "Angstroms"
        return [
            "Euclidean distance between the CB atoms of each residue, measured in " +
            d_units + ". In the case of GLY, use an estimate of ALA's CB position",
        ]


class PhiPsiRadians(Decorator):

    """
    Returns the phi and psi values of each residue position.

    - 2-4 Node Features
    - 0 Edge Features

    Parameters
    ---------
    sincos: bool
        Return the sine and cosine of phi and psi instead of just the raw values.
    """

    def __init__(self, sincos: bool = False):
        self.sincos = sincos

    def get_version_name(self):
        return "PhiPsiRadians"

    def n_node_features(self):
        if self.sincos:
            return 4
        else:
            return 2

    def calc_node_features(self, wrapped_pose, resid, dict_cache=None):
        if self.sincos:
            phi_rad, psi_rad = wrapped_pose.get_phi_psi(resid)
            vec = np.zeros(shape=(4))
            if not wrapped_pose.resid_is_N_term(resid):
                vec[0] = math.sin(phi_rad)
                vec[1] = math.cos(phi_rad)
            if not wrapped_pose.resid_is_C_term(resid):
                vec[2] = math.sin(psi_rad)
                vec[3] = math.cos(psi_rad)
            return vec
        else:
            return wrapped_pose.get_phi_psi(resid)

    def describe_node_features(self):
        if self.sincos:
            return [
                "Sine of Phi of the given residue, 0 if N-term. Ranges from -1 to 1",
                "Cosine of Phi of the given residue, 0 if N-term. Ranges from -1 to 1",
                "Sine of Psi of the given residue, 0 if C-term. Ranges from -1 to 1",
                "Cosine of Psi of the given residue, 0 if C-term. Ranges from -1 to 1",
            ]
        else:
            return [
                "Phi of the given residue, measured in radians. Spans from -pi to pi",
                "Psi of the given residue, measured in radians. Spans from -pi to pi",
            ]


class trRosettaEdges(Decorator):

    """
    Use the residue pair geometries used in this paper:
    https://www.pnas.org/content/117/3/1496/tab-figures-data

    - 0 Node Features
    - 4-7 Edge Features

    Parameters
    ---------
    sincos: bool
        Return the sine and cosine of phi and psi instead of just the raw values.
    use_nm: bool
        If true, measure distance in Angstroms.
        Otherwise use nanometers.

        Note: This default value does not match the default of other decorators.
        This is for the sake of matching the trRosetta paper.
    """

    def __init__(self, sincos: bool = False, use_nm: bool = False):
        self.sincos = sincos
        self.use_nm = use_nm

    def get_version_name(self):
        return "trRosettaEdges"

    def n_node_features(self):
        return 0

    def n_edge_features(self):
        if self.sincos:
            return 7
        else:
            return 4

    def calc_edge_features(self, wrapped_pose, resid1, resid2, dict_cache=None):
        # See Fig1 of https://www.biorxiv.org/content/10.1101/846279v1.full.pdf
        CA1_xyz = wrapped_pose.get_atom_xyz(resid1, "CA")
        N1_xyz = wrapped_pose.get_atom_xyz(resid1, "N")
        CB1_xyz = wrapped_pose.approximate_ALA_CB(resid1)

        CA2_xyz = wrapped_pose.get_atom_xyz(resid2, "CA")
        N2_xyz = wrapped_pose.get_atom_xyz(resid2, "N")
        CB2_xyz = wrapped_pose.approximate_ALA_CB(resid2)

        CB_distance = np.linalg.norm(CB1_xyz - CB2_xyz)
        if self.use_nm:
            CB_distance = CB_distance / 10.0
        omega_rad = dihedral_rad(CA1_xyz, CB1_xyz, CB2_xyz, CA2_xyz)

        theta_12 = dihedral_rad(N1_xyz, CA1_xyz, CB1_xyz, CB2_xyz)
        theta_21 = dihedral_rad(N2_xyz, CA2_xyz, CB2_xyz, CB1_xyz)

        phi_12 = angle_rad(CA1_xyz, CB1_xyz, CB2_xyz)
        phi_21 = angle_rad(CA2_xyz, CB2_xyz, CB1_xyz)

        f_12 = [CB_distance, omega_rad, theta_12, phi_12]
        f_21 = [CB_distance, omega_rad, theta_21, phi_21]

        if self.sincos:
            return self.convert_edges_to_sincos(f_12, f_21)
        else:
            return f_12, f_21

    def convert_edges_to_sincos(self, f_12, f_21):
        f_12_prime = np.zeros(shape=(7))
        f_12_prime[0] = f_12[0]
        f_12_prime[1] = math.sin(f_12[1])
        f_12_prime[2] = math.cos(f_12[1])
        f_12_prime[3] = math.sin(f_12[2])
        f_12_prime[4] = math.cos(f_12[2])
        f_12_prime[5] = math.sin(f_12[3])
        f_12_prime[6] = math.cos(f_12[3])

        f_21_prime = np.zeros(shape=(7))
        f_21_prime[0] = f_21[0]
        f_21_prime[1] = math.sin(f_21[1])
        f_21_prime[2] = math.cos(f_21[1])
        f_21_prime[3] = math.sin(f_21[2])
        f_21_prime[4] = math.cos(f_21[2])
        f_21_prime[5] = math.sin(f_21[3])
        f_21_prime[6] = math.cos(f_21[3])

        np.testing.assert_almost_equal(f_12_prime[0:2], f_21_prime[0:2], decimal=4)

        return f_12_prime, f_21_prime

    def describe_edge_features(self):
        if self.use_nm:
            d_units = "nanometers"
        else:
            d_units = "Angstroms"
        if self.sincos:
            return [
                "Euclidean distance between the two CB atoms of each residue, measured in " + d_units + " (symmetric)",
                "Sine of CA-CB-CB-CA torsion angle, spans from -1 to 1 (symmetric)",
                "Cosine of CA-CB-CB-CA torsion angle, spans from -1 to 1 (symmetric)",
                "Sine of N1-CA1-CB1-CB2 torsion angle, spans from -1 to 1 (asymmetric)",
                "Cosine of N1-CA1-CB1-CB2 torsion angle, spans from -1 to 1 (asymmetric)",
                "Sine of CA1-CB1-CB2 bond angle, spans from 0 to 1 (asymmetric)",
                "Cosine of CA1-CB1-CB2 bond angle, spans from -1 to 1 (asymmetric)",
            ]
        else:
            return [
                "Euclidean distance between the two CB atoms of each residue, measured in " + d_units + " (symmetric)",
                "CA-CB-CB-CA torsion angle in radians, spans from -pi to pi (symmetric)",
                "N1-CA1-CB1-CB2 torsion angle in radians, spans from -pi to pi (asymmetric)",
                "CA1-CB1-CB2 bond angle in radians, spans from 0 to pi (asymmetric)",
            ]


class _SimpleBBGeometry_v0(CombinedDecorator):

    def __init__(self, use_nm=False):
        decorators = [PhiPsiRadians(sincos=False), CBCB_dist(use_nm=use_nm)]
        CombinedDecorator.__init__(self, decorators)

    def get_version_name(self):
        return "_SimpleBBGeometry_v0"


class _StandardBBGeometry_v0(CombinedDecorator):

    def __init__(self, use_nm=False):
        decorators = [PhiPsiRadians(sincos=True), trRosettaEdges(sincos=False, use_nm=use_nm)]
        CombinedDecorator.__init__(self, decorators)

    def get_version_name(self):
        return "_StandardBBGeometry_v0"


class _AdvancedBBGeometry_v0(CombinedDecorator):

    def __init__(self, use_nm=False):
        decorators = [PhiPsiRadians(sincos=True), CACA_dist(use_nm=use_nm), trRosettaEdges(sincos=True, use_nm=use_nm)]
        CombinedDecorator.__init__(self, decorators)

    def get_version_name(self):
        return "_AdvancedBBGeometry_v0"


class ChiAngleDecorator(Decorator):

    """
    Returns the chi values of each residue position. Ranges from -pi to pi or -1 to 1 if sincos=True.

    WARNING: This can behave inconsistantly for proton chis accross modeling frameworks.
    Rosetta adds hydrogens when they are absent from the input file but MDtraj does not.
    This results in Rosetta calculating a chi value in some cases that MDtraj skips!

    - 0-8 Node Features
    - 0 Edge Features

    Parameters
    ---------
    chi1: bool
        Include chi1's value
    chi2: bool
        Include chi2's value
    chi3: bool
        Include chi3's value
    chi4: bool
        Include chi4's value
    sincos: bool
        Return the sine and cosine of chi instead of just the raw values
    """

    def __init__(self, chi1: bool = True, chi2: bool = True, chi3: bool = True, chi4: bool = True, sincos: bool = True):
        self.chis = []
        if chi1:
            self.chis.append(1)
        if chi2:
            self.chis.append(2)
        if chi3:
            self.chis.append(3)
        if chi4:
            self.chis.append(4)
        self.sincos = sincos

        print(("ChiAngleDecorator: Warning, different protein representations (i.e., Rosetta vs MDTraj)"
               " represent proton chis differently if the hydrogen atoms are missing from the input file."))

    def get_version_name(self):
        return "ChiAngleDecorator"

    def n_edge_features(self):
        return 0

    def n_node_features(self):
        if self.sincos:
            return len(self.chis) * 2
        else:
            return len(self.chis)

    def calc_node_features(self, wrapped_pose, resid, dict_cache=None):
        f = []
        for chi in self.chis:
            chi_rad, is_valid = wrapped_pose.get_chi(resid, chi)
            #print( chi_rad )
            if self.sincos:
                if is_valid:
                    f.append(math.sin(chi_rad))
                    f.append(math.cos(chi_rad))
                else:
                    f.append(0)
                    f.append(0)
            else:
                if is_valid:
                    f.append(chi_rad)
                else:
                    f.append(-5)
        return f

    def describe_node_features(self):
        desc = []
        for chi in self.chis:
            if self.sincos:
                desc.append(
                    "Sine of chi angle number " +
                    str(chi) +
                    " of each residue. Spans from -1 to 1, 0 if chi angle is not valid for this residue")
                desc.append(
                    "Cosine of chi angle number " +
                    str(chi) +
                    " of each residue. Spans from -1 to 1, 0 if chi angle is not valid for this residue")
            else:
                desc.append(
                    "Chi angle number " +
                    str(chi) +
                    " of each residue measured in radians. Spans from -pi to pi, -5 if chi angle is not valid for this residue")
        return desc


class SimpleBBGeometry(_SimpleBBGeometry_v0):
    """
    Meta-decorator that combines PhiPsiRadians(sincos=False) and CBCB_dist

    - 2 Node Features
    - 1 Edge Feature

    Parameters
    ---------
    use_nm: bool
        If true, measure distance in Angstroms.
        Otherwise use nanometers.
    """
    pass


class StandardBBGeometry(_StandardBBGeometry_v0):
    """
    Meta-decorator that combines PhiPsiRadians(sincos=True) and trRosettaEdges(sincos=False)

    - 4 Node Features
    - 4 Edge Features

    Parameters
    ---------
    use_nm: bool
        If true, measure distance in Angstroms.
        Otherwise use nanometers.
    """
    pass


class AdvancedBBGeometry(_AdvancedBBGeometry_v0):
    """
    Meta-decorator that combines PhiPsiRadians(sincos=True), CACA_dist, and trRosettaEdges(sincos=True)

    - 4 Node Features
    - 8 Edge Features

    Parameters
    ---------
    use_nm: bool
        If true, measure all distances in Angstroms.
        Otherwise use nanometers.
    """
    pass
