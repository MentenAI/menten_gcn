import math

from menten_gcn.decorators.base import *
from menten_gcn.wrappers import RosettaPoseWrapper

try:
    from pyrosetta import rosetta
except BaseException:
    rosetta = None

#from pyrosetta import rosetta
# Caller needs to call init()

# import scipy  # Jump
from scipy.spatial.transform import Rotation as R

# Convention: All decorators must start with "Rosetta". This allows us to
# standardize them later while still maintaining backwards compatability


class RosettaResidueSelectorDecorator(Decorator):

    """
    Takes a user-provided residue selctor and labels each residue with a 1 or 0 accordingly.

    - 1 Node Feature
    - 0 Edge Features

    Parameters
    ---------
    selector: ResidueSelector
        This residue selector will be applied to the Rosetta pose
    description: str
        This is the string that will label this feature in the final summary. Not technically required but highly recommended
    """

    def __init__(self, selector, description: str):
        assert isinstance(selector, rosetta.core.select.residue_selector.ResidueSelector)
        assert isinstance(description, str)
        self.selector = selector
        self.description = description
        self.unique_key = str(id(self)) + "_selection"

    def get_version_name(self):
        return "RosettaResidueSelectorDecorator"

    def _get_selection(self, wrapped_pose):
        assert isinstance(wrapped_pose, RosettaPoseWrapper)
        pose = wrapped_pose.pose
        return self.selector.apply(pose)

    def cache_data(self, wrapped_pose, dict_cache):
        if self.unique_key in dict_cache:
            pass
            #assert isinstance( dict_cache[ self.unique_key ], rosetta.core.select.residue_selector.ResidueSelection )
        else:
            selection = self._get_selection(wrapped_pose)
            #assert isinstance( selection, rosetta.core.select.residue_selector.ResidueSelection )
            dict_cache[self.unique_key] = selection

    def n_node_features(self):
        return 1

    def calc_node_features(self, wrapped_pose, resid, dict_cache=None):
        if dict_cache is None:
            selection = self._get_selection(wrapped_pose)
        else:
            assert self.unique_key in dict_cache
            selection = dict_cache[self.unique_key]

        if selection[resid]:
            return [1.0]
        else:
            return [0.0]

    def describe_node_features(self):
        return [
            "1.0 if the residue is selected by the residue selector, 0.0 otherwise. "
            + "User defined definition of the residue selector and how to reproduce it: "
            + self.description,
        ]


class RosettaResidueSelectorFromXML(RosettaResidueSelectorDecorator):
    """
    Takes a user-provided residue selctor via XML and labels each residue with a 1 or 0 accordingly.

    - 1 Node Feature
    - 0 Edge Features

    Parameters
    ---------
    xml_str: str
        XML snippet that defines the selector
    res_sele_name: str
        The name of the selector within the snippet
    """

    # Useful resource: https://www.programmersought.com/article/87461668890/
    def __init__(self, xml_str: str, res_sele_name: str):
        xml = rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(xml_str)
        selector = xml.get_residue_selector(res_sele_name)
        description = "Took the residue selector named " + res_sele_name + " from this XML: " + xml_str
        super(RosettaResidueSelectorFromXML, self).__init__(selector, description=description)


class RosettaHBondDecorator_v0(Decorator):

    def __init__(self, sfxn=None, bb_only: bool = False):
        self.key = "hbset"
        self.bb_only = bb_only
        if sfxn is None:
            #self.sfxn = rosetta.core.scoring.ScoreFunctionFactory.get_score_function()
            self.sfxn = rosetta.core.scoring.get_score_function()
        else:
            self.sfxn = sfxn

    def get_version_name(self):
        return "RosettaHBondDecorator_v0"

    def cache_data(self, wrapped_pose, dict_cache):
        if self.key in dict_cache:
            assert isinstance(dict_cache[self.key], rosetta.core.scoring.hbonds.HBondSet)
        else:
            assert isinstance(wrapped_pose, RosettaPoseWrapper)
            pose = wrapped_pose.pose
            hbset = rosetta.core.scoring.hbonds.HBondSet(pose, False)
            dict_cache[self.key] = hbset

    def n_node_features(self):
        return 0

    def n_edge_features(self):
        if self.bb_only:
            return 1
        else:
            return 5

    def calc_edge_features(self, wrapped_pose, resid1, resid2, dict_cache):
        if dict_cache is None:
            assert isinstance(wrapped_pose, RosettaPoseWrapper)
            pose = wrapped_pose.pose
            hbset = rosetta.core.scoring.hbonds.HBondSet(pose, False)
        else:
            assert self.key in dict_cache
            hbset = dict_cache.get(self.key)
        assert hbset is not None

        n_bb_bb = 0
        n_bb_sc = 0
        n_sc_sc = 0
        n_don1 = 0  # n_don2 = n_acc1
        n_acc1 = 0  # n_acc2 = n_don1

        hbonds = hbset.residue_hbonds(resid1, False)
        for hbond in hbonds:
            if resid1 + resid2 == hbond.don_res() + hbond.acc_res():
                res1_is_don = (resid1 == hbond.don_res())
                if res1_is_don:
                    n_don1 += 1
                else:
                    n_acc1 += 1
                if hbond.acc_atm_is_backbone():
                    if hbond.don_hatm_is_backbone():
                        n_bb_bb += 1
                    else:
                        n_bb_sc += 1
                else:
                    if hbond.don_hatm_is_backbone():
                        n_bb_sc += 1
                    else:
                        n_sc_sc += 1

        if self.bb_only:
            return [n_bb_bb], [n_bb_bb]

        f_12 = [n_bb_bb, n_bb_sc, n_sc_sc, n_don1, n_acc1]
        f_21 = [n_bb_bb, n_bb_sc, n_sc_sc, n_acc1, n_don1]

        return f_12, f_21

    def describe_edge_features(self):
        alldesc = [
            "Total number of backbone-backbone hbonds (symmetric)",
            "Total number of backbone-sidechain hbonds (symmetric)",
            "Total number of sidechain-sidechain hbonds (symmetric)",
            "Number of hbonds in which the first residue is the donor (asymmetric)",
            "Number of hbonds in which the first residue is the acceptor (asymmetric)",
        ]
        if self.bb_only:
            return [alldesc[0]]
        else:
            return alldesc


class _RosettaOnebodyEnergies_v0(Decorator):

    def __init__(self, sfxn, individual: bool = False):
        self.sfxn = sfxn
        self.ind = individual
        if individual:
            self.terms = sfxn.get_nonzero_weighted_scoretypes()

    def get_version_name(self):
        raise NotImplementedError  # Child class needs to define this

    def n_edge_features(self):
        return 0

    def n_node_features(self):
        if self.ind:
            return len(self.terms)
        else:
            return 1

    def calc_node_features(self, wrapped_pose, resid, dict_cache=None):
        assert isinstance(wrapped_pose, RosettaPoseWrapper)
        pose = wrapped_pose.pose
        self.sfxn.setup_for_scoring(pose)
        emap = rosetta.core.scoring.EMapVector()
        self.sfxn.eval_ci_1b(pose.residue(resid), pose, emap)
        self.sfxn.eval_cd_1b(pose.residue(resid), pose, emap)
        self.sfxn.eval_intrares_energy(pose.residue(resid), pose, emap)

        if self.ind:
            f = []
            for i in self.terms:
                f.append(emap[i])
            return f
        else:
            features = [emap.dot(self.sfxn.weights())]
            return features

    def describe_node_features(self):
        if self.ind:
            d = []
            for i in self.terms:
                desc = str(i) + " onebody term using " + self.get_version_name() + " (symmetric)"
                d.append(desc)
            return d
        else:
            desc = "Sum of all Rosetta onebody energies using " + self.get_version_name() + " (symmetric)"
            return [desc]


class _RosettaTwobodyEnergies_v0(Decorator):

    def __init__(self, sfxn, individual: bool = False):
        self.sfxn = sfxn
        self.ind = individual
        if individual:
            self.terms = sfxn.get_nonzero_weighted_scoretypes()

    def get_version_name(self):
        raise NotImplementedError  # Child class needs to define this

    def n_node_features(self):
        return 0

    def n_edge_features(self):
        if self.ind:
            return len(self.terms)
        else:
            return 1

    def calc_edge_features(self, wrapped_pose, resid1, resid2, dict_cache=None):
        assert isinstance(wrapped_pose, RosettaPoseWrapper)
        pose = wrapped_pose.pose
        self.sfxn.setup_for_scoring(pose)
        emap = rosetta.core.scoring.EMapVector()
        self.sfxn.eval_ci_2b(pose.residue(resid1), pose.residue(resid2), pose, emap)
        self.sfxn.eval_cd_2b(pose.residue(resid1), pose.residue(resid2), pose, emap)

        if self.ind:
            f = []
            for i in self.terms:
                f.append(emap[i])
            return f, f
        else:
            features = [emap.dot(self.sfxn.weights())]
            return features, features

    def describe_edge_features(self):
        if self.ind:
            d = []
            for i in self.terms:
                desc = str(i) + " twobody term using " + self.get_version_name() + " (symmetric)"
                d.append(desc)
            return d
        else:
            desc = "Sum of all Rosetta twobody energies using " + self.get_version_name() + " (symmetric)"
            return [desc]


class RosettaJumpDecorator(Decorator):

    """
    Measures the translational and rotational relationships between all residue pairs.
    This uses internal coordinate frames so it is agnostic to the global coordinate system.
    You can move/rotate your protein around and these will stay the same.

    - 0 Node Features
    - 6-12 Edge Features

    Parameters
    ---------
    use_nm: bool
        If true (default), measure distance in Angstroms.
        Otherwise use nanometers.
    rottype: str
        How do you want to represent the rotational degrees of freedom?
        Options are "euler" (default), "euler_sincos", "matrix",
        "quat", "rotvec", and "rotvec_sincos".
    """

    def __init__(self, use_nm: bool = False, rottype: str = "euler"):
        assert(rottype in ["euler", "euler_sincos", "matrix", "quat", "rotvec", "rotvec_sincos"])
        self.rottype = rottype
        self.use_nm = use_nm

    def get_version_name(self):
        return "RosettaJumpDecorator"

    def n_node_features(self):
        return 0

    def n_edge_features(self):
        if self.rottype == "euler" or self.rottype == "rotvec":
            return 6
        elif self.rottype == "quat":
            return 7
        elif self.rottype == "euler_sincos" or self.rottype == "rotvec_sincos":
            return 9
        elif self.rottype == "matrix":
            return 12
        else:
            assert False, self.rottype

    def jump_to_vec(self, jump):
        trans = jump.get_translation()
        if self.use_nm:
            trans /= 10.0

        rot_mat = jump.get_rotation()
        rot = R.from_matrix(rot_mat)

        vec = []
        vec.extend(trans)

        if self.rottype == "euler":
            rot_euler = rot.as_euler('xyz', degrees=False)
            vec.extend(rot_euler)
        elif self.rottype == "euler_sincos":
            rot_euler = rot.as_euler('xyz', degrees=False)
            for i in range(0, 3):
                vec.append(math.sin(rot_euler[i]))
                vec.append(math.cos(rot_euler[i]))
        elif self.rottype == "rotvec":
            vec.extend(rot.as_rotvec())
        elif self.rottype == "rotvec_sincos":
            rot_vec = rot.as_rotvec()
            for i in range(0, 3):
                vec.append(math.sin(rot_vec[i]))
                vec.append(math.cos(rot_vec[i]))
        elif self.rottype == "quat":
            vec.extend(rot.as_quat())
        elif self.rottype == "matrix":
            for i in range(0, 3):
                vec.extend(rot_mat[i])

        return vec

    def calc_edge_features(self, wrapped_pose, resid1, resid2, dict_cache=None):
        assert isinstance(wrapped_pose, RosettaPoseWrapper)
        pose = wrapped_pose.pose
        stub1 = rosetta.protocols.hotspot_hashing.StubGenerator.residueStubOrientFrame(pose.residue(resid1))
        stub2 = rosetta.protocols.hotspot_hashing.StubGenerator.residueStubOrientFrame(pose.residue(resid2))
        jump_ij = rosetta.core.kinematics.Jump(stub1, stub2)
        jump_ji = rosetta.core.kinematics.Jump(stub2, stub1)

        f_ij = self.jump_to_vec(jump_ij)
        f_ji = self.jump_to_vec(jump_ji)

        return f_ij, f_ji

    def describe_edge_features(self):
        if self.use_nm:
            d_units = "nanometers"
        else:
            d_units = "Angstroms"

        return ["Value #{} for the Rosetta jump. Distances are measured in {}".format(i, d_units)
                for i in range(0, self.n_edge_features())]


class RosettaHBondDecorator(RosettaHBondDecorator_v0):
    """
    Takes a user-provided residue selctor via XML and labels each residue with a 1 or 0 accordingly.

    - 0 Node Features
    - 1-5 Edge Features (depending on bb_only)

    Parameters
    ---------
    sfxn: ScoreFunction
        Score function used to calculate hbonds.
        We will use Rosetta's default if this is None
    bb_only: bool
        Only consider backbone-backbone hbonds.
        Reduces the number of features from 5 down to 1
    """
    pass


class Rosetta_Ref2015_OneBodyEneriges(_RosettaOnebodyEnergies_v0):
    """
    Label each node with its Rosetta one-body energy

    - 1 - 20-ish Node Features
    - 0 Edge Features

    Parameters
    ---------
    individual: bool
        If true, list the score for each term individually.
        Otherwise sum them all into one value.
    """

    def __init__(self, individual: bool = False):
        sfxn = rosetta.core.scoring.ScoreFunctionFactory.create_score_function("ref2015.wts")
        _RosettaOnebodyEnergies_v0.__init__(self, sfxn=sfxn, individual=individual)

    def get_version_name(self):
        return "Rosetta_Ref2015_OneBodyEneriges"


class Rosetta_Ref2015_TwoBodyEneriges(_RosettaTwobodyEnergies_v0):
    """
    Label each edge with its Rosetta two-body energy

    - 0 Node Features
    - 1 - 20-ish Edge Features

    Parameters
    ---------
    individual: bool
        If true, list the score for each term individually.
        Otherwise sum them all into one value.
    """

    def __init__(self, individual: bool = False):
        sfxn = rosetta.core.scoring.ScoreFunctionFactory.create_score_function("ref2015.wts")
        _RosettaTwobodyEnergies_v0.__init__(self, sfxn=sfxn, individual=individual)

    def get_version_name(self):
        return "Rosetta_Ref2015_TwoBodyEneriges"


class Ref2015Decorator(CombinedDecorator):

    """
    Meta-decorator that combines Rosetta_Ref2015_OneBodyEneriges and Rosetta_Ref2015_TwoBodyEneriges

    - 1 - 20-ish Node Features
    - 1 - 20-ish Edge Features

    Parameters
    ---------
    individual: bool
        If true, list the score for each term individually.
        Otherwise sum them all into one value.
    """

    def __init__(self, individual: bool = False):
        decorators = [Rosetta_Ref2015_OneBodyEneriges(individual=individual), Rosetta_Ref2015_TwoBodyEneriges(individual=individual)]
        CombinedDecorator.__init__(self, decorators)

    def get_version_name(self):
        return "Ref2015Decorator"
