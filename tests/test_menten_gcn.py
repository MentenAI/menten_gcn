from menten_gcn import *
from menten_gcn.decorators import *
from menten_gcn.playground import *
from menten_gcn.util import *

# import spektral
# import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow as tf

import numpy as np
import mdtraj as md


def test_main():
    pass


def NEE3_test_mask(Xval, Aval, Eval, N, F, S):
    # X_in = Input(shape=(N, F), name='X_in')
    A_in = Input(shape=(N, N), sparse=False, name='A_in')
    # E_in = Input(shape=(N, N, S), name='E_in')

    E_mask = make_edge_mask(A_in)

    NEE3_mask = make_NEENEENEE_mask(E_mask)

    model = Model(inputs=[A_in], outputs=NEE3_mask)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    A = np.asarray(Aval).reshape(1, N, N)
    out = model.predict(A)
    print(np.sum(out))
    assert(np.sum(out) == 12)

    # Add edge from 0 - 2
    A2 = np.copy(A)
    A2[0][0][2] = 1.0
    A2[0][2][0] = 1.0
    out = model.predict(A2)
    print(np.sum(out))
    assert(np.sum(out) == 24)

    # Add edge from 1 - 2
    A3 = np.copy(A)
    A3[0][1][2] = 0.0
    A3[0][2][1] = 0.0
    out = model.predict(A3)
    print(np.sum(out))
    assert(np.sum(out) == 6)


def observe_NEE3(Xval, Aval, Eval, N, F, S):
    X_in = Input(shape=(N, F), name='X_in')
    A_in = Input(shape=(N, N), sparse=False, name='A_in')
    E_in = Input(shape=(N, N, S), name='E_in')

    NEE3, _ = make_NEENEENEE(X_in, E_in)

    X = np.asarray(Xval).reshape(1, N, F)
    A = np.asarray(Aval).reshape(1, N, N)
    E = np.asarray(Eval).reshape(1, N, N, S)

    E_mask = make_edge_mask(A_in)
    NEE3_mask = make_NEENEENEE_mask(E_mask)
    print(NEE3.shape)
    print(NEE3_mask.shape)
    Temp = Multiply()([NEE3, NEE3_mask])
    model2 = Model(inputs=[X_in, A_in, E_in], outputs=Temp)
    model2.compile(optimizer='adam', loss='mean_squared_error')
    model2.summary()
    out = model2.predict([X, A, E])
    all_outs = []
    for i in out:
        for j in i:
            for k in j:
                for lvec in k:
                    if np.sum(lvec) > 0:
                        print(lvec)
                        all_outs.append(lvec)
    foo = np.testing.assert_array_equal
    foo(all_outs[0], [0., 13., -13., 100., 3., -3., 300., 1., -1., ])
    foo(all_outs[1], [0., -13., 13., 300., 1., -1., 100., 3., -3., ])
    foo(all_outs[2], [100., 3., -3., 0., 13., -13., 300., -1., 1., ])
    foo(all_outs[3], [100., 23., -23., 200., 13., -13., 300., 12., -12., ])
    foo(all_outs[4], [100., -3., 3., 300., -1., 1., 0., 13., -13., ])
    foo(all_outs[5], [100., -23., 23., 300., 12., -12., 200., 13., -13., ])
    foo(all_outs[6], [200., 13., -13., 100., 23., -23., 300., -12., 12., ])
    foo(all_outs[7], [200., -13., 13., 300., -12., 12., 100., 23., -23., ])
    foo(all_outs[8], [300., 1., -1., 0., -13., 13., 100., -3., 3., ])
    foo(all_outs[9], [300., -1., 1., 100., -3., 3., 0., -13., 13., ])
    foo(all_outs[10], [300., 12., -12., 100., -23., 23., 200., -13., 13., ])
    foo(all_outs[11], [300., -12., 12., 200., -13., 13., 100., -23., 23., ])


def test_NEE3():
    '''
         2
        / \
       1 - 3
        \ /
         0
    '''

    N = 5  # one is missing
    F = 1
    S = 1

    npA = np.zeros(shape=(N, N))
    npX = np.zeros(shape=(N, F))
    npE = np.zeros(shape=(N, N, S))

    npA[0][1] = 1.0
    npA[0][3] = 1.0
    npA[1][2] = 1.0
    npA[1][3] = 1.0
    npA[2][3] = 1.0

    npA += np.transpose(npA)

    npE[0][1][0] = 1
    npE[0][3][0] = 3
    npE[1][2][0] = 12
    npE[1][3][0] = 13
    npE[2][3][0] = 23

    npE -= np.transpose(npE, axes=[1, 0, 2])

    for i in range(0, N):
        npX[i][0] = i * 100

    NEE3_test_mask(npX, npA, npE, N, F, S)
    observe_NEE3(npX, npA, npE, N, F, S)


def test_masks():
    max_res = 3

    # PREP MODEL
    decorators = [SimpleBBGeometry()]
    data_maker = DataMaker(decorators=decorators, edge_distance_cutoff_A=10.0, max_residues=max_res)
    data_maker.summary()

    N, F, S = data_maker.get_N_F_S()

    X_in, A_in, E_in = data_maker.generate_XAE_input_tensors()

    X_mask = make_node_mask(A_in)
    E_mask = make_edge_mask(A_in)

    out1 = make_and_apply_node_mask(X=X_in, A=A_in)
    out2 = make_and_apply_edge_mask(E=E_in, A=A_in)

    model = Model(inputs=[X_in, A_in, E_in], outputs=[X_mask, E_mask, out1, out2])
    model.compile(optimizer='adam', loss='mean_squared_error')

    testX = [[[0., -1.21597895, 1.94387676],
              [1., 2.01919905, -0.10534814],
              [2., 4., 6.]]]
    testA = [[[0., 1., 0.],
              [1., 0., 0.],
              [0., 0., 0.]]]
    testE = [[[[1., 2.],
               [3.1, 4.1],
               [5., 6.], ],

              [[7.1, 8.1],
               [9., 1.],
               [9., 2.], ],

              [[8., 3.],
               [7., 4.],
               [6., 5.], ], ]]

    testX = np.asarray(testX).astype('float32')
    testA = np.asarray(testA).astype('float32')
    testE = np.asarray(testE).astype('float32')

    expected_Xmask = [[[1.],
                       [1.],
                       [0.]]]

    expected_Emask = [[[[0.],
                        [1.],
                        [0.]],

                       [[1.],
                        [0.],
                        [0.]],

                       [[0.],
                        [0.],
                        [0.]]]]

    expected_out1 = [[[0., -1.215979, 1.9438767],
                      [1., 2.0191991, -0.10534814],
                      [0., 0., 0.]]]

    expected_out2 = [[[[0., 0.],
                       [3.1, 4.1],
                       [0., 0.]],

                      [[7.1, 8.1],
                       [0., 0.],
                       [0., 0.]],

                      [[0., 0.],
                       [0., 0.],
                       [0., 0.]]]]

    test_out = model.predict([testX, testA, testE])
    equal = np.testing.assert_almost_equal
    equal(np.asarray(expected_Xmask), test_out[0], decimal=2)
    equal(np.asarray(expected_Emask), test_out[1], decimal=2)
    equal(np.asarray(expected_out1), test_out[2], decimal=2)
    equal(np.asarray(expected_out2), test_out[3], decimal=2)


def test_data_generator_1():
    d = DataHolder()
    for x in range(0, 10):
        t = [x, ]
        d.append(t, t, t, t)
    assert len(d.Xs) == 10
    assert d.size() == 10

    generator = DataHolderInputGenerator(d, 3)
    assert len(generator) == 4
    assert generator.n_elem() == 10

    def Xlen(generator, i):
        inp, out = generator[i]
        print("    length: ", len(inp[0]))
        return len(inp[0])

    def Xval(generator, i, j):
        inp, out = generator[i]
        print("    value: ", inp[0][j])
        return inp[0][j]

    assert Xlen(generator, 0) == 3
    assert Xlen(generator, 1) == 3
    assert Xlen(generator, 2) == 3
    assert Xlen(generator, 3) == 1

    assert Xval(generator, 0, 0) == 0
    assert Xval(generator, 0, 1) == 1
    assert Xval(generator, 0, 2) == 2
    assert Xval(generator, 1, 0) == 3
    assert Xval(generator, 1, 1) == 4
    assert Xval(generator, 1, 2) == 5
    assert Xval(generator, 2, 0) == 6
    assert Xval(generator, 2, 1) == 7
    assert Xval(generator, 2, 2) == 8
    assert Xval(generator, 3, 0) == 9


def test_data_generator_2():
    d = DataHolder()
    for x in range(0, 10):
        t = [x, ]
        d.append(t, t, t, t)
    assert len(d.Xs) == 10
    assert d.size() == 10
    d.save_to_file("test1")

    def run_on_gen(generator):

        assert len(generator) == 3

        def Xlen(generator, i):
            inp, out = generator[i]
            print("    length: ", len(inp[0]))
            return len(inp[0])

        def Xval(generator, i, j):
            inp, out = generator[i]
            print("    value: ", inp[0][j])
            return inp[0][j]

        assert Xlen(generator, 0) == 3
        assert Xlen(generator, 1) == 3
        assert Xlen(generator, 2) == 3

        assert Xval(generator, 0, 0) == 0
        assert Xval(generator, 0, 1) == 1
        assert Xval(generator, 0, 2) == 2
        assert Xval(generator, 1, 0) == 3
        assert Xval(generator, 1, 1) == 4
        assert Xval(generator, 1, 2) == 5
        assert Xval(generator, 2, 0) == 6
        assert Xval(generator, 2, 1) == 7
        assert Xval(generator, 2, 2) == 8

    gen1 = CachedDataHolderInputGenerator(["test1.npz"], cache=True, batch_size=3, autoshuffle=False)
    gen2 = CachedDataHolderInputGenerator(["test1.npz"], cache=False, batch_size=3, autoshuffle=False)
    run_on_gen(gen1)
    run_on_gen(gen2)


def test_data_generator_3():
    d = DataHolder()
    for x in range(0, 10):
        t = [x, ]
        d.append(t, t, t, t)
    assert len(d.Xs) == 10
    assert d.size() == 10
    d.save_to_file("test1")

    def run_on_gen(generator):

        assert len(generator) == 6

        def Xlen(generator, i):
            inp, out = generator[i]
            print("    length: ", len(inp[0]))
            return len(inp[0])

        def Xval(generator, i, j):
            inp, out = generator[i]
            print("    value: ", inp[0][j])
            return inp[0][j]

        assert Xlen(generator, 0) == 3
        assert Xlen(generator, 1) == 3
        assert Xlen(generator, 2) == 3
        assert Xlen(generator, 3) == 3
        assert Xlen(generator, 4) == 3
        assert Xlen(generator, 5) == 3

        assert Xval(generator, 0, 0) == 0
        assert Xval(generator, 0, 1) == 1
        assert Xval(generator, 0, 2) == 2
        assert Xval(generator, 1, 0) == 3
        assert Xval(generator, 1, 1) == 4
        assert Xval(generator, 1, 2) == 5
        assert Xval(generator, 2, 0) == 6
        assert Xval(generator, 2, 1) == 7
        assert Xval(generator, 2, 2) == 8
        assert Xval(generator, 3, 0) == 0
        assert Xval(generator, 3, 1) == 1
        assert Xval(generator, 3, 2) == 2
        assert Xval(generator, 4, 0) == 3
        assert Xval(generator, 4, 1) == 4
        assert Xval(generator, 4, 2) == 5
        assert Xval(generator, 5, 0) == 6
        assert Xval(generator, 5, 1) == 7
        assert Xval(generator, 5, 2) == 8

    gen1 = CachedDataHolderInputGenerator(["test1.npz", "test1.npz"], cache=True, batch_size=3, autoshuffle=False)
    gen2 = CachedDataHolderInputGenerator(["test1.npz", "test1.npz"], cache=False, batch_size=3, autoshuffle=False)
    run_on_gen(gen1)
    run_on_gen(gen2)


def test_data_generator_4():
    d = DataHolder()
    for x in range(0, 10):
        t = [x, ]
        d.append(t, t, t, t)
    assert len(d.Xs) == 10
    assert d.size() == 10
    d.save_to_file("test1")

    def run_on_gen(generator):

        assert len(generator) == 6

        def Xlen(generator, i):
            inp, out = generator[i]
            return len(inp[0])

        def Xval(generator, i, j):
            inp, out = generator[i]
            return inp[0][j]

        assert Xlen(generator, 0) == 3
        assert Xlen(generator, 1) == 3
        assert Xlen(generator, 2) == 3
        assert Xlen(generator, 3) == 3
        assert Xlen(generator, 4) == 3
        assert Xlen(generator, 5) == 3

        for epoch in range(0, 10):
            log = np.zeros(10)
            for i in range(0, 6):
                for j in range(0, 3):
                    log[Xval(generator, i, j)] += 1
            assert(max(log) == 2)
            assert(min(log) <= 1)
            assert(sum(log) == 18)

    gen1 = CachedDataHolderInputGenerator(["test1.npz", "test1.npz"], cache=False, batch_size=3, autoshuffle=True)
    run_on_gen(gen1)


def test_expected_md_traj_results():
    pose = md.load_pdb("tests/6U07.atoms.pdb")
    wrapped_pose = MDTrajPoseWrapper(mdtraj_trajectory=pose)

    focus_resid = 20
    wrapped_pose.set_designable_resids([focus_resid - 1, focus_resid + 1])

    decorators = [
        CACA_dist(False),
        CBCB_dist(True),
        PhiPsiRadians(True),
        ChiAngleDecorator(sincos=False),
        trRosettaEdges(False, False),
        trRosettaEdges(True, True),
        SimpleBBGeometry(True),
        StandardBBGeometry(False),
        AdvancedBBGeometry(True),
        Sequence(),
        DesignableSequence(),
        SequenceSeparation(False),
        SequenceSeparation(True),
        SameChain()
    ]
    max_res = 5
    data_maker = DataMaker(decorators=decorators, edge_distance_cutoff_A=10.0, max_residues=max_res)
    data_maker.summary()
    data_cache = data_maker.make_data_cache(wrapped_pose)
    # N, F, S = data_maker.get_N_F_S()

    X, A, E, resids = data_maker.generate_input_for_resid(wrapped_pose, resid=focus_resid, data_cache=data_cache)

    assert_equal = np.testing.assert_array_almost_equal

    #print( repr(X) )
    expectedX = np.array([[1., -0.89294368, -0.4501684, 0.79902528, -0.60129743,
                           -1.00760889, -5., -5., -5., -2.03775024,
                           2.21592021, -0.89294368, -0.4501684, 0.79902528, -0.60129743,
                           -0.89294368, -0.4501684, 0.79902528, -0.60129743, 0.,
                           1., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           1., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.],
                          [0., -0.91330973, -0.40726568, 0.9341497, -0.35688142,
                           3.0659349, 1.21624207, -5., -5., -1.99025452,
                           1.93572366, -0.91330973, -0.40726568, 0.9341497, -0.35688142,
                           -0.91330973, -0.40726568, 0.9341497, -0.35688142, 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 1., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 1.],
                          [0., -0.79260743, -0.60973228, 0.26000427, -0.96560747,
                           -1.00995135, -5., -5., -5., -2.22651911,
                           2.87856603, -0.79260743, -0.60973228, 0.26000427, -0.96560747,
                           -0.79260743, -0.60973228, 0.26000427, -0.96560747, 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 1., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 1.],
                          [0., -0.98301462, -0.18352725, 0.81931582, -0.57334248,
                           3.13822532, 2.80935764, -0.42981434, -5., -1.75536978,
                           2.18137598, -0.98301462, -0.18352725, 0.81931582, -0.57334248,
                           -0.98301462, -0.18352725, 0.81931582, -0.57334248, 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 1., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 1., 0., 0.,
                           0., 0., 0., 0., 0.],
                          [0., -0.73194581, -0.68136285, 0.629598, -0.77692108,
                           -5., -5., -5., -5., -2.32041931,
                           2.46055698, -0.73194581, -0.68136285, 0.629598, -0.77692108,
                           -0.73194581, -0.68136285, 0.629598, -0.77692108, 1.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 1.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.]])

    expectedE = np.array([[[0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                            0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                            0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                            0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                            0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                            0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                            0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                            0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                            0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                            0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                           [1.00000000e+00, 3.77745056e+00, 5.21257277e-01,
                            5.21257277e+00, 2.41748101e+00, 1.46301712e+00,
                            1.55667868e-01, 5.21257277e-01, 6.62470246e-01,
                            -7.49088228e-01, 9.94197442e-01, 1.07570661e-01,
                            1.55039926e-01, 9.87908205e-01, 5.21257277e-01,
                            5.21257277e+00, 2.41748101e+00, 1.46301712e+00,
                            1.55667868e-01, 3.77745056e-01, 5.21257277e-01,
                            6.62470246e-01, -7.49088228e-01, 9.94197442e-01,
                            1.07570661e-01, 1.55039926e-01, 9.87908205e-01,
                            1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                           [1.00000000e+00, 3.80310774e+00, 5.18470378e-01,
                            5.18470378e+00, -2.43636241e+00, -1.32813630e-01,
                            1.27921111e+00, 5.18470378e-01, -6.48209170e-01,
                            -7.61462325e-01, -1.32423514e-01, 9.91193227e-01,
                            9.57789375e-01, 2.87470891e-01, 5.18470378e-01,
                            5.18470378e+00, -2.43636241e+00, -1.32813630e-01,
                            1.27921111e+00, 3.80310774e-01, 5.18470378e-01,
                            -6.48209170e-01, -7.61462325e-01, -1.32423514e-01,
                            9.91193227e-01, 9.57789375e-01, 2.87470891e-01,
                            1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                           [0.00000000e+00, 4.79972792e+00, 6.35613475e-01,
                            6.35613475e+00, 8.18635498e-01, -1.81740865e+00,
                            1.02962943e+00, 6.35613475e-01, 7.30214257e-01,
                            6.83218222e-01, -9.69744986e-01, -2.44120180e-01,
                            8.57108152e-01, 5.15136502e-01, 6.35613475e-01,
                            6.35613475e+00, 8.18635498e-01, -1.81740865e+00,
                            1.02962943e+00, 4.79972792e-01, 6.35613475e-01,
                            7.30214257e-01, 6.83218222e-01, -9.69744986e-01,
                            -2.44120180e-01, 8.57108152e-01, 5.15136502e-01,
                            1.20000000e+01, 2.48490665e+00, 1.00000000e+00],
                           [0.00000000e+00, 5.06669903e+00, 4.09958379e-01,
                            4.09958379e+00, -1.97356545e+00, 1.65878046e+00,
                            1.74688996e+00, 4.09958379e-01, -9.19979116e-01,
                            -3.91967379e-01, 9.96131892e-01, -8.78706619e-02,
                            9.84535540e-01, -1.75184961e-01, 4.09958379e-01,
                            4.09958379e+00, -1.97356545e+00, 1.65878046e+00,
                            1.74688996e+00, 5.06669903e-01, 4.09958379e-01,
                            -9.19979116e-01, -3.91967379e-01, 9.96131892e-01,
                            -8.78706619e-02, 9.84535540e-01, -1.75184961e-01,
                            4.10000000e+01, 3.71357207e+00, 1.00000000e+00]],

                          [[1.00000000e+00, 3.77745056e+00, 5.21257277e-01,
                            5.21257277e+00, 2.41748101e+00, -2.72808793e-01,
                            1.35779918e+00, 5.21257277e-01, 6.62470246e-01,
                            -7.49088228e-01, -2.69437414e-01, 9.63017902e-01,
                            9.77401737e-01, 2.11390265e-01, 5.21257277e-01,
                            5.21257277e+00, 2.41748101e+00, -2.72808793e-01,
                            1.35779918e+00, 3.77745056e-01, 5.21257277e-01,
                            6.62470246e-01, -7.49088228e-01, -2.69437414e-01,
                            9.63017902e-01, 9.77401737e-01, 2.11390265e-01,
                            1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                           [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                           [0.00000000e+00, 6.53150368e+00, 6.23720245e-01,
                              6.23720245e+00, 1.04258410e+00, -9.39745405e-01,
                              2.01471689e+00, 6.23720245e-01, 8.63709469e-01,
                              5.03990033e-01, -8.07407917e-01, 5.89993606e-01,
                              9.03074792e-01, -4.29483317e-01, 6.23720245e-01,
                              6.23720245e+00, 1.04258410e+00, -9.39745405e-01,
                              2.01471689e+00, 6.53150368e-01, 6.23720245e-01,
                              8.63709469e-01, 5.03990033e-01, -8.07407917e-01,
                              5.89993606e-01, 9.03074792e-01, -4.29483317e-01,
                              2.00000000e+00, 6.93147181e-01, 1.00000000e+00],
                           [0.00000000e+00, 6.42992830e+00, 6.52457531e-01,
                              6.52457531e+00, -2.47192669e+00, 6.19965701e-01,
                              2.08185249e+00, 6.52457531e-01, -6.20724127e-01,
                              -7.84029055e-01, 5.81007245e-01, 8.13898385e-01,
                              8.72228424e-01, -4.89098738e-01, 6.52457531e-01,
                              6.52457531e+00, -2.47192669e+00, 6.19965701e-01,
                              2.08185249e+00, 6.42992830e-01, 6.52457531e-01,
                              -6.20724127e-01, -7.84029055e-01, 5.81007245e-01,
                              8.13898385e-01, 8.72228424e-01, -4.89098738e-01,
                              1.30000000e+01, 2.56494936e+00, 1.00000000e+00],
                           [0.00000000e+00, 6.59098196e+00, 6.70634178e-01,
                              6.70634178e+00, -2.54078206e+00, -7.10189627e-01,
                              8.19914181e-01, 6.70634178e-01, -5.65311298e-01,
                              -8.24877649e-01, -6.51977565e-01, 7.58238257e-01,
                              7.31087279e-01, 6.82283951e-01, 6.70634178e-01,
                              6.70634178e+00, -2.54078206e+00, -7.10189627e-01,
                              8.19914181e-01, 6.59098196e-01, 6.70634178e-01,
                              -5.65311298e-01, -8.24877649e-01, -6.51977565e-01,
                              7.58238257e-01, 7.31087279e-01, 6.82283951e-01,
                              4.00000000e+01, 3.68887945e+00, 1.00000000e+00]],

                          [[1.00000000e+00, 3.80310774e+00, 5.18470378e-01,
                            5.18470378e+00, -2.43636241e+00, 2.51736645e+00,
                            3.37729081e-01, 5.18470378e-01, -6.48209170e-01,
                            -7.61462325e-01, 5.84469581e-01, -8.11415621e-01,
                            3.31345315e-01, 9.43509556e-01, 5.18470378e-01,
                            5.18470378e+00, -2.43636241e+00, 2.51736645e+00,
                            3.37729081e-01, 3.80310774e-01, 5.18470378e-01,
                            -6.48209170e-01, -7.61462325e-01, 5.84469581e-01,
                            -8.11415621e-01, 3.31345315e-01, 9.43509556e-01,
                            1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                           [0.00000000e+00, 6.53150368e+00, 6.23720245e-01,
                              6.23720245e+00, 1.04258410e+00, 2.06165865e+00,
                              1.22842446e+00, 6.23720245e-01, 8.63709469e-01,
                              5.03990033e-01, 8.81926697e-01, -4.71386573e-01,
                              9.41961026e-01, 3.35722244e-01, 6.23720245e-01,
                              6.23720245e+00, 1.04258410e+00, 2.06165865e+00,
                              1.22842446e+00, 6.53150368e-01, 6.23720245e-01,
                              8.63709469e-01, 5.03990033e-01, 8.81926697e-01,
                              -4.71386573e-01, 9.41961026e-01, 3.35722244e-01,
                              2.00000000e+00, 6.93147181e-01, 1.00000000e+00],
                           [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                           [0.00000000e+00, 5.90524387e+00, 7.97905472e-01,
                              7.97905472e+00, 2.48246664e+00, 1.05348782e+00,
                              9.11257157e-01, 7.97905472e-01, 6.12426177e-01,
                              -7.90527785e-01, 8.69153383e-01, 4.94542614e-01,
                              7.90274690e-01, 6.12752735e-01, 7.97905472e-01,
                              7.97905472e+00, 2.48246664e+00, 1.05348782e+00,
                              9.11257157e-01, 5.90524387e-01, 7.97905472e-01,
                              6.12426177e-01, -7.90527785e-01, 8.69153383e-01,
                              4.94542614e-01, 7.90274690e-01, 6.12752735e-01,
                              1.10000000e+01, 2.39789527e+00, 1.00000000e+00],
                           [0.00000000e+00, 5.88012457e+00, 7.38970827e-01,
                              7.38970827e+00, 1.94745415e+00, -3.09860415e+00,
                              7.99894372e-01, 7.38970827e-01, 9.29899129e-01,
                              -3.67814641e-01, -4.29752682e-02, -9.99076136e-01,
                              7.17282495e-01, 6.96782479e-01, 7.38970827e-01,
                              7.38970827e+00, 1.94745415e+00, -3.09860415e+00,
                              7.99894372e-01, 5.88012457e-01, 7.38970827e-01,
                              9.29899129e-01, -3.67814641e-01, -4.29752682e-02,
                              -9.99076136e-01, 7.17282495e-01, 6.96782479e-01,
                              4.20000000e+01, 3.73766962e+00, 1.00000000e+00]],

                          [[0.00000000e+00, 4.79972792e+00, 6.35613475e-01,
                            6.35613475e+00, 8.18635498e-01, -1.68352177e+00,
                            9.53393531e-01, 6.35613475e-01, 7.30214257e-01,
                            6.83218222e-01, -9.93653213e-01, -1.12486856e-01,
                            8.15384777e-01, 5.78919395e-01, 6.35613475e-01,
                            6.35613475e+00, 8.18635498e-01, -1.68352177e+00,
                            9.53393531e-01, 4.79972792e-01, 6.35613475e-01,
                            7.30214257e-01, 6.83218222e-01, -9.93653213e-01,
                            -1.12486856e-01, 8.15384777e-01, 5.78919395e-01,
                            1.20000000e+01, 2.48490665e+00, 1.00000000e+00],
                           [0.00000000e+00, 6.42992830e+00, 6.52457531e-01,
                              6.52457531e+00, -2.47192669e+00, -5.57830363e-01,
                              6.57827040e-01, 6.52457531e-01, -6.20724127e-01,
                              -7.84029055e-01, -5.29346713e-01, 8.48405597e-01,
                              6.11398784e-01, 7.91322644e-01, 6.52457531e-01,
                              6.52457531e+00, -2.47192669e+00, -5.57830363e-01,
                              6.57827040e-01, 6.42992830e-01, 6.52457531e-01,
                              -6.20724127e-01, -7.84029055e-01, -5.29346713e-01,
                              8.48405597e-01, 6.11398784e-01, 7.91322644e-01,
                              1.30000000e+01, 2.56494936e+00, 1.00000000e+00],
                           [0.00000000e+00, 5.90524387e+00, 7.97905472e-01,
                              7.97905472e+00, 2.48246664e+00, -2.46466253e+00,
                              4.07380581e-01, 7.97905472e-01, 6.12426177e-01,
                              -7.90527785e-01, -6.26403009e-01, -7.79499371e-01,
                              3.96205640e-01, 9.18161800e-01, 7.97905472e-01,
                              7.97905472e+00, 2.48246664e+00, -2.46466253e+00,
                              4.07380581e-01, 5.90524387e-01, 7.97905472e-01,
                              6.12426177e-01, -7.90527785e-01, -6.26403009e-01,
                              -7.79499371e-01, 3.96205640e-01, 9.18161800e-01,
                              1.10000000e+01, 2.39789527e+00, 1.00000000e+00],
                           [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                           [0.00000000e+00, 9.79303265e+00, 1.01709727e+00,
                              1.01709727e+01, -8.82493422e-01, -1.45632592e+00,
                              9.70255595e-01, 1.01709727e+00, -7.72325168e-01,
                              6.35227388e-01, -9.93455414e-01, 1.14220578e-01,
                              8.25030174e-01, 5.65088676e-01, 1.01709727e+00,
                              1.01709727e+01, -8.82493422e-01, -1.45632592e+00,
                              9.70255595e-01, 9.79303265e-01, 1.01709727e+00,
                              -7.72325168e-01, 6.35227388e-01, -9.93455414e-01,
                              1.14220578e-01, 8.25030174e-01, 5.65088676e-01,
                              5.30000000e+01, 3.97029191e+00, 1.00000000e+00]],

                          [[0.00000000e+00, 5.06669903e+00, 4.09958379e-01,
                            4.09958379e+00, -1.97356545e+00, 1.50972002e+00,
                            1.59042309e+00, 4.09958379e-01, -9.19979116e-01,
                            -3.91967379e-01, 9.98135422e-01, 6.10383456e-02,
                            9.99807401e-01, -1.96254985e-02, 4.09958379e-01,
                            4.09958379e+00, -1.97356545e+00, 1.50972002e+00,
                            1.59042309e+00, 5.06669903e-01, 4.09958379e-01,
                            -9.19979116e-01, -3.91967379e-01, 9.98135422e-01,
                            6.10383456e-02, 9.99807401e-01, -1.96254985e-02,
                            4.10000000e+01, 3.71357207e+00, 1.00000000e+00],
                           [0.00000000e+00, 6.59098196e+00, 6.70634178e-01,
                              6.70634178e+00, -2.54078206e+00, 6.50273416e-01,
                              1.87043295e+00, 6.70634178e-01, -5.65311298e-01,
                              -8.24877649e-01, 6.05404045e-01, 7.95918301e-01,
                              9.55443812e-01, -2.95173036e-01, 6.70634178e-01,
                              6.70634178e+00, -2.54078206e+00, 6.50273416e-01,
                              1.87043295e+00, 6.59098196e-01, 6.70634178e-01,
                              -5.65311298e-01, -8.24877649e-01, 6.05404045e-01,
                              7.95918301e-01, 9.55443812e-01, -2.95173036e-01,
                              4.00000000e+01, 3.68887945e+00, 1.00000000e+00],
                           [0.00000000e+00, 5.88012457e+00, 7.38970827e-01,
                              7.38970827e+00, 1.94745415e+00, 1.00069872e+00,
                              1.02282430e+00, 7.38970827e-01, 9.29899129e-01,
                              -3.67814641e-01, 8.41848298e-01, 5.39714224e-01,
                              8.53582765e-01, 5.20957257e-01, 7.38970827e-01,
                              7.38970827e+00, 1.94745415e+00, 1.00069872e+00,
                              1.02282430e+00, 5.88012457e-01, 7.38970827e-01,
                              9.29899129e-01, -3.67814641e-01, 8.41848298e-01,
                              5.39714224e-01, 8.53582765e-01, 5.20957257e-01,
                              4.20000000e+01, 3.73766962e+00, 1.00000000e+00],
                           [0.00000000e+00, 9.79303265e+00, 1.01709727e+00,
                              1.01709727e+01, -8.82493422e-01, 1.36264600e+00,
                              1.84424741e+00, 1.01709727e+00, -7.72325168e-01,
                              6.35227388e-01, 9.78414825e-01, 2.06650502e-01,
                              9.62844646e-01, -2.70055899e-01, 1.01709727e+00,
                              1.01709727e+01, -8.82493422e-01, 1.36264600e+00,
                              1.84424741e+00, 9.79303265e-01, 1.01709727e+00,
                              -7.72325168e-01, 6.35227388e-01, 9.78414825e-01,
                              2.06650502e-01, 9.62844646e-01, -2.70055899e-01,
                              5.30000000e+01, 3.97029191e+00, 1.00000000e+00],
                           [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]])

    expectedA = np.array([[0., 1., 1., 1., 1.],
                          [1., 0., 1., 1., 1.],
                          [1., 1., 0., 1., 1.],
                          [1., 1., 1., 0., 1.],
                          [1., 1., 1., 1., 0.]])

    assert_equal(X, expectedX, 2)
    assert_equal(A, expectedA, 2)
    assert_equal(E, expectedE, 2)
    assert_equal(resids, [20, 21, 19, 8, 61], 2)


def test_model_sizes():
    N = 5
    F = 4
    S = 3
    X_in = Input(shape=(N, F), name='X_in')
    A_in = Input(shape=(N, N), sparse=False, name='A_in')
    E_in = Input(shape=(N, N, S), name='E_in')

    def assert_n_params(inp, out, expected_size):
        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer='adam', loss='mean_squared_error')
        print(model.count_params())
        assert(model.count_params() == expected_size)

    NENE = make_NENE(X_in, E_in)
    assert_n_params([X_in, A_in, E_in], NENE, 0)

    NEENEENEE = make_NEENEENEE(X_in, E_in)
    assert_n_params([X_in, A_in, E_in], NEENEENEE, 0)

    X, E = make_1body_conv(X_in, A_in, E_in, 10, 20)
    assert_n_params([X_in, A_in, E_in], [X, E], 130)
    # 130 = ( 4 + 1 ) * 10 + ( 3 + 1 ) * 20
    # 130 = 50 + 80

    X, E = make_2body_conv(X_in, A_in, E_in,
                           [5], 10, 20,
                           attention=False, apply_T_to_E=False)
    assert_n_params([X_in, A_in, E_in], [X, E], 530)
    # t = (4+4+3+3+1)*5 =  75
    # x = (4+5+5+1)*10  = 150
    # e = (4+4+3+3+1)*20= 300
    # p                 =   5    #Prelu
    # total = t+x+e+p   = 530

    X, E = make_2body_conv(X_in, A_in, E_in,
                           5, 10, 20,
                           attention=False, apply_T_to_E=True)
    assert_n_params([X_in, A_in, E_in], [X, E], 350)
    # int vs list:
    X, E = make_2body_conv(X_in, A_in, E_in,
                           [5], 10, 20,
                           attention=False, apply_T_to_E=True)
    assert_n_params([X_in, A_in, E_in], [X, E], 350)
    # t = (4+4+3+3+1)*5 =  75
    # x = (4+5+5+1)*10  = 150
    # e = (5+1)*20      = 120
    # p                 =   5    #Prelu
    # total = t+x+e+p   = 350

    X, E = make_2body_conv(X_in, A_in, E_in,
                           [5], 10, 20,
                           attention=True, apply_T_to_E=True)
    assert_n_params([X_in, A_in, E_in], [X, E], 362)
    # t = (4+4+3+3+1)*5     =  75
    # a = (5+1)*1   *2      =  12    #Attention
    # x = (4+5+5+1)*10      = 150
    # e = (5+1)*20          = 120
    # p                     =   5    #Prelu
    # total = t+a+x+e+p     = 362

    X, E = make_2body_conv(X_in, A_in, E_in,
                           [50, 5], 10, 20,
                           attention=True, apply_T_to_E=True)
    assert_n_params([X_in, A_in, E_in], [X, E], 1342)
    # t1 = (4+4+3+3+1)*50   =  750
    # t2 = (50+1)*5         =  255
    # a = (5+1)*1   *2      =   12  #Attention
    # x = (4+5+5+1)*10      =  150
    # e = (5+1)*20          =  120
    # p = 50 + 5            =   55  #Prelu
    # total = t1+t2+a+x+e+p = 1342

    X, E = make_3body_conv(X_in, A_in, E_in,
                           [5], 10, 20, attention=False)
    assert_n_params([X_in, A_in, E_in], [X, E], 1100)
    # t = (4+4+4+3+3+3+3+3+3+1)*5 =  155
    # x = (4+5+5+5+1)*10          =  200
    # e = (3+3+(6*5)+1)*20        =  740
    # p                           =    5    #Prelu
    # total = t+x+e+p             = 1100

    X, E = make_3body_conv(X_in, A_in, E_in,
                           [5], 10, 20, attention=True)
    assert_n_params([X_in, A_in, E_in], [X, E], 1136)
    # int vs list:
    X, E = make_3body_conv(X_in, A_in, E_in,
                           5, 10, 20, attention=True)
    assert_n_params([X_in, A_in, E_in], [X, E], 1136)
    # t = (4+4+4+3+3+3+3+3+3+1)*5 =  155
    # a = (5+1)*1   *6            =   36    #Attention
    # x = (4+5+5+5+1)*10          =  200
    # e = (3+3+(6*5)+1)*20        =  740
    # p                           =    5    #Prelu
    # total = t+a+x+e+p           = 1136

    X, E = make_3body_conv(X_in, A_in, E_in,
                           [7, 5], 10, 20, attention=True)
    assert_n_params([X_in, A_in, E_in], [X, E], 1245)
    # t1 = (4+4+4+3+3+3+3+3+3+1)*7=  217
    # t2 = (7+1)*5                =   40
    # a = (5+1)*1   *6            =   36    #Attention
    # x = (4+5+5+5+1)*10          =  200
    # e = (3+3+(6*5)+1)*20        =  740
    # p = 5 + 7                   =   12    #Prelu
    # total = t1+t2+a+x+e+p       = 1245


def test_clustering():
    pose = md.load_pdb("tests/6U07.atoms.pdb")
    # pose = md.load_pdb("6U07.atoms.pdb")
    wrapped_pose = MDTrajPoseWrapper(mdtraj_trajectory=pose)
    CAclusters = cluster_all_resids(wrapped_pose, 10, False)
    # print( repr( CAclusters ) )
    assert CAclusters == [
        [
            66, 63, 64, 65, 67, 68, 69], [
            89, 90, 91, 92, 93, 195, 196], [
                193, 190, 191, 192, 194], [
                    197, 198, 199, 200, 201, 202], [
                        1, 2, 3, 4, 24, 25, 26, 56], [
                            28, 27, 29, 30, 31, 79], [
                                81, 78, 80, 82, 83, 84], [
                                    53, 48, 49, 50, 51, 52, 54, 55], [
                                        72, 20, 70, 71, 73, 74, 75, 76], [
                                            77, 32, 33, 34], [
                                                35, 36, 37, 38, 39, 40, 41], [
                                                    87, 7, 8, 85, 86, 88, 106], [
                                                        5, 6, 21, 22, 23, 58, 107, 110], [
                                                            57, 43, 44, 45, 46, 47, 59, 144], [
                                                                61, 18, 19, 42, 60, 62, 164], [
                                                                    16, 10, 11, 12, 13, 14, 15, 17], [
                                                                        111, 108, 109, 112, 113, 114, 170, 171], [
                                                                            105, 9, 102, 103, 104, 216, 217], [
                                                                                147, 145, 146, 148, 149, 150, 163, 165], [
                                                                                    155, 151, 152, 153, 154, 156, 157, 158, 159], [
                                                                                        172, 169, 173, 174, 175, 176, 177, 214, 215], [
                                                                                            178, 135, 179, 180, 181, 213], [
                                                                                                136, 132, 133, 134, 137, 138, 139, 182], [
                                                                                                    140, 131, 141, 142, 143, 167], [
                                                                                                        168, 115, 116, 117, 166], [
                                                                                                            212, 99, 100, 101, 183, 210, 211], [
                                                                                                                118, 98, 119, 120, 121, 130, 184, 185, 209], [
                                                                                                                    97, 94, 95, 96, 122, 123, 162, 208], [
                                                                                                                        160, 124, 125, 126, 161], [
                                                                                                                            127, 128, 129, 186, 187, 188, 189], [
                                                                                                                                203, 204, 205, 206], [207]]

    CBclusters = cluster_all_resids(wrapped_pose, 15, True)
    # print( repr( CBclusters ) )
    assert CBclusters == [
        [
            193, 90, 93, 188, 189, 190, 191, 192, 194, 195, 196, 197, 202, 203, 204, 205, 206], [
            199, 89, 91, 92, 94, 95, 198, 200, 201], [
                157, 37, 124, 125, 126, 151, 152, 153, 154, 155, 156, 158, 159, 160, 161], [
                    53, 1, 47, 48, 49, 50, 51, 52, 54, 55, 56, 112, 113, 141, 142, 143, 168, 169, 170], [
                        66, 16, 17, 18, 20, 36, 38, 39, 40, 62, 63, 64, 65, 67, 68, 69, 70, 71, 72, 73], [
                            35, 30, 31, 32, 33, 34, 41, 42, 43, 59, 61, 74, 75, 76, 77, 78, 79, 150], [
                                81, 2, 3, 4, 5, 6, 22, 24, 25, 26, 28, 29, 80, 82, 83, 84, 85], [
                                    27, 23, 44, 45, 46, 57, 58, 144], [
                                        87, 7, 8, 9, 10, 21, 86, 88, 104, 105, 106, 107, 108, 109, 110], [
                                            111, 103, 114, 115, 116, 117, 166, 171, 172, 173, 174, 175, 217], [
                                                216, 102, 176, 177, 178, 179, 180, 181, 212, 213, 214, 215], [
                                                    15, 11, 12, 13, 14, 19, 99, 100, 101, 119, 210, 211], [
                                                        136, 131, 132, 133, 134, 135, 137, 138, 139, 140, 182, 183, 184, 209], [
                                                            167, 60, 98, 118, 120, 130, 145, 146, 147, 163, 164, 165, 185], [
                                                                148, 123, 127, 128, 129, 149, 162], [
                                                                    207, 96, 97, 186, 187, 208], [
                                                                        121, 122]]


def test_sanity_check_flat_nbody():
    N = 3
    #F = 3
    #S = 2
    testA = [[[0., 1., 0.],
              [1., 0., 0.],
              [0., 0., 0.]]]
    testE = [[[[1., 2.],
               [3.1, 4.1],
               [5., 6.], ],

              [[7.1, 8.1],
               [9., 1.],
               [9., 2.], ],

              [[8., 3.],
               [7., 4.],
               [6., 5.], ], ]]

    #testX = np.asarray(testX).astype('float32')
    testA = np.asarray(testA).astype('float32')
    testE = np.asarray(testE).astype('float32')
    #testE_ind = np.asarray(testE_ind).astype('float32')

    part1 = tf.dynamic_partition(testE, testA, 2)
    print(len(part1))
    print(part1[0])
    print(part1[1])
    """
    2
    tf.Tensor(
    [[1. 2.]
    [5. 6.]
    [9. 1.]
    [9. 2.]
    [8. 3.]
    [7. 4.]
    [6. 5.]], shape=(7, 2), dtype=float32)
    tf.Tensor(
    [[3.1 4.1]
    [7.1 8.1]], shape=(2, 2), dtype=float32)
    """

    sum1 = tf.math.reduce_sum(part1[1], axis=-1, keepdims=1)
    print(sum1)
    """
    tf.Tensor(
    [[ 7.2     ]
    [15.200001]], shape=(2, 1), dtype=float32)
    """

    indices1 = [
        [0],
        # [ 0, 1 ],
        [2],
        # [ 1, 0 ],
        [4],
        [5],
        [6],
        [7],
        [8],
    ]

    indices2 = [
        [1],
        [3],
    ]

    indices = [indices1, indices2]

    partitioned_data = [
        np.zeros(shape=(7, 1)),
        sum1
    ]

    stitch1_flat = tf.dynamic_stitch(indices, partitioned_data)
    print(stitch1_flat)
    """
    tf.Tensor(
    [ 0.        7.2       0.       15.200001  0.        0.        0.
    0.        0.      ], shape=(9,), dtype=float32)
    """

    stitch1 = tf.reshape(stitch1_flat, (N, N, 1))
    print(stitch1)
    """
    tf.Tensor(
    [[[ 0.      ]
    [ 7.2     ]
    [ 0.      ]]

    [[15.200001]
    [ 0.      ]
    [ 0.      ]]

    [[ 0.      ]
    [ 0.      ]
    [ 0.      ]]], shape=(3, 3, 1), dtype=float32)
    """

    stitch1_np = stitch1.numpy()
    #print( repr( stitch1_np ) )
    target = np.array([[[0.],
                        [7.2],
                        [0.]],

                       [[15.200001],
                        [0.],
                        [0.]],

                       [[0.],
                        [0.],
                        [0.]]])

    equal = np.testing.assert_almost_equal
    equal(stitch1_np, target, decimal=3)


def test_sanity_check_flat_nbody2():
    N = 3
    #F = 3
    #S = 2
    testA = [[[0., 1., 0.],
              [1., 0., 0.],
              [0., 0., 0.]]]
    testE = [[[[1., 2.],
               [3.1, 4.1],
               [5., 6.], ],

              [[7.1, 8.1],
               [9., 1.],
               [9., 2.], ],

              [[8., 3.],
               [7., 4.],
               [6., 5.], ], ]]

    #testX = np.asarray(testX).astype('float32')
    testA = np.asarray(testA).astype('float32')
    testE = np.asarray(testE).astype('float32')
    #testE_ind = np.asarray(testE_ind).astype('float32')

    part1 = tf.dynamic_partition(testE, testA, 2)
    print(len(part1))
    print(part1[0])
    print(part1[1])
    """
    2
    tf.Tensor(
    [[1. 2.]
    [5. 6.]
    [9. 1.]
    [9. 2.]
    [8. 3.]
    [7. 4.]
    [6. 5.]], shape=(7, 2), dtype=float32)
    tf.Tensor(
    [[3.1 4.1]
    [7.1 8.1]], shape=(2, 2), dtype=float32)
    """

    sum1 = tf.math.reduce_sum(part1[1], axis=-1, keepdims=1)
    print(sum1)
    """
    tf.Tensor(
    [[ 7.2     ]
    [15.200001]], shape=(2, 1), dtype=float32)
    """

    indices1 = [
        [0],
        # [ 0, 1 ],
        [2],
        # [ 1, 0 ],
        [4],
        [5],
        [6],
        [7],
        [8],
    ]

    indices2 = [
        [1],
        [3],
    ]

    indices = [indices1, indices2]

    zero_padding1 = tf.zeros(tf.shape(indices1), dtype=sum1.dtype)

    partitioned_data = [
        zero_padding1,
        sum1
    ]

    stitch1_flat = tf.dynamic_stitch(indices, partitioned_data)
    print(stitch1_flat)
    """
    tf.Tensor([ 0.        7.2       0.       15.200001], shape=(4,), dtype=float32)
    """

    zero_padding = tf.zeros(9 - tf.shape(stitch1_flat), dtype=stitch1_flat.dtype)
    print(zero_padding)
    # tf.Tensor([0. 0. 0. 0. 0.], shape=(5,), dtype=float32)

    stitch1_flat = tf.concat([stitch1_flat, zero_padding], -1)
    print(stitch1_flat)
    # [ 0.  7.2    0.    15.200001  0.  0.  0.  0.  0. ]

    stitch1 = tf.reshape(stitch1_flat, (N, N, 1))
    print(stitch1)
    """
    tf.Tensor(
    [[[ 0.      ]
    [ 7.2     ]
    [ 0.      ]]

    [[15.200001]
    [ 0.      ]
    [ 0.      ]]

    [[ 0.      ]
    [ 0.      ]
    [ 0.      ]]], shape=(3, 3, 1), dtype=float32)
    """

    stitch1_np = stitch1.numpy()
    #print( repr( stitch1_np ) )
    target = np.array([[[0.],
                        [7.2],
                        [0.]],

                       [[15.200001],
                        [0.],
                        [0.]],

                       [[0.],
                        [0.],
                        [0.]]])

    equal = np.testing.assert_almost_equal
    equal(stitch1_np, target, decimal=3)


def test_flat_nbody_layer():

    class TestFlat(tf.keras.layers.Layer):
        def __init__(self):
            super(TestFlat, self).__init__()
            self.N = -1
            self.S = -1

        def build(self, input_shape):
            print("input_shape: ", input_shape)
            # TensorShape([None, 3, 3]), TensorShape([None, 3, 3, 2])]
            assert(len(input_shape) == 2)
            assert(len(input_shape[0]) == 3)
            assert(len(input_shape[1]) == 4)
            assert(input_shape[0][1] == input_shape[1][1])
            assert(input_shape[0][2] == input_shape[1][2])
            self.N = input_shape[0][1]
            self.S = input_shape[1][3]
            pass

        def call(self, inputs):
            print("inputs: ", inputs)
            # [<tf.Tensor 'A_in:0' shape=(None, 3, 3)    dtype=float32>,
            # <tf.Tensor 'E_in:0' shape=(None, 3, 3, 2) dtype=float32>]

            A = inputs[0]
            E = inputs[1]

            print(A)  # shape=(None, 3, 3)
            print(E)  # shape=(None, 3, 3, 2)

            A_int = tf.cast(A, "int32")
            '''
            A_int_flat = tf.keras.layers.Flatten()(tf.cast( A, "int32" ))
            print( A_int_flat ) #shape=(None, 9)

            E_flat = tf.keras.layers.Flatten()(E)
            print( E_flat ) #shape=(None, 18)
            exit( 0 )
            '''

            part = tf.dynamic_partition(E, A_int, 2)
            print(len(part))
            print(part[0])  # shape=(None, 2)
            print(part[1])  # shape=(None, 2)
            """
            2
            tf.Tensor(
            [[1. 2.]
            [5. 6.]
            [9. 1.]
            [9. 2.]
            [8. 3.]
            [7. 4.]
            [6. 5.]], shape=(7, 2), dtype=float32)
            tf.Tensor(
            [[3.1 4.1]
            [7.1 8.1]], shape=(2, 2), dtype=float32)
            """

            sum1 = tf.math.reduce_sum(part[1], axis=-1, keepdims=True)
            print(sum1.shape)

            x = tf.constant(self.N * self.N)
            n = tf.constant(self.N)
            r = tf.range(x * tf.shape(E)[0])
            print(r)  # Tensor("test_flat/range:0", shape=(9,), dtype=int32)

            print("!!!", tf.shape(E)[0])
            r2 = tf.reshape(r, shape=[tf.shape(E)[0], n, n])
            print(r2)  # Tensor("test_flat/Reshape:0", shape=(1, 3, 3), dtype=int32)
            condition_indices = tf.dynamic_partition(r2, A_int, 2)
            print(condition_indices)
            # [<tf.Tensor 'test_flat/DynamicPartition_1:0' shape=(None,) dtype=int32>,
            # <tf.Tensor 'test_flat/DynamicPartition_1:1' shape=(None,) dtype=int32>]

            s_1 = tf.shape(condition_indices[0])[0]
            s_2 = int(1)
            s = [s_1, s_2]
            print("s", s, s_1, s_2)
            zero_padding1 = tf.zeros(shape=s)

            #indices = [condition_indices[1]]
            partitioned_data = [zero_padding1, sum1]
            stitch_flat = tf.dynamic_stitch(condition_indices, partitioned_data)
            print("stitch_flat", stitch_flat)
            # Tensor("test_flat/DynamicStitch:0", shape=(None, 1), dtype=float32)

            #npad = tf.constant( tf.shape(E)[0]*n*n - tf.shape(stitch_flat)[0] )
            npad1 = tf.shape(E)[0] * n * n
            print("npad1", npad1)
            npad2 = tf.shape(stitch_flat)[0]
            print("npad2", npad2)
            nz = npad1 - npad2
            print("nz", nz)

            zero_padding = tf.zeros(nz, dtype=stitch_flat.dtype)
            print("zeros", zero_padding)
            zero_padding = tf.reshape(zero_padding, [nz, 1])
            print("zeros", zero_padding)

            print("tf.shape(stitch_flat)", tf.shape(stitch_flat))
            stitch = tf.concat([stitch_flat, zero_padding], -2)

            stitch = tf.reshape(stitch, [tf.shape(E)[0], n, n, 1])

            return stitch  # dummy for now

    N = 3
    #F = 3
    S = 2

    # X_in = Input(shape=(N, F), name='X_in')
    A_in = Input(shape=(N, N), name='A_in')
    E_in = Input(shape=(N, N, S), name='E_in')

    out = TestFlat()([A_in, E_in])

    model = Model(inputs=[A_in, E_in], outputs=out)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    testA = [[[0., 1., 0.],
              [1., 0., 0.],
              [0., 0., 0.]]]
    testE = [[[[1., 2.],
               [3.1, 4.1],
               [5., 6.], ],

              [[7.1, 8.1],
               [9., 1.],
               [9., 2.], ],

              [[8., 3.],
               [7., 4.],
               [6., 5.], ], ]]

    target = np.array([[[0.],
                        [7.2],
                        [0.]],

                       [[15.200001],
                        [0.],
                        [0.]],

                       [[0.],
                        [0.],
                        [0.]]])
    assert_almost_equal = np.testing.assert_almost_equal

    testA = np.asarray(testA).astype('float32')
    testE = np.asarray(testE).astype('float32')

    batch1_pred = model([testA, testE])
    print("test1", batch1_pred)
    """
    tf.Tensor(
    [[ 0.      ]
    [ 7.2     ]
    [ 0.      ]
    [15.200001]], shape=(4, 1), dtype=float32)
    """
    assert_almost_equal(batch1_pred[0], target, decimal=3)

    testA2 = np.asarray([testA[0], testA[0]])
    testE2 = np.asarray([testE[0], testE[0]])

    print("testA2", testA2.shape)
    print("testE2", testE2.shape)
    """
    testA2 (2, 3, 3)
    testE2 (2, 3, 3, 2)
    """

    batch2_pred = model([testA2, testE2])
    print("test2", batch2_pred)

    for output in batch2_pred:
        assert_almost_equal(output, target, decimal=3)
