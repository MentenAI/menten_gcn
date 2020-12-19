from menten_gcn import *
from menten_gcn.decorators import *
from menten_gcn.playground import *
from menten_gcn.util import *

import fileinput
import spektral
import tensorflow as tf
import argparse

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import numpy as np

def test_main():
    pass


def NEE3_test_mask(Xval, Aval, Eval, N, F, S):
    X_in = Input(shape=(N, F), name='X_in')
    A_in = Input(shape=(N, N), sparse=False, name='A_in')
    # E_in = Input(shape=(N, N, S), name='E_in')

    E_mask = make_edge_mask(A_in)

    NEE3_mask = make_NEENEENEE_mask(E_mask)

    model = Model(inputs=[A_in], outputs=NEE3_mask)
    #model = Model(inputs=[A_in], outputs=E_mask)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    A = np.asarray(Aval).reshape(1, N, N)
    #A = A.reshape( 1, N, N )
    out = model.predict(A)
    # print( out )
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
                for l in k:
                    if np.sum(l) > 0:
                        print(l)
                        all_outs.append(l)
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
