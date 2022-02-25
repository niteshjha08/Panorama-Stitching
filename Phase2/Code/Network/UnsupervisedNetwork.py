"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow as tf2
import sys
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, InputLayer, BatchNormalization, Layer

from Misc.utils_tensorDLT import *
from Misc.TFSpatialTransformer import transformer
# Don't generate pyc codes
sys.dont_write_bytecode = True

# class TensorDLT(Layer):
#     def __init__(self, batch_size, pts_1_tile):
#         self.batch_size = batch_size
#         self.pts_1_tile = pts_1_tile

#     # TensorDLT call function taken from: https://github.com/tynguyen/unsupervisedDeepHomographyRAL2018/blob/master/code/homography_model.py
#     def call(self,batch_size,pts_1_tile):

#         batch_size = self.params.batch_size
#         pts_1_tile = self.pts_1_tile

#         # Solve for H using DLT
#         pred_h4p_tile = tf.expand_dims(self.pred_h4p, [2]) # BATCH_SIZE x 8 x 1
#         # 4 points on the second image
#         pred_pts_2_tile = tf.add(pred_h4p_tile, pts_1_tile)


#         # Auxiliary tensors used to create Ax = b equation
#         M1 = tf.constant(Aux_M1, tf.float32)
#         M1_tensor = tf.expand_dims(M1, [0])
#         M1_tile = tf.tile(M1_tensor,[batch_size,1,1])

#         M2 = tf.constant(Aux_M2, tf.float32)
#         M2_tensor = tf.expand_dims(M2, [0])
#         M2_tile = tf.tile(M2_tensor,[batch_size,1,1])

#         M3 = tf.constant(Aux_M3, tf.float32)
#         M3_tensor = tf.expand_dims(M3, [0])
#         M3_tile = tf.tile(M3_tensor,[batch_size,1,1])

#         M4 = tf.constant(Aux_M4, tf.float32)
#         M4_tensor = tf.expand_dims(M4, [0])
#         M4_tile = tf.tile(M4_tensor,[batch_size,1,1])

#         M5 = tf.constant(Aux_M5, tf.float32)
#         M5_tensor = tf.expand_dims(M5, [0])
#         M5_tile = tf.tile(M5_tensor,[batch_size,1,1])

#         M6 = tf.constant(Aux_M6, tf.float32)
#         M6_tensor = tf.expand_dims(M6, [0])
#         M6_tile = tf.tile(M6_tensor,[batch_size,1,1])


#         M71 = tf.constant(Aux_M71, tf.float32)
#         M71_tensor = tf.expand_dims(M71, [0])
#         M71_tile = tf.tile(M71_tensor,[batch_size,1,1])

#         M72 = tf.constant(Aux_M72, tf.float32)
#         M72_tensor = tf.expand_dims(M72, [0])
#         M72_tile = tf.tile(M72_tensor,[batch_size,1,1])

#         M8 = tf.constant(Aux_M8, tf.float32)
#         M8_tensor = tf.expand_dims(M8, [0])
#         M8_tile = tf.tile(M8_tensor,[batch_size,1,1])

#         Mb = tf.constant(Aux_Mb, tf.float32)
#         Mb_tensor = tf.expand_dims(Mb, [0])
#         Mb_tile = tf.tile(Mb_tensor,[batch_size,1,1])

#         # Form the equations Ax = b to compute H
#         # Form A matrix
#         A1 = tf.matmul(M1_tile, pts_1_tile) # Column 1
#         A2 = tf.matmul(M2_tile, pts_1_tile) # Column 2
#         A3 = M3_tile                   # Column 3
#         A4 = tf.matmul(M4_tile, pts_1_tile) # Column 4
#         A5 = tf.matmul(M5_tile, pts_1_tile) # Column 5
#         A6 = M6_tile                   # Column 6
#         A7 = tf.matmul(M71_tile, pred_pts_2_tile) *  tf.matmul(M72_tile, pts_1_tile)# Column 7
#         A8 = tf.matmul(M71_tile, pred_pts_2_tile) *  tf.matmul(M8_tile, pts_1_tile)# Column 8

#         A_mat = tf.transpose(tf.stack([tf.reshape(A1,[-1,8]),tf.reshape(A2,[-1,8]),\
#                                     tf.reshape(A3,[-1,8]),tf.reshape(A4,[-1,8]),\
#                                     tf.reshape(A5,[-1,8]),tf.reshape(A6,[-1,8]),\
#             tf.reshape(A7,[-1,8]),tf.reshape(A8,[-1,8])],axis=1), perm=[0,2,1]) # BATCH_SIZE x 8 (A_i) x 8
#         print('--Shape of A_mat:', A_mat.get_shape().as_list())
#         # Form b matrix
#         b_mat = tf.matmul(Mb_tile, pred_pts_2_tile)
#         print('--shape of b:', b_mat.get_shape().as_list())

#         # Solve the Ax = b
#         H_8el = tf.matrix_solve(A_mat , b_mat)  # BATCH_SIZE x 8.
#         print('--shape of H_8el', H_8el)


#         # Add ones to the last cols to reconstruct H for computing reprojection error
#         h_ones = tf.ones([batch_size, 1, 1])
#         H_9el = tf.concat([H_8el,h_ones],1)
#         H_flat = tf.reshape(H_9el, [-1,9])
#         H_mat = tf.reshape(H_flat,[-1,3,3])   # BATCH_SIZE x 3 x 3
#         return H_mat


# def HomographyModel(Img, ImageSize, MiniBatchSize):
# def UnsupervisedHomographyModel(batch_size):

#     """
#     Inputs: 
#     Img is a MiniBatch of the current image
#     ImageSize - Size of the Image
#     Outputs:
#     prLogits - logits output of the network
#     prSoftMax - softmax output of the network
#     """

#     #############################
#     # Fill your network here!
#     #############################
#     input_shape = (128,128,2)

#     model = Sequential()
#     model.add(InputLayer(input_shape))

#     # relu layer after batch normalization
#     # two conv2d layers and maxpooling2d layer on 128x128 size
#     model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu'))
#     model.add(BatchNormalization())

#     model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu'))
#     model.add(BatchNormalization())

#     model.add(MaxPooling2D(pool_size=(2,2)))

#     # two conv2d layers and maxpooling2d layer on 64x64
#     model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu'))
#     model.add(BatchNormalization())

#     model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu'))
#     model.add(BatchNormalization())

#     model.add(MaxPooling2D(pool_size=(2,2)))

#     # two conv2d layers and maxpooling layer on 32x32
#     model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu'))
#     model.add(BatchNormalization())

#     model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu'))
#     model.add(BatchNormalization())

#     model.add(MaxPooling2D(pool_size=(2,2)))

#     # two conv2d layers on 16x16
#     model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu'))
#     model.add(BatchNormalization())

#     model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu'))
#     model.add(BatchNormalization())

#     # Fully connected layer
#     model.add(Flatten())
#     model.add(Dropout(0.5))

#     model.add(Dense(1024,activation='relu'))
#     model.add(Dropout(0.5))

#     model.add(Dense(8))
#     # feed this into tensorDLT, get H, apply H to pA and get pB,
#     model.add(TensorDLT(batch_size=batch_size,h4pt))

#     return model

    # return H4Pt


def TensorDLT(batch_size,Ca,H_4pt):

    # batch_size = self.params.batch_size
    # pts_1_tile = self.pts_1_tile
    pts_1_tile = tf.expand_dims(Ca,[2])
    # Solve for H using DLT
    pred_h4p_tile = tf.expand_dims(H_4pt, [2]) # BATCH_SIZE x 8 x 1
    # 4 points on the second image
    pred_pts_2_tile = tf.add(pred_h4p_tile, pts_1_tile)


    # Auxiliary tensors used to create Ax = b equation
    M1 = tf.constant(Aux_M1, tf.float32)
    M1_tensor = tf.expand_dims(M1, [0])
    M1_tile = tf.tile(M1_tensor,[batch_size,1,1])

    M2 = tf.constant(Aux_M2, tf.float32)
    M2_tensor = tf.expand_dims(M2, [0])
    M2_tile = tf.tile(M2_tensor,[batch_size,1,1])

    M3 = tf.constant(Aux_M3, tf.float32)
    M3_tensor = tf.expand_dims(M3, [0])
    M3_tile = tf.tile(M3_tensor,[batch_size,1,1])

    M4 = tf.constant(Aux_M4, tf.float32)
    M4_tensor = tf.expand_dims(M4, [0])
    M4_tile = tf.tile(M4_tensor,[batch_size,1,1])

    M5 = tf.constant(Aux_M5, tf.float32)
    M5_tensor = tf.expand_dims(M5, [0])
    M5_tile = tf.tile(M5_tensor,[batch_size,1,1])

    M6 = tf.constant(Aux_M6, tf.float32)
    M6_tensor = tf.expand_dims(M6, [0])
    M6_tile = tf.tile(M6_tensor,[batch_size,1,1])


    M71 = tf.constant(Aux_M71, tf.float32)
    M71_tensor = tf.expand_dims(M71, [0])
    M71_tile = tf.tile(M71_tensor,[batch_size,1,1])

    M72 = tf.constant(Aux_M72, tf.float32)
    M72_tensor = tf.expand_dims(M72, [0])
    M72_tile = tf.tile(M72_tensor,[batch_size,1,1])

    M8 = tf.constant(Aux_M8, tf.float32)
    M8_tensor = tf.expand_dims(M8, [0])
    M8_tile = tf.tile(M8_tensor,[batch_size,1,1])

    Mb = tf.constant(Aux_Mb, tf.float32)
    Mb_tensor = tf.expand_dims(Mb, [0])
    Mb_tile = tf.tile(Mb_tensor,[batch_size,1,1])

    # Form the equations Ax = b to compute H
    # Form A matrix
    A1 = tf.matmul(M1_tile, pts_1_tile) # Column 1
    A2 = tf.matmul(M2_tile, pts_1_tile) # Column 2
    A3 = M3_tile                   # Column 3
    A4 = tf.matmul(M4_tile, pts_1_tile) # Column 4
    A5 = tf.matmul(M5_tile, pts_1_tile) # Column 5
    A6 = M6_tile                   # Column 6
    A7 = tf.matmul(M71_tile, pred_pts_2_tile) *  tf.matmul(M72_tile, pts_1_tile)# Column 7
    A8 = tf.matmul(M71_tile, pred_pts_2_tile) *  tf.matmul(M8_tile, pts_1_tile)# Column 8

    A_mat = tf.transpose(tf.stack([tf.reshape(A1,[-1,8]),tf.reshape(A2,[-1,8]),\
                                tf.reshape(A3,[-1,8]),tf.reshape(A4,[-1,8]),\
                                tf.reshape(A5,[-1,8]),tf.reshape(A6,[-1,8]),\
        tf.reshape(A7,[-1,8]),tf.reshape(A8,[-1,8])],axis=1), perm=[0,2,1]) # BATCH_SIZE x 8 (A_i) x 8
    print('--Shape of A_mat:', A_mat.get_shape().as_list())
    # Form b matrix
    b_mat = tf.matmul(Mb_tile, pred_pts_2_tile)
    print('--shape of b:', b_mat.get_shape().as_list())

    # Solve the Ax = b
    H_8el = tf.matrix_solve(A_mat , b_mat)  # BATCH_SIZE x 8.
    print('--shape of H_8el', H_8el)


    # Add ones to the last cols to reconstruct H for computing reprojection error
    h_ones = tf.ones([batch_size, 1, 1])
    H_9el = tf.concat([H_8el,h_ones],1)
    H_flat = tf.reshape(H_9el, [-1,9])
    H_mat = tf.reshape(H_flat,[-1,3,3])   # BATCH_SIZE x 3 x 3
    return H_mat

def UnsupervisedModel(patches_batch,batch_size,Ca,patches_b, img_a):
    # conv2d on 128 size
    s = tf.layers.conv2d(inputs=patches_batch,name='conv2d_1',kernel_size=(3,3),filters=64,padding='same')
    s = tf.layers.batch_normalization(s,name='batchnorm_1')
    s = tf.nn.relu(s,name='relu_1')

    s = tf.layers.conv2d(inputs=s,name='conv2d_2',kernel_size=(3,3),filters=64,padding='same')
    s = tf.layers.batch_normalization(s,name='batchnorm_2')
    s = tf.nn.relu(s,name='relu_2')

    s = tf.layers.max_pooling2d(s,pool_size=(2,2),strides=2)

    # conv2d on 64 size
    s = tf.layers.conv2d(inputs=s,name='conv2d_3',kernel_size=(3,3),filters=64,padding='same')
    s = tf.layers.batch_normalization(s,name='batchnorm_3')
    s = tf.nn.relu(s,name='relu_3')

    s = tf.layers.conv2d(inputs=s,name='conv2d_4',kernel_size=(3,3),filters=64,padding='same')
    s = tf.layers.batch_normalization(s,name='batchnorm_4')
    s = tf.nn.relu(s,name='relu_4')

    s = tf.layers.max_pooling2d(s,pool_size=(2,2),strides=2)

    # conv2d on 32 size
    s = tf.layers.conv2d(inputs=s,name='conv2d_5',kernel_size=(3,3),filters=128,padding='same')
    s = tf.layers.batch_normalization(s,name='batchnorm_5')
    s = tf.nn.relu(s,name='relu_5')

    s = tf.layers.conv2d(inputs=s,name='conv2d_6',kernel_size=(3,3),filters=128,padding='same')
    s = tf.layers.batch_normalization(s,name='batchnorm_6')
    s = tf.nn.relu(s,name='relu_6')

    s = tf.layers.max_pooling2d(s,pool_size=(2,2),strides=2)

    # conv2d on 16 size
    s = tf.layers.conv2d(inputs=s,name='conv2d_7',kernel_size=(3,3),filters=128,padding='same')
    s = tf.layers.batch_normalization(s,name='batchnorm_7')
    s = tf.nn.relu(s,name='relu_7')

    s = tf.layers.conv2d(inputs=s,name='conv2d_8',kernel_size=(3,3),filters=128,padding='same')
    s = tf.layers.batch_normalization(s,name='batchnorm_8')
    s = tf.nn.relu(s,name='relu_8')

    s = tf.layers.flatten(s)
    s = tf.layers.dense(inputs=s,name='dense1',units=1024)
    s = tf.nn.relu(s,name='relu_9')

    s = tf.layers.dropout(s,rate=0.5,training=True)

    H_4pt = tf.layers.dense(inputs=s,name='dense2',units=8)

    Ca = tf.reshape(Ca,[batch_size,8]) # flattening (4,2) shape

    H_matrix = TensorDLT(batch_size=batch_size, Ca=Ca,H_4pt=H_4pt)

    # Spatial transform
    H = 128.0
    W = 128.0
    M = np.array([[W/2, 0.0, W/2],
                 [0,H/2, H/2],
                 [0, 0, 1]],dtype=np.float32)
    M_inverse = np.linalg.inv(M)

    M_tensor = tf.constant(M,tf.float32)
    M_tensor_inverse = tf.constant(M_inverse,tf.float32)

    M_tile = tf.tile(tf.expand_dims(M_tensor,[0]),[batch_size,1,1])
    M_tile_inverse = tf.tile(tf.expand_dims(M_tensor_inverse,[0]),[batch_size,1,1])

    H_matrix = tf.matmul(tf.matmul(M_tile_inverse,H_matrix),M_tile)

    Ia = tf.slice(img_a,[0,0,0,0],[batch_size,128,128,1])
    warped_Ia,_ = transformer(Ia,H_matrix, (128,128))

    warped_Ia = tf.reshape(warped_Ia, [batch_size,128,128])
    print("warpedIa shape is:",tf.shape(warped_Ia))
    warped_Ia_gray = tf.reduce_mean(warped_Ia,axis=0)
    print("warpedIa gray shape is:",tf.shape(warped_Ia_gray))

    # warped_Ia_gray =tf.reduce_mean(warped_Ia,)
    pred_Ib = tf.reshape (warped_Ia_gray,[batch_size,128,128,1])

    return pred_Ib, patches_b

    


    

