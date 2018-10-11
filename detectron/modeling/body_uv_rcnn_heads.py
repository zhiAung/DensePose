# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core
from detectron.utils.net import get_group_gn
from detectron.core.config import cfg

import detectron.modeling.ResNet as ResNet
import detectron.utils.blob as blob_utils

from caffe2.python import workspace #modified by zhiang.hao
from detectron.utils.net import get_group_gn
# ---------------------------------------------------------------------------- #
# Body UV heads
# ---------------------------------------------------------------------------- #

def add_body_uv_outputs(model, blob_in, dim, pref=''):
    ####
    model.ConvTranspose(blob_in, 'AnnIndex_lowres'+pref, dim, 15,cfg.BODY_UV_RCNN.DECONV_KERNEL, pad=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1), stride=2, weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}), bias_init=('ConstantFill', {'value': 0.}))    
    ####
    model.ConvTranspose(blob_in, 'Index_UV_lowres'+pref, dim, cfg.BODY_UV_RCNN.NUM_PATCHES+1,cfg.BODY_UV_RCNN.DECONV_KERNEL, pad=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1), stride=2, weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}), bias_init=('ConstantFill', {'value': 0.}))    
    ####
    model.ConvTranspose(
        blob_in, 'U_lowres'+pref, dim, (cfg.BODY_UV_RCNN.NUM_PATCHES+1),
        cfg.BODY_UV_RCNN.DECONV_KERNEL, 
        pad=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1),
        stride=2,
        weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
        bias_init=('ConstantFill', {'value': 0.}))
    #####
    model.ConvTranspose(
            blob_in, 'V_lowres'+pref, dim, cfg.BODY_UV_RCNN.NUM_PATCHES+1,
            cfg.BODY_UV_RCNN.DECONV_KERNEL,
            pad=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1),
            stride=2,
            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.}))
    ####
    blob_Ann_Index = model.BilinearInterpolation('AnnIndex_lowres'+pref, 'AnnIndex'+pref,  cfg.BODY_UV_RCNN.NUM_PATCHES+1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1, cfg.BODY_UV_RCNN.UP_SCALE)
    blob_Index = model.BilinearInterpolation('Index_UV_lowres'+pref, 'Index_UV'+pref,  cfg.BODY_UV_RCNN.NUM_PATCHES+1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1, cfg.BODY_UV_RCNN.UP_SCALE)
    blob_U = model.BilinearInterpolation('U_lowres'+pref, 'U_estimated'+pref,  cfg.BODY_UV_RCNN.NUM_PATCHES+1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1, cfg.BODY_UV_RCNN.UP_SCALE)
    blob_V = model.BilinearInterpolation('V_lowres'+pref, 'V_estimated'+pref,  cfg.BODY_UV_RCNN.NUM_PATCHES+1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1, cfg.BODY_UV_RCNN.UP_SCALE)
    ###
    return blob_U,blob_V,blob_Index,blob_Ann_Index
    
    
    
def add_body_uv_outputs_test2(model, blob_in, blob_in2, dim, pref=''):
    ####
    model.ConvTranspose(blob_in, 'AnnIndex_lowres'+pref, dim, 15,cfg.BODY_UV_RCNN.DECONV_KERNEL, pad=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1), stride=2, weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}), bias_init=('ConstantFill', {'value': 0.}))    
    ####
    model.ConvTranspose(blob_in, 'Index_UV_lowres'+pref, dim, cfg.BODY_UV_RCNN.NUM_PATCHES+1,cfg.BODY_UV_RCNN.DECONV_KERNEL, pad=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1), stride=2, weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}), bias_init=('ConstantFill', {'value': 0.}))    
    ####
    model.ConvTranspose(
        blob_in2, 'U_lowres'+pref, dim, (cfg.BODY_UV_RCNN.NUM_PATCHES+1),
        cfg.BODY_UV_RCNN.DECONV_KERNEL, 
        pad=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1),
        stride=2,
        weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
        bias_init=('ConstantFill', {'value': 0.}))
    #####
    model.ConvTranspose(
            blob_in2, 'V_lowres'+pref, dim, cfg.BODY_UV_RCNN.NUM_PATCHES+1,
            cfg.BODY_UV_RCNN.DECONV_KERNEL,
            pad=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1),
            stride=2,
            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.}))
    ####
    blob_Ann_Index = model.BilinearInterpolation('AnnIndex_lowres'+pref, 'AnnIndex'+pref,  cfg.BODY_UV_RCNN.NUM_PATCHES+1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1, cfg.BODY_UV_RCNN.UP_SCALE)
    blob_Index = model.BilinearInterpolation('Index_UV_lowres'+pref, 'Index_UV'+pref,  cfg.BODY_UV_RCNN.NUM_PATCHES+1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1, cfg.BODY_UV_RCNN.UP_SCALE)
    blob_U = model.BilinearInterpolation('U_lowres'+pref, 'U_estimated'+pref,  cfg.BODY_UV_RCNN.NUM_PATCHES+1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1, cfg.BODY_UV_RCNN.UP_SCALE)
    blob_V = model.BilinearInterpolation('V_lowres'+pref, 'V_estimated'+pref,  cfg.BODY_UV_RCNN.NUM_PATCHES+1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1, cfg.BODY_UV_RCNN.UP_SCALE)
    ###
    return blob_U,blob_V,blob_Index,blob_Ann_Index

def add_body_uv_outputs_test1(model, blob_in, dim, pref=''):
    ####
    model.ConvTranspose(blob_in, 'AnnIndex_lowres'+pref, dim, 15, 3, pad=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1), stride=1, weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}), bias_init=('ConstantFill', {'value': 0.}))    
    ####
    model.ConvTranspose(blob_in, 'Index_UV_lowres'+pref, dim, cfg.BODY_UV_RCNN.NUM_PATCHES+1,3, pad=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1), stride=1, weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}), bias_init=('ConstantFill', {'value': 0.}))    
    ####
    model.ConvTranspose(
        blob_in, 'U_lowres'+pref, dim, (cfg.BODY_UV_RCNN.NUM_PATCHES+1),
        3, #default
        pad=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1),
        stride=1,
        weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
        bias_init=('ConstantFill', {'value': 0.}))
    #####
    model.ConvTranspose(
            blob_in, 'V_lowres'+pref, dim, cfg.BODY_UV_RCNN.NUM_PATCHES+1,
            3,
            pad=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1),
            stride=1,
            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.}))
    ####
    blob_Ann_Index = model.BilinearInterpolation('AnnIndex_lowres'+pref, 'AnnIndex'+pref,  cfg.BODY_UV_RCNN.NUM_PATCHES+1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1, cfg.BODY_UV_RCNN.UP_SCALE)
    blob_Index = model.BilinearInterpolation('Index_UV_lowres'+pref, 'Index_UV'+pref,  cfg.BODY_UV_RCNN.NUM_PATCHES+1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1, cfg.BODY_UV_RCNN.UP_SCALE)
    blob_U = model.BilinearInterpolation('U_lowres'+pref, 'U_estimated'+pref,  cfg.BODY_UV_RCNN.NUM_PATCHES+1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1, cfg.BODY_UV_RCNN.UP_SCALE)
    blob_V = model.BilinearInterpolation('V_lowres'+pref, 'V_estimated'+pref,  cfg.BODY_UV_RCNN.NUM_PATCHES+1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1, cfg.BODY_UV_RCNN.UP_SCALE)
    ###
    return blob_U,blob_V,blob_Index,blob_Ann_Index    


def add_body_uv_losses(model, pref=''):

    ## Reshape for GT blobs.
    
    model.net.Reshape( ['body_uv_X_points'], ['X_points_reshaped'+pref, 'X_points_shape'+pref],  shape=( -1 ,1 ) )
    model.net.Reshape( ['body_uv_Y_points'], ['Y_points_reshaped'+pref, 'Y_points_shape'+pref],  shape=( -1 ,1 ) )
    model.net.Reshape( ['body_uv_I_points'], ['I_points_reshaped'+pref, 'I_points_shape'+pref],  shape=( -1 ,1 ) )
    model.net.Reshape( ['body_uv_Ind_points'], ['Ind_points_reshaped'+pref, 'Ind_points_shape'+pref],  shape=( -1 ,1 ) )
    ## Concat Ind,x,y to get Coordinates blob.
    model.net.Concat( ['Ind_points_reshaped'+pref,'X_points_reshaped'+pref, \
                       'Y_points_reshaped'+pref],['Coordinates'+pref,'Coordinate_Shapes'+pref ], axis = 1 )
    ##
    ### Now reshape UV blobs, such that they are 1x1x(196*NumSamples)xNUM_PATCHES 
    ## U blob to
    ##
    model.net.Reshape(['body_uv_U_points'], \
                      ['U_points_reshaped'+pref, 'U_points_old_shape'+pref],\
                      shape=(-1,cfg.BODY_UV_RCNN.NUM_PATCHES+1,196))
    model.net.Transpose(['U_points_reshaped'+pref] ,['U_points_reshaped_transpose'+pref],axes=(0,2,1) )
    model.net.Reshape(['U_points_reshaped_transpose'+pref], \
                      ['U_points'+pref, 'U_points_old_shape2'+pref], \
                      shape=(1,1,-1,cfg.BODY_UV_RCNN.NUM_PATCHES+1))
    ## V blob
    ##
    model.net.Reshape(['body_uv_V_points'], \
                      ['V_points_reshaped'+pref, 'V_points_old_shape'+pref],\
                      shape=(-1,cfg.BODY_UV_RCNN.NUM_PATCHES+1,196))
    model.net.Transpose(['V_points_reshaped'+pref] ,['V_points_reshaped_transpose'+pref],axes=(0,2,1) )
    model.net.Reshape(['V_points_reshaped_transpose'+pref], \
                      ['V_points'+pref, 'V_points_old_shape2'+pref], \
                      shape=(1,1,-1,cfg.BODY_UV_RCNN.NUM_PATCHES+1))
    ###
    ## UV weights blob
    ##
    model.net.Reshape(['body_uv_point_weights'], \
                      ['Uv_point_weights_reshaped'+pref, 'Uv_point_weights_old_shape'+pref],\
                      shape=(-1,cfg.BODY_UV_RCNN.NUM_PATCHES+1,196))
    model.net.Transpose(['Uv_point_weights_reshaped'+pref] ,['Uv_point_weights_reshaped_transpose'+pref],axes=(0,2,1) )
    model.net.Reshape(['Uv_point_weights_reshaped_transpose'+pref], \
                      ['Uv_point_weights'+pref, 'Uv_point_weights_old_shape2'+pref], \
                      shape=(1,1,-1,cfg.BODY_UV_RCNN.NUM_PATCHES+1))

    #####################
    ###  Pool IUV for points via bilinear interpolation.
    model.PoolPointsInterp(['U_estimated','Coordinates'+pref], ['interp_U'+pref])
    model.PoolPointsInterp(['V_estimated','Coordinates'+pref], ['interp_V'+pref])
    model.PoolPointsInterp(['Index_UV'+pref,'Coordinates'+pref], ['interp_Index_UV'+pref])

    ## Reshape interpolated UV coordinates to apply the loss.
    
    model.net.Reshape(['interp_U'+pref], \
                      ['interp_U_reshaped'+pref, 'interp_U_shape'+pref],\
                      shape=(1, 1, -1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1))
    
    model.net.Reshape(['interp_V'+pref], \
                      ['interp_V_reshaped'+pref, 'interp_V_shape'+pref],\
                      shape=(1, 1, -1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1))
    ###

    ### Do the actual labels here !!!!
    model.net.Reshape( ['body_uv_ann_labels'],    \
                      ['body_uv_ann_labels_reshaped'   +pref, 'body_uv_ann_labels_old_shape'+pref], \
                      shape=(-1, cfg.BODY_UV_RCNN.HEATMAP_SIZE , cfg.BODY_UV_RCNN.HEATMAP_SIZE))
    
    model.net.Reshape( ['body_uv_ann_weights'],   \
                      ['body_uv_ann_weights_reshaped'   +pref, 'body_uv_ann_weights_old_shape'+pref], \
                      shape=( -1 , cfg.BODY_UV_RCNN.HEATMAP_SIZE , cfg.BODY_UV_RCNN.HEATMAP_SIZE))
    ###
    model.net.Cast( ['I_points_reshaped'+pref], ['I_points_reshaped_int'+pref], to=core.DataType.INT32)
    ### Now add the actual losses 
    ## The mask segmentation loss (dense)
    probs_seg_AnnIndex, loss_seg_AnnIndex = model.net.SpatialSoftmaxWithLoss( \
                          ['AnnIndex'+pref, 'body_uv_ann_labels_reshaped'+pref,'body_uv_ann_weights_reshaped'+pref],\
                          ['probs_seg_AnnIndex'+pref,'loss_seg_AnnIndex'+pref], \
                           scale=cfg.BODY_UV_RCNN.INDEX_WEIGHTS / cfg.NUM_GPUS)
    ## Point Patch Index Loss.
    probs_IndexUVPoints, loss_IndexUVPoints = model.net.SoftmaxWithLoss(\
                          ['interp_Index_UV'+pref,'I_points_reshaped_int'+pref],\
                          ['probs_IndexUVPoints'+pref,'loss_IndexUVPoints'+pref], \
                          scale=cfg.BODY_UV_RCNN.PART_WEIGHTS / cfg.NUM_GPUS, spatial=0)
    ## U and V point losses.
    
    loss_Upoints = model.net.SmoothL1Loss( \
                          ['interp_U_reshaped'+pref, 'U_points'+pref, \
                               'Uv_point_weights'+pref, 'Uv_point_weights'+pref], \
                          'loss_Upoints'+pref, \
                            scale=cfg.BODY_UV_RCNN.POINT_REGRESSION_WEIGHTS  / cfg.NUM_GPUS)
    
    loss_Vpoints = model.net.SmoothL1Loss( \
                          ['interp_V_reshaped'+pref, 'V_points'+pref, \
                               'Uv_point_weights'+pref, 'Uv_point_weights'+pref], \
                          'loss_Vpoints'+pref, scale=cfg.BODY_UV_RCNN.POINT_REGRESSION_WEIGHTS / cfg.NUM_GPUS)
    ## Add the losses.
    loss_gradients = blob_utils.get_loss_gradients(model, \
                       [ loss_Upoints, loss_Vpoints, loss_seg_AnnIndex, loss_IndexUVPoints])
    model.losses = list(set(model.losses + \
                       ['loss_Upoints'+pref , 'loss_Vpoints'+pref , \
                        'loss_seg_AnnIndex'+pref ,'loss_IndexUVPoints'+pref]))

    return loss_gradients


# ---------------------------------------------------------------------------- #
# Body UV heads
# ---------------------------------------------------------------------------- #

def add_ResNet_roi_conv5_head_for_bodyUV(
        model, blob_in, dim_in, spatial_scale
):
    """Add a ResNet "conv5" / "stage5" head for body UV prediction."""
    model.RoIFeatureTransform(
        blob_in, '_[body_uv]_pool_5',#default=_[body_uv]_pool5
        blob_rois='body_uv_rois',
        method=cfg.BODY_UV_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.BODY_UV_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.BODY_UV_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)
    # Using the prefix '_[body_uv]_' to 'res5' enables initializing the head's
    # parameters using pretrained 'res5' parameters if given (see
    # utils.net.initialize_from_weights_file)
    s, dim_in = ResNet.add_stage(
        model,
        '_[body_uv]_res_5',
        '_[body_uv]_pool_5',
        8,#default 3
        dim_in,
        2048,
        512,
        cfg.BODY_UV_RCNN.DILATION,
        stride_init=int(cfg.BODY_UV_RCNN.ROI_XFORM_RESOLUTION / 7)
    )
    return s, 2048
    
def add_roi_body_uv_head_v1convX(model, blob_in, dim_in, spatial_scale):
    """v1convX design: X * (conv)."""
    hidden_dim = cfg.BODY_UV_RCNN.CONV_HEAD_DIM
    kernel_size = cfg.BODY_UV_RCNN.CONV_HEAD_KERNEL
    pad_size = kernel_size // 2
    current = model.RoIFeatureTransform(
        blob_in,
        '_[body_uv]_roi_feat',
        blob_rois='body_uv_rois',
        method=cfg.BODY_UV_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.BODY_UV_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.BODY_UV_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
    for i in range(cfg.BODY_UV_RCNN.NUM_STACKED_CONVS):
        current = model.Conv(
            current,
            'body_conv_fcn' + str(i + 1),# modified by zhiang.hao
            dim_in,
            hidden_dim,
            kernel_size,
            stride=1,
            pad=pad_size,
            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        current = model.Relu(current, current)
        dim_in = hidden_dim  
    return current, hidden_dim
    

def add_roi_body_uv_head_v1convX_Decoupling(model, blob_in, dim_in, spatial_scale):
    """v1convX design: X * (conv)."""
    hidden_dim = cfg.BODY_UV_RCNN.CONV_HEAD_DIM
    kernel_size = cfg.BODY_UV_RCNN.CONV_HEAD_KERNEL
    pad_size = kernel_size // 2
    current = model.RoIFeatureTransform(
        blob_in,
        '_[body_uv]_roi_feat',
        blob_rois='body_uv_rois',
        method=cfg.BODY_UV_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.BODY_UV_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.BODY_UV_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
    current1=current
    current2=current
    dim_in2=dim_in
    for i in range(cfg.BODY_UV_RCNN.NUM_STACKED_CONVS):
        current1 = model.Conv(
            current1,
            'body_conv_fcn_sen' + str(i + 1),# modified by zhiang.hao
            dim_in,
            hidden_dim,
            kernel_size,
            stride=1,
            pad=pad_size,
            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        current1 = model.Relu(current1, current1)
        dim_in = hidden_dim
    for i in range(cfg.BODY_UV_RCNN.NUM_STACKED_CONVS):
        current2 = model.Conv(
            current2,
            'body_conv_fcn_uv' + str(i + 1),# modified by zhiang.hao
            dim_in2,
            hidden_dim,
            kernel_size,
            stride=1,
            pad=pad_size,
            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        current2 = model.Relu(current2, current2)
        dim_in2 = hidden_dim
    return current1,current2, hidden_dim
    
    
def add_roi_body_uv_head_v1convX_test(model, blob_in, dim_in, spatial_scale):
    """v1convX design: X * (conv)."""
    hidden_dim = cfg.BODY_UV_RCNN.CONV_HEAD_DIM
    kernel_size = cfg.BODY_UV_RCNN.CONV_HEAD_KERNEL
    pad_size = kernel_size // 2
    current = model.RoIFeatureTransform(
        blob_in,
        '_[body_uv]_roi_feat',
        blob_rois='body_uv_rois',
        method=cfg.BODY_UV_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.BODY_UV_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.BODY_UV_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
    for i in range(cfg.BODY_UV_RCNN.NUM_STACKED_CONVS):
        current = model.Conv(
            current,
            'conv_fcn' + str(i + 1),# modified by zhiang.hao
            dim_in,
            hidden_dim,
            kernel_size,
            stride=1,
            pad=pad_size,
            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        current = model.Relu(current, current)
        dim_in = hidden_dim  
    return current, hidden_dim    


def add_roi_body_uv_head_Modification_densenet(model, blob_in, dim_in, spatial_scale):
    hidden_dim = cfg.BODY_UV_RCNN.CONV_HEAD_DIM
    kernel_size = cfg.BODY_UV_RCNN.CONV_HEAD_KERNEL
    pad_size = kernel_size // 2
    current = model.RoIFeatureTransform(
        blob_in,
        '_[body_uv]_roi_feat',
        blob_rois='body_uv_rois',
        method=cfg.BODY_UV_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.BODY_UV_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.BODY_UV_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
    current = model.Conv(
            current,
            'conv_base', # modified by zhiang.hao
            dim_in,
            hidden_dim,
            kernel_size,
            stride=1,
            pad=pad_size,
            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.})
        )
    n0=3
    n1=4
    n2=5
    sc0 = basic_gn_shortcut(model, 'body_uv_dense0', current, hidden_dim, dim_out=512, stride=1)
    bs0, dim_in = add_stage(model, 'body_uv0', current, n0, hidden_dim, dim_out=512, dim_inner=512, dilation=1)
    #s1 = model.net.Sum([bs0, sc0], 'body_uv_dense0_sum')
    s1 = model.net.Sum([bs0, sc0], 'body_uv_dense0_sum',broadcast=1)
    s1 = model.Relu(s1, s1)
    sc1 = basic_gn_shortcut(model, 'body_uv_dense1', s1, dim_in, dim_out=512, stride=1)
    bs1, dim_in = add_stage(model, 'body_uv1', s1, n1, dim_in, dim_out=512, dim_inner=512, dilation=1)
    #s1 = model.net.Sum([bs1, sc1, sc0], 'body_uv_dense1_sum')
    s1 = model.net.Sum([bs1, sc1, sc0], 'body_uv_dense1_sum',broadcast=1)
    s1 = model.Relu(s1, s1)
    sc2 = basic_gn_shortcut(model, 'body_uv_dense2', s1, dim_in, dim_out=512, stride=1)
    bs2, dim_in = add_stage(model, 'body_uv2', s1, n2, dim_in, dim_out=512, dim_inner=512, dilation=1)
    #s1 = model.net.Sum([bs2, sc2, sc1, sc0], 'body_uv_dense2_sum')
    s1 = model.net.Sum([bs2, sc2, sc1, sc0], 'body_uv_dense2_sum',broadcast=1)
    s1 = model.Relu(s1, s1)
    return s1, dim_in

def add_roi_body_uv_head_Modification_densenet1(model, blob_in, dim_in, spatial_scale):
    current = model.RoIFeatureTransform(
        blob_in,
        '_[body_uv]_roi_feat',
        blob_rois='body_uv_rois',
        method=cfg.BODY_UV_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.BODY_UV_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.BODY_UV_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
    n0=3
    n1=4
    n2=3
    sc0 = basic_gn_shortcut(model, 'body_uv_dense0', current, dim_in, dim_out=256, stride=1)
    bs0, dim_in = add_stage(model, 'body_uv0', current, n0, dim_in, dim_out=256, dim_inner=256, dilation=1)
    #s1 = model.net.Sum([bs0, sc0], 'body_uv_dense0_sum')
    s1 = model.net.Sum([bs0, sc0], 'body_uv_dense0_sum',broadcast=1)
    s1 = model.Relu(s1, s1)
    sc1 = basic_gn_shortcut(model, 'body_uv_dense1', s1, dim_in, dim_out=256, stride=1)
    bs1, dim_in = add_stage(model, 'body_uv1', s1, n1, dim_in, dim_out=256, dim_inner=256, dilation=1)
    #s1 = model.net.Sum([bs1, sc1, sc0], 'body_uv_dense1_sum')
    s1 = model.net.Sum([bs1, sc1, sc0], 'body_uv_dense1_sum',broadcast=1)
    s1 = model.Relu(s1, s1)
    sc2 = basic_gn_shortcut(model, 'body_uv_dense2', s1, dim_in, dim_out=256, stride=1)
    bs2, dim_in = add_stage(model, 'body_uv2', s1, n2, dim_in, dim_out=256, dim_inner=256, dilation=1)
    #s1 = model.net.Sum([bs2, sc2, sc1, sc0], 'body_uv_dense2_sum')
    s1 = model.net.Sum([bs2, sc2, sc1, sc0], 'body_uv_dense2_sum',broadcast=1)
    s1 = model.Relu(s1, s1)
    return s1, dim_in



def add_roi_body_uv_head_Modification_resnet(model, blob_in, dim_in, spatial_scale):
    current = model.RoIFeatureTransform(
        blob_in,
        '_[body_uv]_roi_feat',
        blob_rois='body_uv_rois',
        method=cfg.BODY_UV_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.BODY_UV_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.BODY_UV_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
    n0=2
    n1=5
    n2=3
    s, dim_in = add_stage(model, 'body_uv0', current, n0, dim_in, dim_out=512, dim_inner=256, dilation=1)
    s, dim_in = add_stage(model, 'body_uv1', s, n1, dim_in, dim_out=512, dim_inner=256, dilation=1)
    s, dim_in = add_stage(model, 'body_uv2', s, n2, dim_in, dim_out=512, dim_inner=256, dilation=1)
    return s, dim_in


def add_stage(
    model,
    prefix,
    blob_in,
    n,
    dim_in,
    dim_out,
    dim_inner,
    dilation,
):
    """Add a ResNet stage to the model by stacking n residual blocks."""
    # e.g., prefix = res2
    for i in range(n):
        blob_in = add_residual_block(
            model,
            '{}_{}'.format(prefix, i),
            blob_in,
            dim_in,
            dim_out,
            dim_inner,
            dilation,
            # Not using inplace for the last block;
            # it may be fetched externally or used by FPN
            inplace_sum=i < n - 1
        )
        dim_in = dim_out
    return blob_in, dim_in


def add_residual_block(
    model,
    prefix,
    blob_in,
    dim_in,
    dim_out,
    dim_inner,
    dilation=1,
    inplace_sum=False
):
    
    tr = bottleneck_gn_transformation(
        model,
        blob_in,
        dim_in,
        dim_out,
        prefix,
        dim_inner,
        group=1,
        dilation=dilation
    )

    sc = basic_gn_shortcut(model, prefix, blob_in, dim_in, dim_out, stride=1)
    
    if inplace_sum:
        s = model.net.Sum([tr, sc], tr)
    else:
        s = model.net.Sum([tr, sc], prefix + '_sum')
    return model.Relu(s, s)

def bottleneck_gn_transformation(
    model,
    blob_in,
    dim_in,
    dim_out,
    prefix,
    dim_inner,
    dilation=1,
    group=1
):
    """Add a bottleneck transformation with GroupNorm to the model."""
    # In original resnet, stride=2 is on 1x1.
    # In fb.torch resnet, stride=2 is on 3x3.
    (str1x1, str3x3) = (1, 1)    ########################################

    # conv 1x1 -> GN -> ReLU
    cur = model.ConvGN(
        blob_in,
        prefix + '_branch2a',
        dim_in,
        dim_inner,
        kernel=1,
        group_gn=get_group_gn(dim_inner),
        stride=str1x1,
        pad=0,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.})
    )
    cur = model.Relu(cur, cur)

    # conv 3x3 -> GN -> ReLU
    cur = model.ConvGN(
        cur,
        prefix + '_branch2b',
        dim_inner,
        dim_inner,
        kernel=3,
        group_gn=get_group_gn(dim_inner),
        stride=str3x3,
        pad=1 * dilation,
        dilation=dilation,
        group=group,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.})
    )
    cur = model.Relu(cur, cur)

    # conv 1x1 -> GN (no ReLU)
    cur = model.ConvGN(
        cur,
        prefix + '_branch2c',
        dim_inner,
        dim_out,
        kernel=1,
        group_gn=get_group_gn(dim_out),
        stride=1,
        pad=0,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.})
    )
    return cur

def basic_gn_shortcut(model, prefix, blob_in, dim_in, dim_out, stride):
    if dim_in == dim_out:
        return blob_in

    # output name is prefix + '_branch1_gn'
    return model.ConvGN(
        blob_in,
        prefix + '_branch1',
        dim_in,
        dim_out,
        kernel=1,
        group_gn=get_group_gn(dim_out),
        stride=stride,
        pad=0,
        group=1,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.})
    )

def add_roi_body_uv_head_increase_resolution(model, blob_in, dim_in, spatial_scale):
    """v1convX design: X * (conv)."""
    hidden_dim = cfg.BODY_UV_RCNN.CONV_HEAD_DIM
    kernel_size = cfg.BODY_UV_RCNN.CONV_HEAD_KERNEL
    pad_size = kernel_size // 2
    current = model.RoIFeatureTransform(
        blob_in,
        '_[body_uv]_roi_feat',
        blob_rois='body_uv_rois',
        method=cfg.BODY_UV_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.BODY_UV_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.BODY_UV_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
    
    for i in range(cfg.BODY_UV_RCNN.NUM_STACKED_CONVS):
        current = model.Conv(
            current,
            'body_conv_fcn' + str(i + 1),# modified by zhiang.hao
            dim_in,
            hidden_dim,
            kernel_size,
            stride=1,
            pad=pad_size,
            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        current = model.Relu(current, current)
        dim_in = hidden_dim 
    
        #dim_in = hidden_dim  
    return current, hidden_dim    
    
def add_roi_body_uv_head_cascad_body(model, blob_in, dim_in, spatial_scale):
    """v1convX design: X * (conv)."""
    hidden_dim = cfg.BODY_UV_RCNN.CONV_HEAD_DIM
    kernel_size = cfg.BODY_UV_RCNN.CONV_HEAD_KERNEL
    pad_size = kernel_size // 2
    current = model.RoIFeatureTransform(
        blob_in,
        '_[body_uv]_roi_feat',
        blob_rois='body_uv_rois',
        method=cfg.BODY_UV_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.BODY_UV_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.BODY_UV_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    for i in range(cfg.BODY_UV_RCNN.NUM_STACKED_CONVS):
        current = model.Conv(
            current,
            'body_conv_fcn' + str(i + 1),# modified by zhiang.hao
            dim_in,
            hidden_dim,
            kernel_size,
            stride=1,
            pad=pad_size,
            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        
        current = model.Relu(current, current)
        if i==1:
          current_1 = model.Conv(
              current,
              'body_conv_fcn_branch' + str(1 + 1),
              dim_in,
              hidden_dim,
              kernel_size,
              stride=1,
              pad=pad_size,
              weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),
              bias_init=('ConstantFill', {'value': 0.})
          )
          current_1 = model.Relu(current_1, current_1)
        if i==3:
          current_3 = model.Conv(
              current,
              'body_conv_fcn_branch' + str(3 + 1),
              dim_in,
              hidden_dim,
              kernel_size,
              stride=1,
              pad=pad_size,
              weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),
              bias_init=('ConstantFill', {'value': 0.})
          )
          current_3 = model.Relu(current_3, current_3)
          
        dim_in = hidden_dim    
    #return current,current_1,current_3, hidden_dim
    current_sum=model.Sum([current, current_1, current_3], "sum_branch" ,broadcast=1)
    return current, current_sum, hidden_dim


def add_roi_body_uv_head_ResNet24(model, blob_in, dim_in, spatial_scale):
    """v1convX design: X * (conv)."""
    hidden_dim = cfg.BODY_UV_RCNN.CONV_HEAD_DIM
    kernel_size = cfg.BODY_UV_RCNN.CONV_HEAD_KERNEL
    pad_size = kernel_size // 2
    current = model.RoIFeatureTransform(
        blob_in,
        '_[body_uv]_roi_feat',
        blob_rois='body_uv_rois',
        method=cfg.BODY_UV_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.BODY_UV_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.BODY_UV_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    for i in range(cfg.BODY_UV_RCNN.NUM_STACKED_CONVS):
        current = model.ConvGN(
                current,
                'body_conv_fcn_1_' + str(i + 1),
                dim_in,
                hidden_dim,
                group_gn=get_group_gn(hidden_dim),
                kernel=kernel_size,
                pad=pad_size,
                stride=1,
                weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.01}),
                bias_init=('ConstantFill', {'value': 0.})
            )
        current = model.Relu(current, current)
        if i==1:
          current_1 = model.ConvGN(
              current,
              'body_conv_fcn_1_branch' + str(1 + 1),
              dim_in,
              hidden_dim,
              group_gn=get_group_gn(hidden_dim),
              kernel=kernel_size,
              pad=pad_size,
              stride=1,
              weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.01}),
              bias_init=('ConstantFill', {'value': 0.})
          )
          current_1 = model.Relu(current_1, current_1)
        dim_in = hidden_dim
    current=model.Sum([current, current_1], "sum_branch_1" ,broadcast=1)
    for i in range(cfg.BODY_UV_RCNN.NUM_STACKED_CONVS):
        current = model.ConvGN(
            current,
            'body_conv_fcn_2_' + str(i + 1),
            hidden_dim,
            hidden_dim,
            group_gn=get_group_gn(hidden_dim),
            kernel=kernel_size,
            pad=pad_size,
            stride=1,
            weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        
        current = model.Relu(current, current)
        if i==1:
          current_2 = model.ConvGN(
              current,
              'body_conv_fcn_2_branch' + str(1 + 1),
              hidden_dim,
              hidden_dim,
              group_gn=get_group_gn(hidden_dim),
              kernel=kernel_size,
              pad=pad_size,
              stride=1,
              weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.01}),
              bias_init=('ConstantFill', {'value': 0.})
          )
          current_2 = model.Relu(current_2, current_2)
    current=model.Sum([current, current_2], "sum_branch_2" ,broadcast=1)    
    for i in range(cfg.BODY_UV_RCNN.NUM_STACKED_CONVS):
        current = model.ConvGN(
            current,
            'body_conv_fcn_3_' + str(i + 1),
            hidden_dim,
            hidden_dim,
            group_gn=get_group_gn(hidden_dim),
            kernel=kernel_size,
            pad=pad_size,
            stride=1,
            weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        
        current = model.Relu(current, current)
        if i==1:
          current_3 = model.ConvGN(
              current,
              'body_conv_fcn_3_branch' + str(1 + 1),
              hidden_dim,
              hidden_dim,
              group_gn=get_group_gn(hidden_dim),
              kernel=kernel_size,
              pad=pad_size,
              stride=1,
              weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.01}),
              bias_init=('ConstantFill', {'value': 0.})
          )
          current_3 = model.Relu(current_3, current_3)
    current=model.Sum([current, current_3], "sum_branch_3" ,broadcast=1)
    return current, hidden_dim
    


def add_roi_body_uv_head_ResNet32(model, blob_in, dim_in, spatial_scale):
    """v1convX design: X * (conv)."""
    hidden_dim = cfg.BODY_UV_RCNN.CONV_HEAD_DIM
    kernel_size = cfg.BODY_UV_RCNN.CONV_HEAD_KERNEL
    pad_size = kernel_size // 2
    current = model.RoIFeatureTransform(
        blob_in,
        '_[body_uv]_roi_feat',
        blob_rois='body_uv_rois',
        method=cfg.BODY_UV_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.BODY_UV_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.BODY_UV_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    for i in range(cfg.BODY_UV_RCNN.NUM_STACKED_CONVS):
        current = model.ConvGN(
                current,
                'body_conv_fcn_1_' + str(i + 1),
                dim_in,
                hidden_dim,
                group_gn=get_group_gn(hidden_dim),
                kernel=kernel_size,
                pad=pad_size,
                stride=1,
                weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.01}),
                bias_init=('ConstantFill', {'value': 0.})
            )
        current = model.Relu(current, current)
        dim_in = hidden_dim
        if i==0:
          current_1 = model.ConvGN(
              current,
              'body_conv_fcn_1_shortcut' + str(1 + 1),
              hidden_dim,
              hidden_dim,
              group_gn=get_group_gn(hidden_dim),
              kernel=kernel_size,
              pad=pad_size,
              stride=1,
              weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.01}),
              bias_init=('ConstantFill', {'value': 0.})
          )
          current_1 = model.Relu(current_1, current_1)
    current=model.Sum([current, current_1], "sum_branch_1" ,broadcast=1)
    
    current=block_4conv(model, current, hidden_dim, kernel_size, pad_size, 'body_conv_fcn_2_', 'body_conv_fcn_2_shortcut', 'sum_branch_2')
    current=block_4conv(model, current, hidden_dim, kernel_size, pad_size, 'body_conv_fcn_3_', 'body_conv_fcn_3_shortcut', 'sum_branch_3')
    current=block_4conv(model, current, hidden_dim, kernel_size, pad_size, 'body_conv_fcn_4_', 'body_conv_fcn_4_shortcut', 'sum_branch_4')
    current=block_4conv(model, current, hidden_dim, kernel_size, pad_size, 'body_conv_fcn_5_', 'body_conv_fcn_5_shortcut', 'sum_branch_5')
    current=block_4conv(model, current, hidden_dim, kernel_size, pad_size, 'body_conv_fcn_6_', 'body_conv_fcn_6_shortcut', 'sum_branch_6')
    current=block_4conv(model, current, hidden_dim, kernel_size, pad_size, 'body_conv_fcn_7_', 'body_conv_fcn_7_shortcut', 'sum_branch_7')
    current=block_4conv(model, current, hidden_dim, kernel_size, pad_size, 'body_conv_fcn_8_', 'body_conv_fcn_8_shortcut', 'sum_branch_8')
    return current, hidden_dim

def block_4conv(model,current,hidden_dim, kernel_size, pad_size, prefix_master, prefix_branch, prefix_sum):
    current_short_cut = model.ConvGN(
              current,
              prefix_branch,
              hidden_dim,
              hidden_dim,
              group_gn=get_group_gn(hidden_dim),
              kernel=kernel_size,
              pad=pad_size,
              stride=1,
              weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.01}),
              bias_init=('ConstantFill', {'value': 0.})
          )
    current_short_cut = model.Relu(current_short_cut, current_short_cut)
    for i in range(cfg.BODY_UV_RCNN.NUM_STACKED_CONVS):
        current = model.ConvGN(
            current,
            prefix_master + str(i + 1),
            hidden_dim,
            hidden_dim,
            group_gn=get_group_gn(hidden_dim),
            kernel=kernel_size,
            pad=pad_size,
            stride=1,
            weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        current = model.Relu(current, current)   
    current=model.Sum([current, current_short_cut], prefix_sum ,broadcast=1)
    return current

#hourglass+res
def add_roi_body_uv_head_v1convX_hourglass_res(model, blob_in, dim_in, spatial_scale):
    """v1convX design: X * (conv)."""
    hidden_dim = cfg.BODY_UV_RCNN.CONV_HEAD_DIM
    kernel_size = cfg.BODY_UV_RCNN.CONV_HEAD_KERNEL
    pad_size = kernel_size // 2
    current = model.RoIFeatureTransform(
        blob_in,
        '_[body_uv]_roi_feat',
        blob_rois='body_uv_rois',
        method=cfg.BODY_UV_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.BODY_UV_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.BODY_UV_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    for i in range(cfg.BODY_UV_RCNN.NUM_STACKED_CONVS):
        current = model.Conv(
            current,
            'body_conv_fcn_1_' + str(i + 1),# modified by zhiang.hao
            dim_in,
            hidden_dim,
            kernel_size,
            stride=1,
            pad=pad_size,
            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        
        current = model.Relu(current, current)
        if i==1:
          current_1 = model.Conv(
              current,
              'body_conv_fcn_1_branch' + str(1 + 1),
              dim_in,
              hidden_dim,
              kernel_size,
              stride=1,
              pad=pad_size,
              weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),
              bias_init=('ConstantFill', {'value': 0.})
          )
          current_1 = model.Relu(current_1, current_1)
        dim_in = hidden_dim
        
    current=model.Sum([current, current_1], "sum_branch_1" ,broadcast=1)    
    for i in range(cfg.BODY_UV_RCNN.NUM_STACKED_CONVS):
        current = model.Conv(
            current,
            'body_conv_fcn_2_' + str(i + 1),
            hidden_dim,
            hidden_dim,
            kernel_size,
            stride=1,
            pad=pad_size,
            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        
        current = model.Relu(current, current)
        if i==1:
          current_2 = model.Conv(
              current,
              'body_conv_fcn_2_branch' + str(1 + 1),
              hidden_dim,
              hidden_dim,
              kernel_size,
              stride=1,
              pad=pad_size,
              weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),
              bias_init=('ConstantFill', {'value': 0.})
          )
          current_2 = model.Relu(current_2, current_2)
    current=model.Sum([current, current_2], "sum_branch_2" ,broadcast=1)
#    for i in range(cfg.BODY_UV_RCNN.NUM_STACKED_CONVS):
#        current = model.Conv(
#            current,
#            'body_conv_fcn' + str(i + 1),
#            dim_in,
#            hidden_dim,
#            kernel_size,
#            stride=1,
#            pad=pad_size,
#            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),
#            bias_init=('ConstantFill', {'value': 0.})
#        )
#        current = model.Relu(current, current)
#        dim_in = hidden_dim    
    hour_branch_up_1 = model.Conv(
        current,
        'hour_branch_up_1',
        hidden_dim ,
        hidden_dim//2,
        kernel=1,
        stride=1,
        pad=0
        )
    hour_branch_up_1 = model.Relu(hour_branch_up_1,hour_branch_up_1)
    
    hour_branch_up_2 = model.Conv(
            hour_branch_up_1,
            'hour_branch_up_2',
            hidden_dim//2,
            hidden_dim//2,
            kernel=3,
            stride=1,
            pad=1,
            #weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),
            #bias_init=('ConstantFill', {'value': 0.})
        )
    hour_branch_up_2 = model.Relu(hour_branch_up_2,hour_branch_up_2)
    hour_branch_up_3 = model.Conv(hour_branch_up_2,'hour_branch_up_3',hidden_dim//2,hidden_dim//2,kernel=3,pad=1,stride=1)
    hour_branch_up_3 = model.Relu(hour_branch_up_3,hour_branch_up_3)
    hour_branch_up_4 = model.Conv(hour_branch_up_3,'hour_branch_up_4',hidden_dim//2,hidden_dim,1)
    hour_branch_down_1 = model.MaxPool(current, 'hour_branch_down_1', kernel=2 , stride=2)
    hour_branch_down_2 = model.Conv(hour_branch_down_1,'hour_branch_down_2', hidden_dim , (hidden_dim//2) ,kernel=1 ,stride=1,pad=0)
    hour_branch_down_2 = model.Relu(hour_branch_down_2,hour_branch_down_2)
    hour_branch_down_3 = model.Conv(hour_branch_down_2,'hour_branch_down_3',hidden_dim//2,hidden_dim//2,3,stride=1,pad=1,weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),bias_init=('ConstantFill', {'value': 0.}))
    hour_branch_down_3 =model.Relu(hour_branch_down_3,hour_branch_down_3)
    hour_branch_down_4 = model.Conv(hour_branch_down_3,'hour_branch_down_4',hidden_dim//2,hidden_dim//2,3,stride=1,pad=1,weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),bias_init=('ConstantFill', {'value': 0.}))
    hour_branch_down_4 = model.Relu(hour_branch_down_4,hour_branch_down_4)
    hour_branch_down_5 = model.Conv(hour_branch_down_4,'hour_branch_down_5',hidden_dim//2,hidden_dim,1,weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),bias_init=('ConstantFill', {'value': 0.}))
    hour_branch_down_6 = model.ConvTranspose(hour_branch_down_5,'hour_branch_down_6',hidden_dim,hidden_dim,kernel=4,stride=2,pad=1,weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),bias_init=('ConstantFill', {'value': 0.}))    
    current = model.Sum(['hour_branch_down_6','hour_branch_up_4'],"current",broadcast=1)
    current = model.Conv(current,'current_res_hour',hidden_dim,hidden_dim,kernel=3,pad=1,stride=1)
    
    return current, hidden_dim


#hourglass+res8*3
def add_roi_body_uv_head_v1convX_hourglass_res_3(model, blob_in, dim_in, spatial_scale):
    """v1convX design: X * (conv)."""
    hidden_dim = cfg.BODY_UV_RCNN.CONV_HEAD_DIM
    kernel_size = cfg.BODY_UV_RCNN.CONV_HEAD_KERNEL
    pad_size = kernel_size // 2
    current = model.RoIFeatureTransform(
        blob_in,
        '_[body_uv]_roi_feat',
        blob_rois='body_uv_rois',
        method=cfg.BODY_UV_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.BODY_UV_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.BODY_UV_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    for i in range(cfg.BODY_UV_RCNN.NUM_STACKED_CONVS):
        current = model.Conv(
            current,
            'body_conv_fcn_1_' + str(i + 1),# modified by zhiang.hao
            dim_in,
            hidden_dim,
            kernel_size,
            stride=1,
            pad=pad_size,
            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        
        current = model.Relu(current, current)
        if i==1:
          current_1 = model.Conv(
              current,
              'body_conv_fcn_1_branch' + str(1 + 1),
              dim_in,
              hidden_dim,
              kernel_size,
              stride=1,
              pad=pad_size,
              weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),
              bias_init=('ConstantFill', {'value': 0.})
          )
          current_1 = model.Relu(current_1, current_1)
        dim_in = hidden_dim
        
    current=model.Sum([current, current_1], "sum_branch_1" ,broadcast=1)    
    for i in range(cfg.BODY_UV_RCNN.NUM_STACKED_CONVS):
        current = model.Conv(
            current,
            'body_conv_fcn_2_' + str(i + 1),
            hidden_dim,
            hidden_dim,
            kernel_size,
            stride=1,
            pad=pad_size,
            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        
        current = model.Relu(current, current)
        if i==1:
          current_2 = model.Conv(
              current,
              'body_conv_fcn_2_branch' + str(1 + 1),
              hidden_dim,
              hidden_dim,
              kernel_size,
              stride=1,
              pad=pad_size,
              weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),
              bias_init=('ConstantFill', {'value': 0.})
          )
          current_2 = model.Relu(current_2, current_2)
    current=model.Sum([current, current_2], "sum_branch_2" ,broadcast=1)

    for i in range(cfg.BODY_UV_RCNN.NUM_STACKED_CONVS):
        current = model.Conv(
            current,
            'body_conv_fcn_3_' + str(i + 1),
            hidden_dim,
            hidden_dim,
            kernel_size,
            stride=1,
            pad=pad_size,
            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        
        current = model.Relu(current, current)
        if i==1:
          current_3 = model.Conv(
              current,
              'body_conv_fcn_3_branch' + str(1 + 1),
              hidden_dim,
              hidden_dim,
              kernel_size,
              stride=1,
              pad=pad_size,
              weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),
              bias_init=('ConstantFill', {'value': 0.})
          )
          current_3 = model.Relu(current_3, current_3)
    current=model.Sum([current, current_3], "sum_branch_3" ,broadcast=1)    
    hour_branch_up_1 = model.Conv(
        current,
        'hour_branch_up_1',
        hidden_dim ,
        hidden_dim//2,
        kernel=1,
        stride=1,
        pad=0
        )
    hour_branch_up_1 = model.Relu(hour_branch_up_1,hour_branch_up_1)
    
    hour_branch_up_2 = model.Conv(
            hour_branch_up_1,
            'hour_branch_up_2',
            hidden_dim//2,
            hidden_dim//2,
            kernel=3,
            stride=1,
            pad=1,
            #weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),
            #bias_init=('ConstantFill', {'value': 0.})
        )
    hour_branch_up_2 = model.Relu(hour_branch_up_2,hour_branch_up_2)
    hour_branch_up_3 = model.Conv(hour_branch_up_2,'hour_branch_up_3',hidden_dim//2,hidden_dim//2,kernel=3,pad=1,stride=1)
    hour_branch_up_3 = model.Relu(hour_branch_up_3,hour_branch_up_3)
    hour_branch_up_4 = model.Conv(hour_branch_up_3,'hour_branch_up_4',hidden_dim//2,hidden_dim,1)
    hour_branch_down_1 = model.MaxPool(current, 'hour_branch_down_1', kernel=2 , stride=2)
    hour_branch_down_2 = model.Conv(hour_branch_down_1,'hour_branch_down_2', hidden_dim , (hidden_dim//2) ,kernel=1 ,stride=1,pad=0)
    hour_branch_down_2 = model.Relu(hour_branch_down_2,hour_branch_down_2)
    hour_branch_down_3 = model.Conv(hour_branch_down_2,'hour_branch_down_3',hidden_dim//2,hidden_dim//2,3,stride=1,pad=1,weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),bias_init=('ConstantFill', {'value': 0.}))
    hour_branch_down_3 =model.Relu(hour_branch_down_3,hour_branch_down_3)
    hour_branch_down_4 = model.Conv(hour_branch_down_3,'hour_branch_down_4',hidden_dim//2,hidden_dim//2,3,stride=1,pad=1,weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),bias_init=('ConstantFill', {'value': 0.}))
    hour_branch_down_4 = model.Relu(hour_branch_down_4,hour_branch_down_4)
    hour_branch_down_5 = model.Conv(hour_branch_down_4,'hour_branch_down_5',hidden_dim//2,hidden_dim,1,weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),bias_init=('ConstantFill', {'value': 0.}))
    hour_branch_down_6 = model.ConvTranspose(hour_branch_down_5,'hour_branch_down_6',hidden_dim,hidden_dim,kernel=4,stride=2,pad=1,weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),bias_init=('ConstantFill', {'value': 0.}))    
    current = model.Sum(['hour_branch_down_6','hour_branch_up_4'],"current",broadcast=1)
    current = model.Conv(current,'current_res_hour',hidden_dim,hidden_dim,kernel=3,pad=1,stride=1)
    
    return current, hidden_dim



#hourglass_new

def add_roi_body_uv_head_v1convX_hourglass_new1(model, blob_in, dim_in, spatial_scale):
    """v1convX design: X * (conv)."""
    hidden_dim = cfg.BODY_UV_RCNN.CONV_HEAD_DIM
    kernel_size = cfg.BODY_UV_RCNN.CONV_HEAD_KERNEL
    pad_size = kernel_size // 2
    current = model.RoIFeatureTransform(
        blob_in,
        '_[body_uv]_roi_feat',
        blob_rois='body_uv_rois',
        method=cfg.BODY_UV_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.BODY_UV_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.BODY_UV_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    #14*14 feature map
    hour_14_up = model.Conv(current,'hour_14_up',dim_in,hidden_dim,kernel=4,stride=2,pad=1)
    hour_14_up = model.Relu(hour_14_up,hour_14_up)
    
    #7*7 feature map
    hour_7_up = model.Conv(hour_14_up,'hour_7_up',hidden_dim,hidden_dim,kernel=4,stride=2,pad=1)
    hour_7_up = model.Relu(hour_7_up,hour_7_up)
    
    #7*7 feature map
    hour_7_down = model.Conv(hour_7_up,'hour_7_down',hidden_dim,hidden_dim,kernel=3,stride=1,pad=1)
    hour_7_down = model.Relu(hour_7_down,hour_7_down)
    
    #7*7 sum
    hour_7_sum = model.Sum(['hour_7_up','hour_7_down'],'hour_7_sum',broadcast=1)
    hour_7_sum = model.Relu(hour_7_sum,hour_7_sum)
    
    #14*14 feature map
    hour_14_down = model.ConvTranspose(hour_7_sum,'hour_14_down',hidden_dim,hidden_dim,kernel=4,stride=2,pad=1,weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),bias_init=('ConstantFill', {'value': 0.}))
    hour_14_down = model.Relu(hour_14_down,hour_14_down)
    hour_14_down = model.Conv(hour_14_down,'hour_14_down_conv',hidden_dim,hidden_dim,kernel=3,stride=1,pad=1)
    hour_14_down = model.Relu(hour_14_down,hour_14_down)
    
    #14*14 sum
    hour_14_sum = model.Sum(['hour_14_up','hour_14_down_conv'],'hour_14_sum',broadcast=1)
    hour_14_sum = model.Relu(hour_14_sum,hour_14_sum)
    
    #28*28 feature map
    hour_28_down = model.ConvTranspose(hour_14_sum,'hour_28_down',hidden_dim,dim_in,kernel=4,stride=2,pad=1,weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),bias_init=('ConstantFill', {'value': 0.}))
    hour_28_down = model.Relu(hour_28_down,hour_28_down)
    hour_28_down = model.Conv(hour_28_down,'hour_28_down_conv',dim_in,dim_in,kernel=3,stride=1,pad=1)
    hour_28_down = model.Relu(hour_28_down,hour_28_down)
    
    #28*28 sum
    hour_28_sum = model.Sum([hour_28_down,current],'hour_28_sum',broadcast=1)
    hour_28_sum = model.Relu(hour_28_sum,hour_28_sum)
    
    
    #ite2 
    
    #14*14 feature map
    hour_14_up_2 = model.Conv(hour_28_sum,'hour_14_up_2',dim_in,hidden_dim,kernel=4,stride=2,pad=1)
    hour_14_up_2 = model.Relu(hour_14_up_2,hour_14_up_2)
    
    #7*7 feature map
    hour_7_up_2 = model.Conv(hour_14_up_2,'hour_7_up_2',hidden_dim,hidden_dim,kernel=4,stride=2,pad=1)
    hour_7_up_2 = model.Relu(hour_7_up_2,hour_7_up_2)
    
    #7*7 feature map
    hour_7_down_2 = model.Conv(hour_7_up_2,'hour_7_down_2',hidden_dim,hidden_dim,kernel=3,stride=1,pad=1)
    hour_7_down_2 = model.Relu(hour_7_down_2,hour_7_down_2)
    
    #7*7 sum
    hour_7_sum_2 = model.Sum(['hour_7_up_2','hour_7_down_2'],'hour_7_sum_2',broadcast=1)
    hour_7_sum_2 = model.Relu(hour_7_sum_2,hour_7_sum_2)
    
    #14*14 feature map
    hour_14_down_2 = model.ConvTranspose(hour_7_sum_2,'hour_14_down_2',hidden_dim,hidden_dim,kernel=4,stride=2,pad=1,weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),bias_init=('ConstantFill', {'value': 0.}))
    hour_14_down_2 = model.Relu(hour_14_down_2,hour_14_down_2)
    hour_14_down_2 = model.Conv(hour_14_down_2,'hour_14_down_conv_2',hidden_dim,hidden_dim,kernel=3,stride=1,pad=1)
    hour_14_down_2 = model.Relu(hour_14_down_2,hour_14_down_2)
    
    #14*14 sum
    hour_14_sum_2 = model.Sum(['hour_14_up_2','hour_14_down_conv_2'],'hour_14_sum_2',broadcast=1)
    hour_14_sum_2 = model.Relu(hour_14_sum_2,hour_14_sum_2)
    
    #28*28 feature map
    hour_28_down_2 = model.ConvTranspose(hour_14_sum_2,'hour_28_down_2',hidden_dim,dim_in,kernel=4,stride=2,pad=1,weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),bias_init=('ConstantFill', {'value': 0.}))
    hour_28_down_2 = model.Relu(hour_28_down_2,hour_28_down_2)
    hour_28_down_2 = model.Conv(hour_28_down_2,'hour_28_down_conv_2',dim_in,dim_in,kernel=3,stride=1,pad=1)
    hour_28_down_2 = model.Relu(hour_28_down_2,hour_28_down_2)
    
    #28*28 sum
    hour_28_sum_2 = model.Sum(['hour_28_down_conv_2','hour_28_sum'],'hour_28_sum_2',broadcast=1)
    hour_28_sum_2 = model.Relu(hour_28_sum_2,hour_28_sum_2)
    
    return hour_28_sum_2, hidden_dim
