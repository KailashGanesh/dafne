#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pickle
from generate_thigh_split_model import coscia_unet as unet
import dl.common.preprocess_train as pretrain

model=unet()
#model.load_weights('weights/weights_coscia_split.hdf5') ## old
model.load_weights('Weights_incremental_split/thigh/weights -  5 -  47699.60.hdf5') ## incremental

seg_list = pickle.load(open('testImages/test_segment.pickle', 'rb'))
image_list = pickle.load(open('testImages/test_data.pickle', 'rb'))

LABELS_DICT = {
        1: 'VL',
        2: 'VM',
        3: 'VI',
        4: 'RF',
        5: 'SAR',
        6: 'GRA',
        7: 'AM',
        8: 'SM',
        9: 'ST',
        10: 'BFL',
        11: 'BFS',
        12: 'AL'
    }
'''
LABELS_DICT = {
        1: 'SOL',
        2: 'GM',
        3: 'GL',
        4: 'TA',
        5: 'ELD',
        6: 'PE',
        }
'''
MODEL_RESOLUTION = np.array([1.037037, 1.037037])
MODEL_SIZE = (432, 432)
MODEL_SIZE_SPLIT = (250, 250)

image_list, mask_list = pretrain.common_input_process_split(LABELS_DICT, MODEL_RESOLUTION, MODEL_SIZE, MODEL_SIZE_SPLIT, {'image_list': image_list, 'resolution': MODEL_RESOLUTION}, seg_list)

ch = mask_list[0].shape[2]
aggregated_masks = []
mask_list_no_overlap = []
for masks in mask_list:
    agg, new_masks = pretrain.calc_aggregated_masks_and_remove_overlap(masks)
    aggregated_masks.append(agg)
    mask_list_no_overlap.append(new_masks)

for slice_number in range(len(image_list)):
    img = image_list[slice_number]
    segmentation = model.predict(np.expand_dims(np.stack([img,np.zeros(MODEL_SIZE_SPLIT)],axis=-1),axis=0))
    segmentationnum = np.argmax(np.squeeze(segmentation[0,:,:,:ch]), axis=2)
    cateseg=np.zeros((MODEL_SIZE_SPLIT[0],MODEL_SIZE_SPLIT[1],ch),dtype='float32')
    for i in range(MODEL_SIZE_SPLIT[0]):
        for j in range(MODEL_SIZE_SPLIT[1]):
            cateseg[i,j,int(segmentationnum[i,j])]=1.0
    acc=0
    y_pred=cateseg
    y_true=mask_list_no_overlap[slice_number]
    for j in range(ch):  ## Dice
        elements_per_class=y_true[:,:,j].sum()
        predicted_per_class=y_pred[:,:,j].sum()
        intersection=(np.multiply(y_pred[:,:,j],y_true[:,:,j])).sum()
        intersection=2.0*intersection
        union=elements_per_class+predicted_per_class
        acc+=intersection/(union+0.000001)
    acc=acc/ch
    print(str(slice_number)+'__'+str(acc))
