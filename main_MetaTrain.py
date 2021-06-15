"""
@author : Hao
# zzh: to be modified (like 'main_MetaTrain_parallel.py')
"""

import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_eager_execution()

import numpy as np
import os
import random
import scipy.io as sci
from utils import generate_masks_MAML, generate_meas
import time
from tqdm import tqdm
from MetaFunc import construct_weights_modulation, MAML_modulation

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpus=[1]

# data file
datadir = "../[data]/dataset/training_truth/data_augment_256_8f_demo/"
maskpath = "E:/project/HCA-SCI/algorithm/MetaSCI-CVPR2021/dataset/mask/origDemo_mask_256_Cr8_4.mat"
# datadir = "../[data]/dataset/training_truth/data_augment_512_10f/"
# maskpath = "E:/project/HCA-SCI/algorithm/MetaSCI-CVPR2021/dataset/mask/demo_mask_512_Cr10_N4.mat"

# saving path
path = './result/task'

# setting global parameters
batch_size = 1
Total_batch_size = batch_size*2
num_frame = 8
image_dim = 256
Epoch = 100
sigmaInit = 0.01
step = 1
update_lr = 1e-5
num_updates = 5
num_task = 1

weights, weights_m = construct_weights_modulation(sigmaInit,num_frame)

mask = tf.placeholder('float32', [num_task, image_dim, image_dim, num_frame])
X_meas_re = tf.placeholder('float32', [num_task, batch_size, image_dim, image_dim, 1])
X_gt = tf.placeholder('float32', [num_task, batch_size, image_dim, image_dim, num_frame])
Y_meas_re = tf.placeholder('float32', [num_task, batch_size, image_dim, image_dim, 1])
Y_gt = tf.placeholder('float32', [num_task, batch_size, image_dim, image_dim, num_frame])

final_output = MAML_modulation(mask, X_meas_re, X_gt, Y_meas_re, Y_gt, weights, weights_m, batch_size, num_frame, image_dim, update_lr, num_updates)

optimizer = tf.train.AdamOptimizer(learning_rate=0.00025).minimize(final_output['Loss'])
#
nameList = os.listdir(datadir)
mask_sample, mask_s_sample = generate_masks_MAML(maskpath, num_task)

if not os.path.exists(path):
    os.mkdir(path)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(Epoch):
        random.shuffle(nameList)
        epoch_loss = 0
        begin = time.time()

        for iter in tqdm(range(int(len(nameList)/Total_batch_size))):
            sample_name = nameList[iter*Total_batch_size: (iter+1)*Total_batch_size]
            X_gt_sample = np.zeros([num_task, batch_size, image_dim, image_dim, num_frame])
            X_meas_sample = np.zeros([num_task, batch_size, image_dim, image_dim])
            Y_gt_sample = np.zeros([num_task, batch_size, image_dim, image_dim, num_frame])
            Y_meas_sample = np.zeros([num_task, batch_size, image_dim, image_dim])

            for task_index in range(num_task):
                for index in range(len(sample_name)):
                    mask_sample_idx = mask_sample[task_index]
                    gt_tmp = sci.loadmat(datadir + sample_name[index])
                    if "patch_save" in gt_tmp:
                        gt_tmp = gt_tmp['patch_save'] / 255
                    elif "orig" in gt_tmp:
                        gt_tmp = gt_tmp['orig'] / 255      

                    meas_tmp = generate_meas(gt_tmp, mask_sample_idx) # zzh: calculate meas
                    

                    if index < batch_size:
                        X_gt_sample[task_index, index, :, :] = gt_tmp
                        X_meas_sample[task_index, index, :, :] = meas_tmp
                    else:
                        Y_gt_sample[task_index, index-batch_size, :, :] = gt_tmp
                        Y_meas_sample[task_index, index-batch_size, :, :] = meas_tmp

            X_meas_re_sample = X_meas_sample / np.expand_dims(mask_s_sample, axis=1)
            X_meas_re_sample = np.expand_dims(X_meas_re_sample, axis=-1)

            Y_meas_re_sample = Y_meas_sample / np.expand_dims(mask_s_sample, axis=1)
            Y_meas_re_sample = np.expand_dims(Y_meas_re_sample, axis=-1)

            _, Loss = sess.run([optimizer, final_output['Loss']],
                               feed_dict={mask: mask_sample,
                                          X_meas_re: X_meas_re_sample,
                                          X_gt: X_gt_sample,
                                          Y_meas_re: Y_meas_re_sample,
                                          Y_gt: Y_gt_sample})

            epoch_loss += Loss

        end = time.time()

        print("===> Epoch {} Complete: Avg. Loss: {:.7f}".format(epoch, epoch_loss / int(len(nameList)/batch_size)),
              "  time: {:.2f}".format(end - begin))

        if (epoch+1) % step == 0:
            saver.save(sess, path + '/model_{}.ckpt'.format(epoch))

