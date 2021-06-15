"""
@author : Hao

"""

import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_eager_execution()

import numpy as np
import os
import random
import scipy.io as sci
from utils import generate_masks_metaTest
import time
from tqdm import tqdm
from MetaFunc import construct_weights_modulation, forward_modulation

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# data file
filename = "../data/train_multi_mask2/"

# saving path
path = './Result/task'

test_path = "../data/test/cacti"

# setting global parameters
batch_size = 4
num_frame = 8
image_dim = 256
Epoch = 100
sigmaInit = 0.01
step = 1


mask = tf.placeholder('float32', [image_dim, image_dim, num_frame])
meas_re = tf.placeholder('float32', [batch_size, image_dim, image_dim, 1])
gt = tf.placeholder('float32', [batch_size, image_dim, image_dim, num_frame])

weights, weights_m = construct_weights_modulation(sigmaInit,num_frame)

# feed forward
output = forward_modulation(mask, meas_re, gt, weights, weights_m, batch_size, num_frame, image_dim)

optimizer = tf.train.AdamOptimizer(learning_rate=0.00025)
#grads = optimizer.compute_gradients(output['loss'])
grads = optimizer.compute_gradients(output['loss'], list(weights_m.values()))
train_op = optimizer.apply_gradients(grads)

#
nameList = os.listdir(filename + 'gt/')
mask_sample, mask_s_sample = generate_masks_metaTest(filename)

def test(test_path):
    test_list = os.listdir(test_path)
    psnr_cnn = np.zeros([len(test_list), 1])

    for i in range(len(test_list)):
        pic = sci.loadmat(test_path + '/' + test_list[i])

        if "orig" in pic:
            pic = pic['orig']
        elif "patch_save" in pic:
            pic = pic['patch_save']
        pic = pic / 255

        pic_gt = np.zeros([pic.shape[2] // 8, 256, 256, 8])
        for jj in range(pic.shape[2]):
            if jj % 8 == 0:
                meas_t = np.zeros([256, 256])
                n = 0
            pic_t = pic[:, :, jj]
            mask_t = mask_sample[:, :, n]

            pic_gt[jj // 8, :, :, n] = pic_t
            n += 1
            meas_t = meas_t + np.multiply(mask_t, pic_t)

            if jj == 7:
                meas_t = np.expand_dims(meas_t, 0)
                meas_sample = meas_t
            elif (jj + 1) % 8 == 0 and jj != 7:
                meas_t = np.expand_dims(meas_t, 0)
                meas_sample = np.concatenate((meas_sample, meas_t), axis=0)

        meas_re_sample = meas_sample / np.expand_dims(mask_s_sample, axis=0)
        meas_re_sample = np.expand_dims(meas_re_sample, axis=-1)

        psnr_1 = 0
        for ii in range(meas_sample.shape[0]):
            out_pic1 = sess.run(output['pred'],
                                feed_dict={mask: mask_sample,
                                           meas_re: np.expand_dims(meas_re_sample[ii, :, :, :], axis=0).repeat(batch_size,axis=0)})

            for jj in range(8):
                out_pic_forward = out_pic1[0, :, :, jj]
                gt_t = pic_gt[ii, :, :, jj]

                mse = ((out_pic_forward * 255 - gt_t * 255) ** 2).mean()
                psnr_1 += 10 * np.log10(255 * 255 / mse)

        psnr_1 = psnr_1 / (meas_sample.shape[0] * 8)
        psnr_cnn[i] = psnr_1

    print("cnn result: {:.4f}".format(np.mean(psnr_cnn)))

if not os.path.exists(path):
    os.mkdir(path)

saver_tmp = tf.train.Saver()
variables = tf.all_variables()[0:78]
loader = tf.train.Saver(variables)


saver = tf.train.Saver()
epoch_loaded = 62
with tf.Session() as sess:
    if os.path.exists(path + '/checkpoint'):
        sess.run(tf.global_variables_initializer())
        loader.restore(sess, path + '/model_{}.ckpt'.format(epoch_loaded))
        print('*********Begin testing*********')
    else:
        print('*********The model does not exist! Initialize parameters !*********')
        sess.run(tf.global_variables_initializer())

    for epoch in range(Epoch):
        random.shuffle(nameList)
        epoch_loss = 0
        begin = time.time()

        try:
            with tqdm(range(int(len(nameList)/batch_size))) as t:
                for iter in t:
                    sample_name = nameList[iter*batch_size: (iter+1)*batch_size]
                    gt_sample = np.zeros([batch_size, image_dim, image_dim, num_frame])
                    meas_sample = np.zeros([batch_size, image_dim, image_dim])

                    for index in range(len(sample_name)):
                        gt_tmp = sci.loadmat(filename + 'gt/' + sample_name[index])
                        meas_tmp = sci.loadmat(filename + 'measurement4/' + sample_name[index])

                        if "patch_save" in gt_tmp:
                            gt_sample[index] = gt_tmp['patch_save'] / 255
                        elif "p1" in gt_tmp:
                            gt_sample[index] = gt_tmp['p1'] / 255
                        elif "p2" in gt_tmp:
                            gt_sample[index] = gt_tmp['p2'] / 255
                        elif "p3" in gt_tmp:
                            gt_sample[index] = gt_tmp['p3'] / 255

                        meas_sample[index] = meas_tmp['meas'] / 255

                    meas_re_sample = meas_sample / np.expand_dims(mask_s_sample, axis=0)
                    meas_re_sample = np.expand_dims(meas_re_sample, axis=-1)

                    _, Loss = sess.run([train_op, output['loss']],
                                       feed_dict={mask: mask_sample,
                                                  meas_re: meas_re_sample,
                                                  gt: gt_sample})

                    epoch_loss += Loss

        except KeyboardInterrupt:
            t.close()
            raise

        t.close()

        end = time.time()

        print("===> Epoch {} Complete: Avg. Loss: {:.7f}".format(epoch, epoch_loss / int(len(nameList)/batch_size)),
              "  time: {:.2f}".format(end - begin))

        #if (epoch+1) % step == 0:
            #saver.save(sess, path + '/model_{}.ckpt'.format(epoch))

        test(test_path)

