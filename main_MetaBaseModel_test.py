"""
@author : Hao
# Basemodel for metaSCI
# modified: Zhihong Zhang, 2021.6

Note:
- real 'meas' - [H,W,num_task]; simulated 'meas' (auto generated); - [H,W]

Todo:
- real 'meas' test
- batch_size > 1 to speed up (how to save the result?)

"""

import tensorflow as tf
# from tensorflow import InteractiveSession
# from tensorflow import ConfigProto
import numpy as np
from datetime import datetime
import os
import logging
from os.path import join as opj
import random
import scipy.io as sci
from tensorflow.python.framework.errors_impl import NotFoundError
from utils import generate_masks_MAML, generate_meas
from my_util.plot_util import plot_multi
from my_util.quality_util import cal_psnrssim
import time
from tqdm import tqdm
from MetaFunc import construct_weights_modulation, forward_modulation

#%% setting
## envir config
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 0.8
tf.reset_default_graph()  
 
# params config
# setting global parameters
batch_size = 1
num_frame = 10
image_dim = 256
sigmaInit = 0.01
update_lr = 1e-5
num_updates = 5
picked_task = [0] # pick masks for base model train
num_task = len(picked_task) # num of picked masks
run_mode = 'test'  # 'train', 'test','finetune'
test_real = False  # test real data
pretrain_model_idx = -1  # pretrained model index, 0 for no pretrained
exp_name = "Realmask_Test_256_Cr10_zzhTest"
timestamp = '{:%m-%d_%H-%M}'.format(datetime.now())  # date info

# data path
# datadir = "../[data]/dataset/training_truth/data_augment_256_8f_demo/"
# maskpath = "./dataset/mask/origDemo_mask_256_Cr8_4.mat"
datadir = "../[data]/dataset/testing_truth/bm_256_10f/"
maskpath = "./dataset/mask/realMask_256_Cr10_N576_overlap50.mat"
# datadir = "../[data]/dataset/training_truth/data_augment_512_10f/"
# maskpath = "./dataset/mask/demo_mask_512_Cr10_N4.mat"

# model path
# pretrain_model_path = './result/_pretrained_model/simulate_data_256_Cr8/'
pretrain_model_path = './result/simulate_data_256_Cr8_zzhTest/trained_model/'

# saving path
save_path = './result/test/'+exp_name+'_'+timestamp+'/'

# logging setting
logger = logging.getLogger()
logger.setLevel('INFO')
BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
chlr = logging.StreamHandler() # 输出到控制台的handler
chlr.setFormatter(formatter)
chlr.setLevel('INFO')  # 也可以不设置，不设置就默认用logger的level
fhlr = logging.FileHandler(save_path+'train.log') # 输出到文件的handler
fhlr.setFormatter(formatter)
logger.addHandler(chlr)
logger.addHandler(fhlr)

logger.info('Exp. name: '+exp_name)
logger.info('Mask path: '+maskpath)
logger.info('Data dir: '+datadir)
logger.inf('Params: batch_size {:d}, num_frame {:d}, image_dim {:d}, sigmaInit {:f}, picked_task {:s}, run_mode {:s}, pretrain_model_idx {:d}'.format(batch_size, num_frame, image_dim, sigmaInit, str(picked_task), run_mode, pretrain_model_idx))

#%% construct graph, load pretrained params ==> train, finetune, test
weights, weights_m = construct_weights_modulation(sigmaInit,num_frame)

mask = tf.placeholder('float32', [num_task, image_dim, image_dim, num_frame])
meas_re = tf.placeholder('float32', [num_task, batch_size, image_dim, image_dim, 1])
gt = tf.placeholder('float32', [num_task, batch_size, image_dim, image_dim, num_frame])

final_output = forward_modulation(mask, meas_re, gt, weights, weights_m, batch_size, num_frame, image_dim)

optimizer = tf.train.AdamOptimizer(learning_rate=0.00025).minimize(final_output['loss'])
#
nameList = os.listdir(datadir)
mask_sample, mask_s_sample = generate_masks_MAML(maskpath, picked_task)

if not os.path.exists(save_path):
    os.mkdir(save_path)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # load pretrained params
    if run_mode in ['test', 'finetune']:
        ckpt = tf.train.get_checkpoint_state(pretrain_model_path)
        if ckpt:
            ckpt_states = ckpt.all_model_checkpoint_paths
            saver.restore(sess, ckpt_states[pretrain_model_idx]) 
            logger.info('===> Load pretrained model from: '+pretrain_model_path)
        else:
            logger.error('===> No pretrained model found')
            raise NotFoundError('No pretrained model found')
                                       
    # [==> test]             
    if (run_mode in ['test']):    
        for index in tqdm(range(len(nameList))):
            time_all = 0
            psnr_all = np.zeros(num_frame)
            ssim_all = np.zeros(num_frame) 
            gt_sample = np.zeros([num_task, 1, image_dim, image_dim, num_frame])
            meas_sample = np.zeros([num_task, 1, image_dim, image_dim])

            for task_index in range(num_task):
                # load data
                mask_sample_i = mask_sample[task_index]
                data_tmp = sci.loadmat(datadir + nameList[index])
    
                if test_real:
                    gt_tmp = []
                    assert "meas" in data_tmp, 'NotFound ERROR: No MEAS in dataset'
                    meas_tmp = data_tmp['meas'][task_index]
                    # meas_tmp = data_tmp['meas']task_index / 255
                else:
                    try:
                        if "patch_save" in data_tmp:
                            gt_tmp = data_tmp['patch_save'] / 255
                        elif "orig" in data_tmp:
                            gt_tmp = data_tmp['orig'] / 255
                    except:
                        print('NotFound ERROR: No ORIG in dataset')           
                    meas_tmp = generate_meas(data_tmp, mask_sample_i)
                
                # normalize data
                mask_max = np.max(mask_sample_i) 
                mask_sample_i = mask_sample_i/mask_max
                meas_tmp = meas_tmp/mask_max
                    
                gt_sample[task_index, 1, ...] = gt_tmp
                meas_sample[task_index, 1,...] = meas_tmp
                
                meas_re_sample = meas_sample / np.expand_dims(mask_s_sample, axis=1)
                meas_re_sample = np.expand_dims(meas_re_sample, axis=-1)

            
                # test data
                begin = time.time()
                pred = sess.run([final_output['pred']],
                        feed_dict={mask: mask_sample,
                                    meas_re: meas_re_sample,
                                    gt: gt_sample}) # pred for Y_meas
                time_all += time.time() - begin
                
                pred = np.array(pred[0])
                pred = np.squeeze(pred)
                gt_sample = np.squeeze(gt_sample)
                
                # eval: psnr, ssim
                if len(gt_sample)>0:
                    for k in range(num_frame):      
                        psnr_all[k], ssim_all[k] = cal_psnrssim(gt_sample[...,num_frame], pred[...,num_frame])
                    
                    mean_psnr = np.mean(psnr_all)
                    mean_ssim = np.mean(ssim_all)
                    
                    logger.info('===> Task {} - Measurement {} complete: PSNR {}, SSIM {}, Time {}'.format(task_index, mean_psnr, mean_ssim, time_all))
                
                # save image and data
                plot_multi(pred[..., task_index], 'MeasRecon_Task%d_Meas%d'%(task_index, index), col_num=num_frame//2, savename='MeasRecon_Task%d_Meas%d'%(task_index, index), savedir=save_path+'recon_img/')
                
                sci.savemat('MeasRecon_Task%d_Meas%d.mat'%(task_index, index),
                            {'recon':pred, 
                            'gt':gt_sample,
                            'psnr_all':psnr_all,
                            'ssim_all':ssim_all,
                            'mean_psnr':mean_psnr,
                            'mean_ssim':mean_ssim,
                            'time_all':time_all,
                            'task_index':task_index            
                            })
                logger.info('===> data saved to: '+save_path)
