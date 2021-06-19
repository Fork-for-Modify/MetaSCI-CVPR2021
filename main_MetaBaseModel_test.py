"""
@author : Hao
# Base model test for metaSCI
# modified: Zhihong Zhang, 2021.6

Note:
- real 'meas' - [H,W,num_task]; simulated 'meas' (auto generated); - [H,W]

Todo:
- real 'meas' test (data format)
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
from os.path import exists as ope
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
exp_name = "M_Realmask_data_256_Cr10_zzhTest"
# exp_name = "real_data_256_Cr10"
timestamp = '{:%m-%d_%H-%M}'.format(datetime.now())  # date info

# data path
# datadir = "../[data]/dataset/training_truth/data_augment_256_8f_demo/"
# maskpath = "./dataset/mask/origDemo_mask_256_Cr8_4.mat"
# datadir = "../[data]/dataset/testing_truth/bm_256_10f/"
datadir = "../[data]/dataset/testing_truth/test_256_10f/"
maskpath = "./dataset/mask/realMask_256_Cr10_N576_overlap50.mat"
# datadir = "../[data]/dataset/training_truth/data_augment_512_10f/"
# maskpath = "./dataset/mask/demo_mask_512_Cr10_N4.mat"

# model path
# pretrain_model_path = './result/_pretrained_model/simulate_data_256_Cr8/'
pretrain_model_path = './result/train/M_RealmaskDemo_data_256_Cr10_zzhTest/trained_model/'
# pretrain_model_path = './result/train/real_data_512_Cr10/'

# saving path
save_path = './result/test/'+exp_name+'_'+timestamp+'/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# logging setting
logger = logging.getLogger()
logger.setLevel('INFO')
BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
chlr = logging.StreamHandler()# handler for console output
chlr.setFormatter(formatter)
chlr.setLevel('INFO')
fhlr = logging.FileHandler(save_path+'test.log')# handler for log file
fhlr.setFormatter(formatter)
logger.addHandler(chlr)
logger.addHandler(fhlr)

logger.info('Exp. name: '+exp_name)
logger.info('Mask path: '+maskpath)
logger.info('Data dir: '+datadir)
logger.info('Params: batch_size {:d}, num_frame {:d}, image_dim {:d}, sigmaInit {:f}, picked_task {:s}, run_mode-{:s}, pretrain_model_idx {:d}'.format(batch_size, num_frame, image_dim, sigmaInit, str(picked_task), run_mode, pretrain_model_idx))

#%% construct graph, load pretrained params ==> train, finetune, test
weights, weights_m = construct_weights_modulation(sigmaInit,num_frame)

mask = tf.placeholder('float32', [image_dim, image_dim, num_frame])
meas_re = tf.placeholder('float32', [batch_size, image_dim, image_dim, 1])
gt = tf.placeholder('float32', [batch_size, image_dim, image_dim, num_frame])

final_output = forward_modulation(mask, meas_re, gt, weights, weights_m, batch_size, num_frame, image_dim)

nameList = os.listdir(datadir)
mask_sample, mask_s_sample = generate_masks_MAML(maskpath, picked_task)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # load pretrained params
    if run_mode in ['test', 'finetune']:
        ckpt = tf.train.get_checkpoint_state(pretrain_model_path)
        if ckpt:
            ckpt_states = ckpt.all_model_checkpoint_paths
            saver.restore(sess, ckpt_states[pretrain_model_idx]) 
            logger.info('===> Load pretrained model from: '+ckpt_states[pretrain_model_idx])
        else:
            logger.error('===> No pretrained model found')
            raise FileNotFoundError('No pretrained model found')
                                       
    # [==> test]             
    if (run_mode in ['test']):  
        validset_psnr = 0
        validset_ssim = 0      
        for task_index in range(num_task):
            time_all = 0
            psnr_all = np.zeros(num_frame)
            ssim_all = np.zeros(num_frame) 
            mask_sample_i = mask_sample[task_index]
            
            for index in tqdm(range(len(nameList))):
                # load data
                data_tmp = sci.loadmat(datadir + nameList[index])
    
                if test_real:
                    gt_sample = np.zeros([image_dim, image_dim, num_frame])
                    assert "meas" in data_tmp, 'NotFound ERROR: No MEAS in dataset'
                    meas_sample = data_tmp['meas'][task_index]
                    # meas_tmp = data_tmp['meas']task_index / 255
                else:
                    if "patch_save" in data_tmp:
                        gt_sample = data_tmp['patch_save'] / 255
                    elif "orig" in data_tmp:
                        gt_sample = data_tmp['orig'] / 255
                    else:
                        raise FileNotFoundError('No ORIG in dataset')           
                    meas_sample = generate_meas(gt_sample, mask_sample_i)
                
                # normalize data
                mask_max = np.max(mask_sample_i) 
                mask_sample_i = mask_sample_i/mask_max
                meas_sample = meas_sample/mask_max
                meas_sample_re = meas_sample / mask_s_sample[task_index]
                
                gt_sample = np.expand_dims(gt_sample, 0)
                meas_sample_re = np.expand_dims(meas_sample_re, (0,-1))

            
                # test data
                begin = time.time()
                pred = sess.run([final_output['pred']],
                        feed_dict={mask: mask_sample_i,
                                    meas_re: meas_sample_re,
                                    gt: gt_sample}) # pred for Y_meas
                time_all += time.time() - begin
                
                pred = np.array(pred[0])
                pred = np.squeeze(pred)
                gt_sample = np.squeeze(gt_sample)
                
                # eval: psnr, ssim
                if len(gt_sample)>0:
                    for k in range(num_frame):      
                        psnr_all[k], ssim_all[k] = cal_psnrssim(gt_sample[...,k], pred[...,k])
                    
                    mean_psnr = np.mean(psnr_all)
                    mean_ssim = np.mean(ssim_all)
                    
                    validset_psnr += mean_psnr
                    validset_ssim += mean_ssim  
                                      
                    logger.info('---> Task {} - {:<20s} Recon complete: PSNR {:.2f}, SSIM {:.2f}, Time {:.2f}'.format(task_index, nameList[index], mean_psnr, mean_ssim, time_all))
                
                # save image and data
                plot_multi(pred, 'MeasRecon_Task%d_%s'%(task_index, nameList[index].split('.')[0]), col_num=num_frame//2, titles=psnr_all,savename='MeasRecon_Task%d_%s_psnr%.2f_ssim%.2f'%(task_index, nameList[index].split('.')[0],mean_psnr,mean_ssim), savedir=save_path+'recon_img/')
                
                if not ope(save_path+'recon_mat/'):
                    os.makedirs(save_path+'recon_mat/')
                sci.savemat(save_path+'recon_mat/MeasRecon_Task%d_%s_psnr%.2f_ssim%.2f.mat'%(task_index, nameList[index].split('.')[0],mean_psnr,mean_ssim),
                            {'recon':pred, 
                            'gt':gt_sample,
                            'psnr_all':psnr_all,
                            'ssim_all':ssim_all,
                            'mean_psnr':mean_psnr,
                            'mean_ssim':mean_ssim,
                            'time_all':time_all,
                            'task_index':task_index            
                            })
                logger.info('---> Recon data saved to: '+save_path)
        validset_psnr /= len(nameList)
        validset_ssim /= len(nameList)       
        logger.info('===> Task {} Recon complete: Aver. PSNR {:.2f}, Aver.SSIM {:.2f}'.format(task_index, validset_psnr, validset_ssim))