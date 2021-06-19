"""
@author : Hao
# Base model parallel training for metaSCI
# modified: Zhihong Zhang, 2021.6
TODO: : to be modified according to 'main_MetaBaseModel_train.py'
"""

import numpy as np
from datetime import datetime
import os
import logging
from os.path import join as opj
import random
import scipy.io as sci
from utils import generate_masks_MAML, generate_meas
from my_util.plot_util import plot_multi
from my_util.quality_util import cal_psnrssim
import time
from tqdm import tqdm
from MetaFunc import construct_weights_modulation, MAML_modulation,forward_modulation

# %% setting
# envir config
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
gpus = [0,1]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # hide tensorflow warning

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8
tf.reset_default_graph()


# params config
# setting global parameters
batch_size = 2
Total_batch_size = batch_size*2 # X_meas & Y_meas
num_frame = 10
image_dim = 256
Epoch = 100
sigmaInit = 0.01
step = 1
update_lr = 1e-5
num_updates = 5
picked_task = [200] # pick masks for base model train
num_task = len(picked_task) # num of picked masks
run_mode = 'finetune'  # 'train', 'test','finetune'
test_real = False  # test real data
pretrain_model_idx = -1  # pretrained model index, 0 for no pretrained
exp_name = "Realmask_Train_256_Cr10_zzhTest"
timestamp = '{:%m-%d_%H-%M}'.format(datetime.now())  # date info

# data path
# datadir = "../[data]/dataset/training_truth/data_augment_256_8f_demo/"
# maskpath = "./dataset/mask/origDemo_mask_256_Cr8_4.mat"
datadir = "../[data]/dataset/training_truth/data_augment_256_10f/"
maskpath = "./dataset/mask/realMask_256_Cr10_N576_overlap50.mat"
# datadir = "../[data]/dataset/training_truth/data_augment_512_10f/"
# maskpath = "./dataset/mask/demo_mask_512_Cr10_N4.mat"

# model path
# pretrain_model_path = './result/_pretrained_model/simulate_data_256_Cr8/'
pretrain_model_path = './result/train/M_Realmask_data_256_Cr10_zzhTest_06-17_21-10/trained_model/'

# saving path
save_path = './result/train/'+exp_name+'_'+timestamp+'/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# logging setting
logger = logging.getLogger()
logger.setLevel('INFO')
BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
chlr = logging.StreamHandler() # handler for console output
chlr.setFormatter(formatter)
chlr.setLevel('INFO')
fhlr = logging.FileHandler(save_path+'train.log') # handler for log file
fhlr.setFormatter(formatter)
logger.addHandler(chlr)
logger.addHandler(fhlr)

logger.info('\t Exp. name: '+exp_name)
logger.info('\t Mask path: '+maskpath)
logger.info('\t Data dir: '+datadir)
logger.info('\t Params: batch_size {:d}, num_frame {:d}, image_dim {:d}, sigmaInit {:f}, update_lr {:f}, num_updates {:d}, picked_task {:s}, run_mode- {:s}, pretrain_model_idx {:d}'.format(batch_size, num_frame, image_dim, sigmaInit, update_lr, num_updates, str(picked_task), run_mode, pretrain_model_idx))

# %% construct graph, load pretrained params ==> train, finetune, test
weights, weights_m = construct_weights_modulation(sigmaInit,num_frame)

mask = tf.placeholder('float32', [num_task, image_dim, image_dim, num_frame])
X_meas_re = tf.placeholder('float32', [num_task, batch_size, image_dim, image_dim, 1])
X_gt = tf.placeholder('float32', [num_task, batch_size, image_dim, image_dim, num_frame])
Y_meas_re = tf.placeholder('float32', [num_task, batch_size, image_dim, image_dim, 1])
Y_gt = tf.placeholder('float32', [num_task, batch_size, image_dim, image_dim, num_frame])

nameList = os.listdir(datadir)
mask_sample, mask_s_sample = generate_masks_MAML(maskpath, num_task)


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

tower_grads = []
tower_loss = []
reuse_vars = False

for i in range(len(gpus)):
    with tf.device('/gpu:%d' % i):
        with tf.variable_scope('forward', reuse=reuse_vars):
            xtask_output = forward_modulation(mask[i], X_meas_re[i], X_gt[i], weights, weights_m, batch_size, num_frame, image_dim)
            maml_grads = tf.gradients(xtask_output['loss'], list(weights_m.values()))
            gradients = dict(zip(weights_m.keys(), maml_grads))
            fast_weights = dict(zip(weights_m.keys(), [weights_m[key] - update_lr * gradients[key] for key in weights_m.keys()]))

            for j in range(num_updates - 1):
                xtask_output = forward_modulation(mask[i], X_meas_re[i], X_gt[i], weights, fast_weights, batch_size, num_frame, image_dim)
                maml_grads = tf.gradients(xtask_output['loss'], list(fast_weights.values()))
                gradients = dict(zip(fast_weights.keys(), maml_grads))
                fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - update_lr * gradients[key] for key in fast_weights.keys()]))

            ytask_output = forward_modulation(mask[i], Y_meas_re[i], Y_gt[i], weights, fast_weights, batch_size, num_frame, image_dim)

            loss_op = ytask_output['loss']

            optimizer = tf.train.AdamOptimizer(learning_rate=0.00025)
            grads = optimizer.compute_gradients(loss_op)

            reuse_vars = True
            tower_grads.append(grads)
            tower_loss.append(loss_op)

tower_grads = average_gradients(tower_grads)

update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update):
    train_op = optimizer.apply_gradients(tower_grads)
    tower_loss = tf.reduce_sum(tower_loss)

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

            _, Loss = sess.run([train_op, tower_loss],
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
            saver.save(sess, save_path + '/model_{}.ckpt'.format(epoch))

