import os
import time
import random
import numpy as np
import datetime

import tensorflow as tf
import tensorflow.keras.backend as K
tf.keras.backend.set_floatx('float32')
tf.keras.backend.clear_session()
from tensorflow.keras.callbacks import LearningRateScheduler, History

from module.architecture4hs import *
from module.dataset import HSDataset
from module.utils import *


class HSTrainer():
    def __init__(self, batch_size, encoder, decoder, q_predictor, t_predictor, lld_loss, mi_upper_loss, sim_loss, y_loss, alpha_lr, beta_lr, theta_lr, optimizer, model_save_path):
        self.batch_size = batch_size
        self.en = encoder
        self.de = decoder
        self.q = q_predictor
        self.t = t_predictor
        self.lld_loss = lld_loss
        self.mi_upper_loss = mi_upper_loss
        self.sim_loss = sim_loss
        self.y_loss = y_loss
        self.alpha_optimizer = optimizer(alpha_lr)
        self.beta_optimizer = optimizer(beta_lr)
        self.theta_optimizer = optimizer(theta_lr)
        self.model_save_path = model_save_path

        self.en_checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                                  psnr=tf.Variable(0),
                                                  optimizer=self.beta_optimizer,
                                                  model=self.en)
        self.en_checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.en_checkpoint,
                                                             directory=os.path.join(self.model_save_path, 'en_ckpt'),
                                                             max_to_keep=3)
        self.de_checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                                  psnr=tf.Variable(0),
                                                  optimizer=self.beta_optimizer,
                                                  model=self.de)
        self.de_checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.de_checkpoint,
                                                             directory=os.path.join(self.model_save_path, 'de_ckpt'),
                                                             max_to_keep=3)
        self.q_checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                                  psnr=tf.Variable(0),
                                                  optimizer=self.alpha_optimizer,
                                                  model=self.q)
        self.q_checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.q_checkpoint,
                                                             directory=os.path.join(self.model_save_path, 'q_ckpt'),
                                                             max_to_keep=3)
        self.t_checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                                  psnr=tf.Variable(0),
                                                  optimizer=self.alpha_optimizer,
                                                  model=self.t)
        self.t_checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.t_checkpoint,
                                                             directory=os.path.join(self.model_save_path, 't_ckpt'),
                                                             max_to_keep=3)

        self.restore()

        self.train_loss_mean = tf.keras.metrics.Mean(name='train_loss')
        self.train_acc = tf.keras.metrics.CategoricalAccuracy(name='train_acc')

        self.val_loss_mean = tf.keras.metrics.Mean(name='test_loss')
        self.val_acc = tf.keras.metrics.CategoricalAccuracy(name='test_acc')

    # Train the model
    def train(self, batch_size, train_size, epoches, dataset):
        train_iters = train_size // batch_size

        summary_writer = tf.summary.create_file_writer(os.path.join(self.model_save_path, 'log'))
        
        x_masks = []
        x_mask = np.ones((self.batch_size, 4000, 1))
        for i in range(5):
            mask_copy = x_mask.copy()
            mask_copy[:, 800*i: 800*(i+1), :] -= 1
            x_masks.append(mask_copy)
        
        z_masks = []
        z_mask = np.ones((self.batch_size, 250, 192), dtype=np.float32)
        for i in range(5):
            mask_copy = z_mask.copy()
            mask_copy[:, 50*i: 50*(i+1), :] -= 1
            z_masks.append(mask_copy)
        
        best_acc = 0.0
        wait = 0
        patience = 10
        time_begin = datetime.datetime.now()
        with summary_writer.as_default():
            for epoch in range(epoches):
                for step in range((epoch*train_iters), (epoch+1)*train_iters):
                    batch_x, batch_y = dataset.inputs(is_training=True)
                    
                    loss, logits = self.train_step(batch_x, batch_y, x_masks, z_masks)
                    
                    self.train_loss_mean(loss)
                    self.train_acc(labels, logits)

                    if step%10 == 0:
                        step_template = 'Epoch {}, Step {}/{}, Loss: {}, Acc: {}'
                        print (step_template.format(epoch+1,
                                                    step-(epoch*train_iters),
                                                    train_iters,
                                                    self.train_loss_mean.result(),
                                                    self.train_acc.result()*100))
                    self.train_loss_mean.reset_states()
                    self.train_acc.reset_states()
                
                # test data
                val_x, val_y = dataset.inputs(is_training=False)
                _ = self.val_step(val_x, val_y)
                val_template = 'Epoch {}, Test Loss: {}, Test Acc: {}'
                print (val_template.format(epoch+1,
                                           self.val_loss_mean.result(),
                                           self.val_acc.result()*100))
                
                val_acc = self.val_acc.result()

                wait += 1
                if val_acc > best_acc:
                    best_acc = val_acc

                    self.en_checkpoint_manager.save()
                    self.de_checkpoint_manager.save()
                    self.q_checkpoint_manager.save()
                    self.t_checkpoint_manager.save()

                    wait = 0
                if wait >= patience:
                    
                    break          

                self.val_loss_mean.reset_states()
                self.val_acc.reset_states()      

        time_end = datetime.datetime.now()
        print ('training done.')
        print("Time consuming: {}s".format((time_end-time_begin).total_seconds()))

    @tf.function
    def train_step(self, x, labels, x_masks, z_masks):
        w = -1

        for i in range(5):
            
            for j in range(5):

                mask_x_rhy = x * x_masks[j]
                mask_x_mor = x * x_masks[j] + w * x * (1 - x_masks[j])

                with tf.GradientTape() as alpha_tape:
                    z = self.en(x, training=False)
                    mask_z_rhy = self.en(mask_x_rhy, training=False)
                    mask_z_mor = self.en(mask_x_mor, training=False)
                    
                    mu_rhy, logvar_rhy = self.t(mask_z_rhy, training=True)
                    mu_mor, logvar_mor = self.q(mask_z_mor, training=True)
                    
                    lld_loss_rhy = self.lld_loss(z[:, 50*j: 50*(j+1), :], 
                                                 mu_rhy[:, 50*j: 50*(j+1), :], 
                                                 logvar_rhy[:, 50*j: 50*(j+1), :])
                    lld_loss_mor = self.lld_loss(z[:, 50*j: 50*(j+1), :], 
                                                 mu_mor[:, 50*j: 50*(j+1), :], 
                                                 logvar_mor[:, 50*j: 50*(j+1), :])
                    
                    alpha_loss = lld_loss_rhy + lld_loss_mor

                trainable_variables_alpha = self.t.trainable_variables + self.q.trainable_variables
                alpha_gradients = alpha_tape.gradient(alpha_loss, trainable_variables_alpha)
                self.alpha_optimizer.apply_gradients(zip(alpha_gradients, trainable_variables_alpha))
            
            mask_x_rhy = x * x_masks[i]
            mask_x_mor = x * x_masks[i] + w * x * (1 - x_masks[i])

            with tf.GradientTape() as beta_tape:

                z = self.en(x, training=True)
                mask_z_rhy = self.en(mask_x_rhy, training=True)
                mask_z_mor = self.en(mask_x_mor, training=True)
                
                logits = self.de(z, training=True)
                
                mu_rhy, logvar_rhy = self.t(mask_z_rhy, training=False)
                mu_mor, logvar_mor = self.q(mask_z_mor, training=False)

                mi_upper_loss_rhy = self.mi_upper_loss(z[:, 50*i: 50*(i+1), : ], 
                                                       mu_rhy[:, 50*i: 50*(i+1), :], 
                                                       logvar_rhy[:, 50*i: 50*(i+1), :])
                mi_upper_loss_mor = self.mi_upper_loss(z[:, 50*i: 50*(i+1), : ], 
                                                       mu_mor[:, 50*i: 50*(i+1), :], 
                                                       logvar_mor[:, 50*i: 50*(i+1), :])
                
                bce_loss = self.y_loss(logits, labels)
                sim_loss_rhy = self.sim_loss(mask_z_rhy * z_masks[i], z * z_masks[i])
                sim_loss_mor = self.sim_loss(mask_z_mor * z_masks[i], z * z_masks[i])

                beta_loss = bce_loss + .5 * mi_upper_loss_rhy + .5 * mi_upper_loss_mor + .5 * sim_loss_mor + .5 * sim_loss_rhy

            trainable_variables_beta =  self.en.trainable_variables + self.de.trainable_variables
            beta_gradients = beta_tape.gradient(beta_loss, trainable_variables_beta)
            self.beta_optimizer.apply_gradients(zip(beta_gradients, trainable_variables_beta))

        return beta_loss, logits
    
    @tf.function
    def val_step(self, x, labels):

        z = self.en(x, training=False)
        val_logits = self.de(z, training=False)

        val_loss = self.y_loss(val_logits, labels)

        self.val_loss_mean(val_loss)
        self.val_acc(labels, val_logits)

        return val_loss

    def restore(self):
        if self.en_checkpoint_manager.latest_checkpoint:
            self.en_checkpoint.restore(self.en_checkpoint_manager.latest_checkpoint)

        if self.de_checkpoint_manager.latest_checkpoint:
            self.de_checkpoint.restore(self.de_checkpoint_manager.latest_checkpoint)

        if self.q_checkpoint_manager.latest_checkpoint:
            self.q_checkpoint.restore(self.q_checkpoint_manager.latest_checkpoint)

        if self.t_checkpoint_manager.latest_checkpoint:
            self.t_checkpoint.restore(self.t_checkpoint_manager.latest_checkpoint)


if __name__ == '__main__':
    CV_PATH = './split_data_hs/'
    DATA_PATH = '../data/hs_segmentaion/hs_data/'
    LABEL_PATH = '../data/hs_segmentaion/hs_ref/'
    BATCH_SIZE = 100
    EPOCHES = 100
    LR1 = 0.001
    LR2 = 0.001
    LR3 = 0.001
    FOLD = 1
    TRAIN_SIZE = 1000

    MODEL_SAVE_PATH = os.path.join('./evaluation/model_hs/', 'mbcnn_cci_k{}'.format(str(FOLD)))

    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)

    dataset = HSPairDataset(batch_size=BATCH_SIZE, 
                          cv_path=CV_PATH, 
                          data_path=DATA_PATH, 
                          label_path=LABEL_PATH,
                          fold=FOLD)
   
    en = encoder()
    de = decoder()
    q_predictor = q_predictor()
    t_predictor = t_predictor()

    hs_trainer = HSTrainer(BATCH_SIZE, en, de, q_predictor, t_predictor, lld_loss_func, mi_upper_loss_func, sim_loss_func, y_loss_func4hs, LR1, LR2, LR3, opt_func, MODEL_SAVE_PATH)
    hs_trainer.train(BATCH_SIZE, TRAIN_SIZE, EPOCHES, dataset)

    ### Summary
    hs_trainer = HSTrainer(BATCH_SIZE, en, de, q_predictor, t_predictor, lld_loss_func, mi_upper_loss_func, sim_loss_func, y_loss_func4hs, LR1, LR2, LR3, opt_func, MODEL_SAVE_PATH)
    x_in = Input(shape=(4000, 1))

    z = hs_trainer.en(x_in, training=False)
    logits = hs_trainer.de(z, training=False)

    z_in = Input(shape=(250, 192))
    mu_t, logvar_t = hs_trainer.t(z_in, training=False)
    mu_q, logvar_q = hs_trainer.q(z_in, training=False)
    tq_pred = Model(z_in, [mu_t, mu_q, logvar_t, logvar_q], name='tq_predictor')
    tq_pred.summary()

    hs_segmentor = Model(x_in, logits, name='hs_segmentor')
    hs_segmentor.summary()
    config = hs_segmentor.to_json()
    with open(os.path.join(MODEL_SAVE_PATH, 'hs_segmentor.json'), 'w') as json_file:
        json_file.write(config)
    hs_segmentor.save_weights(os.path.join(MODEL_SAVE_PATH, 'hs_segmentor.h5'))
