#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pdb
import random
import os
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Dense,Input
#from keras.layers.noise import GaussianDropout
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.layers import GaussianDropout
from collections import Counter
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,roc_curve, precision_recall_curve
#from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import *
#from Analyser.Analyser import *
from tensorflow.keras.models import load_model
#from keras.engine.topology import Layer,InputSpec
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import InputSpec
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
#from keras import backend as K
from tensorflow.keras import backend as K
import keras
import tensorflow as tf
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from validation import _compute_metrics as cm
from .Model import AbstractModel

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# For stronger determinism (TF ≥ 2.9)
tf.config.experimental.enable_op_determinism()
       
class DeepAMRManager:
    def __init__(self,n):
        super().__init__()
        self.X = None
        self.Y=None
        self.n_splits=None
        self.test_size=None
        self.x_train=None
        self.x_val=None
        self.x_test=None
        self.y_train=None
        self.y_val=None
        self.y_test=None
        self.batch_size=32
        self.autoencoder=None
        self.encoder=None
        self.ae_layer_dims=None
        self.deepamr_model=None
        self.class_threshold=None

    

    @property
    def name(self) -> str:
        return "DeepAMR"

    @property
    def model(self) -> any:
        return self


    def reset_data(self,X,Y):
        self.X=X
        self.Y=Y
    @property
    def param_grid(self) -> dict[str, any]:
        return {
                'C':0
                }

    @property
    def static_params(self) -> dict[str, any]:
        return None


        
    def makedivisible_to_all(self):

        self.x_train,self.y_train=self.makedivisible(self.x_train,self.y_train)
        self.x_test,self.y_test=self.makedivisible(self.x_test,self.y_test)
        
        self.x_val,self.y_val=self.makedivisible(self.x_val,self.y_val) 
        
    def makedivisible(self,x,y):
        b_s=self.batch_size
       
        if x.shape[0]%b_s!=0:
            to_remove=np.size(x,axis=0)-int(np.floor(np.size(x,axis=0)/b_s)*b_s)
            x=x[:-to_remove]
            y=y[:-to_remove]
        return x,y



    def data_prep(self,train_idx,test_idx,Batch_size=32):      
        #msss = MultilabelStratifiedShuffleSplit(n_splits=N_splits, 
        #                                        test_size=Test_size, random_state=rand_sta)
#       msss=MultilabelStratifiedKFold(n_splits=n_fold,random_state=rand_sta)

    
        train=self.X[train_idx]
        train_label=self.Y[train_idx]
        
        x_test=self.X[test_idx]
        y_test=self.Y[test_idx]

        val=self.X[test_idx]
        val_label=self.Y[test_idx]   
        
        self.x_train=train
        self.y_train=train_label
        self.x_val=val
        self.y_val=val_label
        self.x_test=x_test
        self.y_test=y_test
        self.batch_size=Batch_size
        
        self.makedivisible_to_all()
    
    
    
    def AutoEncoder(self,dims=[45345,500,1000,20],act='relu',init='uniform',drop_rate=0.3):
        """
        Fully connected auto-encoder model, symmetric.
        Arguments:
            dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
                The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
            act: activation, not applied to Input, Hidden and Output layers
        return:
            (ae_model, encoder_model), Model of autoencoder and model of encoder
        """
        n_stacks = len(dims) - 1
        # input
        input_img = Input(shape=(dims[0],), name='input')
        x = input_img
        x=GaussianDropout(drop_rate)(x)
        # internal layers in encoder
        for i in range(n_stacks-1):
            x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)
    
        # hidden layer
        encoded = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)  # hidden layer, features are extracted from here
    
        x = encoded
        # internal layers in decoder
        for i in range(n_stacks-1, 0, -1):
            x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)
    
        # output
        x = Dense(dims[0], kernel_initializer=init, activation='sigmoid', name='decoder_0')(x)
        decoded = x
        self.autoencoder=Model(inputs=input_img, outputs=decoded, name='AE')
        self.encoder=Model(inputs=input_img, outputs=encoded, name='encoder')
        self.ae_layer_dims=dims
        
    def pre_train(self,fold_idx,Epochs=100, Optimizer='SGD',Loss='binary_crossentropy',Callbacks=None):
        ae=self.autoencoder
        enc=self.encoder
        ae.compile(optimizer=Optimizer, loss=Loss)
#        
        ae.fit(self.x_train, self.x_train, 
                    batch_size=min(self.batch_size, len(self.x_train)), 
                    epochs=Epochs,
                    shuffle=False,
                    validation_data=[self.x_val,self.x_val],
                    callbacks=Callbacks)  # pretrin the 
        ae.save_weights('./DeepAMR_weights/ae_best_'+str(fold_idx+1)+'.weights.h5')          # save the best model
        ae.load_weights('./DeepAMR_weights/ae_best_'+str(fold_idx+1)+'.weights.h5')          # load the best model?
        self.autoencoder=ae
        self.encoder=enc
        
    def build_model(self,act='sigmoid',init='uniform'):
        t11=self.encoder.output
        t1 = Dense(1, kernel_initializer=init, activation=act,name = 'task1')(t11)
        # No.2
        #    st2 = Dense(cls_hidden_layer[1], init='normal')(cls2)
        t22=self.encoder.output
        t2 = Dense(1, kernel_initializer=init, activation=act,name='task2')(t22)
        # No.3
        #    st3 = Dense(cls_hidden_layer[1], init='normal')(cls2)
        t33=self.encoder.output
        t3 = Dense(1, kernel_initializer=init, activation=act,name='task3')(t33)
        # No.4
        #    st4 = Dense(cls_hidden_layer[1], init='normal')(cls2)
        t44=self.encoder.output
        t4 = Dense(1, kernel_initializer=init, activation=act,name='task4')(t44)
        # Compile model
        self.deepamr_model = Model(inputs = self.encoder.input, 
                                   outputs=[t1,self.autoencoder.output])
        #                           outputs=[t1,t2,t3,t4,self.autoencoder.output]) 

    #def train(self,Epochs=100,Loss=['binary_crossentropy', 'binary_crossentropy',
    #                     'binary_crossentropy','binary_crossentropy','binary_crossentropy'],Loss_weights=[1,1,1,1,1], Optimizer='Nadam',Callbacks=None):
    def train(self,Epochs=100,Loss=['binary_crossentropy', 'binary_crossentropy'],
                    Loss_weights=[1,1], Optimizer='Nadam',Callbacks=None,lr=None,save=False):
        if(lr):
            optimizer_instance=Nadam(learning_rate=float(lr))
        else:
            optimizer_instance=Optimizer
            
        self.deepamr_model.compile(loss=Loss,
                    loss_weights=Loss_weights,
                    optimizer=Optimizer, metrics = ['accuracy', 'mse']) 

    
        history=self.deepamr_model.fit(self.x_train, [self.y_train,self.x_train],
                             validation_data=(self.x_val, [self.y_val,self.x_val]),
                             shuffle=False,
                             epochs=Epochs,
                             batch_size=min(self.batch_size, len(self.x_train)),
                             callbacks=Callbacks)

        
        return history
    
    def _deepamr(self,train_idx, test_idx, fold_idx):

        #M=deepamr(X,Y)
        
        no_of_variants=self.X.shape[1]

        #-------------------------------------------------------
        print('preparing data in train, validation and test...')
        self.data_prep(train_idx,test_idx)
        #-----------------------------------------------------------------
        print('construct deep autoencoder...')
        self.AutoEncoder(dims=[no_of_variants,500,1000,20])
        #----------------------------------------------------------
        print('pre_training...')
        best_weights_filepath = f"./DeepAMR_weights/fold{fold_idx+1}_best.keras"
        saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
        esp=EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
        clr = CyclicLR(base_lr=0.001, max_lr=0.9,
                                    step_size=100., mode='triangular2')
        call_backs=[clr,esp,saveBestModel]
        self.pre_train(fold_idx,Epochs=5,Callbacks=call_backs)
        #-------------------------------------------------------------
        print('contruct deepamr model...')
        self.build_model()
        #print(M.deepamr_model.summary())
        #-----------------------------------------------------------
        print('train deeparm...')
        best_weights_filepath = f"./DeepAMR_weights/fold{fold_idx+1}_best.keras"
        saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
        esp=EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
        clr = CyclicLR(base_lr=0.0001, max_lr=0.003,
                                    step_size=100., mode='triangular2')
        call_backs=[clr,esp,saveBestModel]
        history=self.train(Epochs=20,Callbacks=call_backs)

        best_epoch = np.argmin(history.history['val_loss'])

        # Number of batches per epoch
        steps_per_epoch = len(self.x_train) // 32  # adjust to your batch size

        # Take LR at the last batch of the best epoch
        best_lr = clr.history['lr'][best_epoch * steps_per_epoch - 1]

        print(f"Best epoch: {best_epoch+1}, Best LR: {best_lr}")

        return best_lr
        #print('testing....')
        #fold_metrics.append(M.predict(drug_name))

    def tune_hyperparams(self, X_train_val, y_train_val, outer_cv):
        
            lr=[]
            for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_train_val, y_train_val)):
            
                self.reset_data(X_train_val,y_train_val)
                best_lr=self._deepamr(train_idx, test_idx, fold_idx)
                lr.append(best_lr)

            fold_best_lr=np.zeros((4,4))
            for ind,learning_rate in enumerate(lr):
                for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_train_val, y_train_val)):
                    X_val_inner, y_val_inner = X_train_val[test_idx], y_train_val[test_idx]
                    self.reset_data(X_train_val,y_train_val)
                    self.data_prep(train_idx,test_idx)
                    history=self.train(Epochs=20,Callbacks=None)
                    fold_metrics = cm(self, X_val_inner, y_val_inner,0.5)
                    fold_best_lr[ind,fold_idx]=fold_metrics["PR_AUC"]

            avg_auc_per_lr = np.mean(fold_best_lr, axis=1)
            best_index = np.argmax(avg_auc_per_lr)
            best_lr = lr[best_index]

            self.best_params = {"Learning rate": best_lr}
            print(f"{self.name} best learning rate: {best_lr}")
            return self.best_params

            #model.best_params = {"Learning rate" : best_lr}
           
    def fit(self,X_train_val,y_train_val):
        self.x_test=X_train_val
        self.y_test=y_train_val
        self.x_train=X_train_val
        self.y_train=y_train_val;
        self.train(Epochs=5,Callbacks=None,lr=self.best_params['Learning rate'])

        return self

    def load(self,data_key):

        best_model_path = "./DeepAMR_weights/best_"+data_key+".keras"
        self.deepamr_model  = load_model(best_model_path)
        self.deepamr_model.load_weights('./DeepAMR_weights/best_final_'+data_key+'_.weights.h5')
        
        return self


class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            #K.set_value(self.model.optimizer.lr, self.base_lr)
            self.model.optimizer.learning_rate.assign(self.base_lr)
        else:
            #K.set_value(self.model.optimizer.lr, self.clr())     
            self.model.optimizer.learning_rate.assign(self.clr())
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(self.model.optimizer.learning_rate.numpy())
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        #K.set_value(self.model.optimizer.lr, self.clr())
        self.model.optimizer.learning_rate.assign(self.clr())
   
