####################################################
# 	This code is written by Reem Al Faifi      #
#	please cite my page when you use it        #
####################################################

import numpy as np
import Divide_dataset as div_data

import tensorflow as tf
import tensorflow_hub as hub

from keras.utils import np_utils

total_samples=160
batch_train=128
batch_val=32
num_frm=49
img_size=224
num_ch=3 
nb_classes=16

def get_model_name(k):
    return 'model_'+str(k)+'.hf5'

def pre_processing(x_train, x_val):
   x_train = x_train.astype('float32')
   x_val = x_val.astype('float32')
   mean_t=np.mean(x_train)
   std_t=np.std(x_train)
   x_train=x_train-mean_t
   x_train=x_train/std_t
   x_val=x_val-mean_t
   x_val=x_val/std_t
   return x_train, x_val
def model ()   
  for split_idx in range (0, 5):
    
     x_true, x_pred, y_true, y_pred = div_data.div_train_val (total_samples, split_idx, num_frm, img_size, num_ch)
     x_true, x_pred= pre_processing(x_true, x_pred)
    
     x_true= np.reshape(x_true, (batch_train , num_frm, img_size, img_size, num_ch))
     x_pred= np.reshape(x_pred, (batch_val , num_frm, img_size, img_size, num_ch))

     input_1 = tf.keras.layers.Input((num_frm, img_size, img_size, num_ch))
     model = hub.KerasLayer('https://tfhub.dev/deepmind/i3d-kinetics-400/1', trainable=False)
  
     x_1 = model(input_1)
  
     dropout=tf.keras.layers.Dropout(0.87)(x_1)
     output=tf.keras.layers.Dense(nb_classes)(dropout)
     model = tf.keras.Model(inputs=input_1, outputs=output)
     
     return model

prposed_model=model()
prposed_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
checkpoint = tf.keras.callbacks.ModelCheckpoint(get_model_name(split_idx), monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
prposed_model.summary()
history = model.fit(x_true,  y_true, epochs=500, validation_data=(x_pred, y_pred), batch_size=16, callbacks=[checkpoint])
