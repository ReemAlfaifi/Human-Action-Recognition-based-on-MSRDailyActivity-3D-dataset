####################################################
# 	This code is written by Reem Al Faifi      #
#	please cite my page when you use it        #
####################################################

import numpy as np
import Divide_dataset as div_data

import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

total_samples=160
batch_train=128
batch_val=32
num_frm=49
img_size=224
num_ch=3 
nb_classes=16

for split_idx in range (0, 1):

  xt, xv, yt, yv = div_data.div_train_val (total_samples, split_idx, num_frm, img_size, num_ch)

  xt=np.reshape(xt, (batch_train * num_frm, img_size, img_size, num_ch))
  xv=np.reshape(xv, (batch_val * num_frm, img_size, img_size, num_ch))
  
  input_1 = tf.keras.layers.Input(((batch_train * num_frm, img_size, num_ch))
  model = hub.KerasLayer('https://tfhub.dev/google/tiny_video_net/tvn1/1', trainable=False)
  
  x_1 = model(input_1)
  
  dropout=tf.keras.layers.Dropout(0.87)(x_1)
  output=tf.keras.layers.Dense(nb_classes)(dropout)
  model = tf.keras.Model(inputs=input_1, outputs=output)
  
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
  
  checkpoint = tf.keras.callbacks.ModelCheckpoint(get_model_name(j), monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
     
  model.summary()
  history = model.fit(X_train_RGB, ytrain, epochs=500, validation_data=(xv, yv))

