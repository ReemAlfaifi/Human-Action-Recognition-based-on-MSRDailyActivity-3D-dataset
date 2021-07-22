####################################################
# 	This code is written by Reem Al Faifi      #
#	please cite my page when you use it        #
####################################################

import numpy as np

def div_train_val (num_split, img_depth, img_size, num_channels):

   X=np.load(X.npy)
   Y=np.load(Y.npy)

   X_train=np.zeros((128,img_depth, img_size,img_size,num_channels))
   X_val=np.zeros((32, img_depth, img_size,img_size,num_channels))

   ytrain=np.zeros((128))
   yval=np.zeros((32))
        
   if (num_split==0):
       idx_st=0
       idx_sv=0
       endidx_t = idx_st+8
       endidx_v = idx_sv + 2
       
       startidx_train=0
       endidx_train=startidx_train+8            
       endidx_val=endidx_train+2
       

       X_train[idx_st:endidx_t][:][:][:][:]=np.copy(X[startidx_train:endidx_train, :, :, :, :])
       X_val[idx_sv:endidx_v][:][:][:][:]=np.copy(X[endidx_train:endidx_val, :, :, :,:])
                   
       ytrain[idx_st:endidx_t]=np.copy(Y[startidx_train:endidx_train])
       yval[idx_sv:endidx_v]=np.copy(Y[endidx_train:endidx_val])
                    
       startidx_train=endidx_val
       idx_st = endidx_t
       idx_sv = endidx_v
       del X,  Y  # deleting entire array
   elif (num_split==1):
       idx_st=0
       idx_sv=0
       endidx_t = idx_st+8
       endidx_v = idx_sv + 2
       
       startidx_train=0
       endidx_train=endidx_val+8            
       endidx_val=startidx_train+2
       
       X_train_RGB[idx_st:endidx_t][:][:][:][:]=np.copy(X[endidx_val:endidx_train, :, :, :, :])
       X_val_RGB[idx_sv:endidx_v][:][:][:][:]=np.copy(X[startidx_train:endidx_val, :, :, :,:])
                         
       ytrain[idx_st:endidx_t]=np.copy(Y[endidx_val:endidx_train])
       yval[idx_sv:endidx_v]=np.copy(Y[startidx_train:endidx_val])
                    
       startidx_train=endidx_train
       idx_st = endidx_t
       idx_sv = endidx_v
       del X,  Y  # deleting entire array
       
   elif (num_split==2):
       idx_st=0
       idx_sv=0
       
       endidxt1 = idx_st+6
       endidx_v = idx_sv+2
       endidxt2 = endidxt1+2
       
       startidx_train=0
       endidx_train1=startidx_train+6            
       endidx_val=endidx_traint1+2
       endidx_train2=endidx_val+2
       
       X_train[idx_st:endidxt1][:][:][:][:]=np.copy(X[startidx_train:endidx_train1, :, :, :, :])
       X_val[idx_sv:endidx_v][:][:][:][:]=np.copy(X[endidx_train1:endidx_val, :, :, :,:])
       X_train[endidxt1:endidxt2][:][:][:][:]=np.copy(X[endidx_val:endidx_train2, :, :, :, :])
                         
       ytrain[idx_st:endidxt1]=np.copy(Y[startidx_train:endidx_train1])
       yval[idx_sv:endidx_v]=np.copy(Y[endidx_train1:endidx_val])
       ytrain[endidxt1:endidxt2]=np.copy(Y[endidx_val:endidx_train2])
                    
       startidx_train=endidx_train2
       idx_st = endidxt2
       idx_sv = endidx_v
       del X,  Y  # deleting entire array
       
   elif (num_split==3):
       idx_st=0
       idx_sv=0
       
       endidxt1 = idx_st+4
       endidx_v = idx_sv+2
       endidxt2 = endidxt1+4
       
       startidx_train=0
       endidx_train1=startidx_train+4            
       endidx_val=endidx_traint1+2
       endidx_train2=endidx_val+4
       
       X_train[idx_st:endidxt1][:][:][:][:]=np.copy(X[startidx_train:endidx_train1, :, :, :, :])
       X_val[idx_sv:endidx_v][:][:][:][:]=np.copy(X[endidx_train1:endidx_val, :, :, :,:])
       X_train[endidxt1:endidxt2][:][:][:][:]=np.copy(X[endidx_val:endidx_train2, :, :, :, :])
                         
       ytrain[idx_st:endidxt1]=np.copy(Y[startidx_train:endidx_train1])
       yval[idx_sv:endidx_v]=np.copy(Y[endidx_train1:endidx_val])
       ytrain[endidxt1:endidxt2]=np.copy(Y[endidx_val:endidx_train2])
                    
       startidx_train=endidx_train2
       idx_st = endidxt2
       idx_sv = endidx_v
       del X,  Y  # deleting entire array
   elif (num_split==4):
       idx_st=0
       idx_sv=0
       
       endidxt1 = idx_st+6
       endidx_v = idx_sv+2
       endidxt2 = endidxt1+2
       
       startidx_train=0
       endidx_train1=startidx_train+6            
       endidx_val=endidx_traint1+2
       endidx_train2=endidx_val+2
       
       X_train[idx_st:endidxt1][:][:][:][:]=np.copy(X[startidx_train:endidx_train1, :, :, :, :])
       X_val[idx_sv:endidx_v][:][:][:][:]=np.copy(X[endidx_train1:endidx_val, :, :, :,:])
       X_train[endidxt1:endidxt2][:][:][:][:]=np.copy(X[endidx_val:endidx_train2, :, :, :, :])
                         
       ytrain[idx_st:endidxt1]=np.copy(Y[startidx_train:endidx_train1])
       yval[idx_sv:endidx_v]=np.copy(Y[endidx_train1:endidx_val])
       ytrain[endidxt1:endidxt2]=np.copy(Y[endidx_val:endidx_train2])
                    
       startidx_train=endidx_train2
       idx_st = endidxt2
       del X,  Y  # deleting entire array
   return X_train, X_val, ytrain, yval  

