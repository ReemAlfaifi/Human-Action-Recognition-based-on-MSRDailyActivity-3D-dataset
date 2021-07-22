###################################################################
# This code is written by using the open source code of Tensorflow#
#Tanks Tensorflow Team                                            #
#Reem Alfaifi                                                     #
###################################################################
import numpy as np
import tempfile
import os
import ssl
from  cv2 import cv2
from urllib import request  # requires python3
import sys
from absl import logging
logging.set_verbosity(logging.ERROR)

train_classes_RGB=[]
val_classes_RGB=[]

# Utilities to fetch videos from UCF101 dataset
MSR_ROOT = "URL/PATH"
_VIDEO_LIST = None
_CACHE_DIR = tempfile.mkdtemp()
# As of July 2020, crcv.ucf.edu doesn't use a certificate accepted by the
# default Colab environment anymore.
unverified_context = ssl._create_unverified_context()

def fetch_ucf_video(video):
  """Fetchs a video and cache into local filesystem."""
  cache_path = os.path.join(_CACHE_DIR, video)
  if not os.path.exists(cache_path):
    urlpath = request.urljoin(MSR_ROOT, video)
    print("Fetching %s => %s" % (urlpath, cache_path))
    data = request.urlopen(urlpath, context=unverified_context).read()
    open(cache_path, "wb").write(data)
  return cache_path

def load_video(vid_path, max_frames=49, resize=(224, 224)):
        vidCap = cv2.VideoCapture(vid_path)
        framesPerSecond = vidCap.get(5)
        totalFrames = int (vidCap.get(cv2.CAP_PROP_FRAME_COUNT)) #total number of frames
        jumper=int(totalFrames/max_frames)
        if jumper==0:
           jumper=1
        frameIndex=1
        listFrames=[]
        for j in range (1, (max_frames+1)):
            #if frameIndex >= 0 & frameIndex <= totalFrames:
                    #set frame position
            vidCap.set(cv2.CAP_PROP_POS_FRAMES,frameIndex)
            frameIndex=(jumper+frameIndex)
            success, frame = vidCap.read()
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            listFrames.append(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        vidCap.release()
        cv2.destroyAllWindows()
        arraylistFrames = np.array(listFrames) #convert list of frames into array
        return arraylistFrames

filepath = 'train01.txt'
videos_train_list=[]
with open(filepath) as fp:
  line = fp.readline()
  while line: 
       path=format(line.strip())
       vid_name_folder, idx=path.split() 
       video_path = fetch_ucf_video(vid_name_folder)
       sample_video = load_video(video_path)
       videos_train_list.append(sample_video)
       train_classes_RGB.append(int(int (idx)-1))
       line = fp.readline()

videos_train_list = np.array(videos_train_list, dtype=np.float32)  
np.save('X_train.npy', videos_train_list)

train_classes_RGB = np.array(train_classes_RGB, dtype=np.int8)  
np.save('y_train.npy', train_classes_RGB)

