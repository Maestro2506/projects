import sys
import argparse
import cv2
import time
from config_reader import config_reader
from keras.models import load_model
import pandas as pd
import numpy as np
from collections import deque
import math
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras import layers

from tracker.utils.parser import get_config
from tracker.utils.draw import draw_boxes
from deep_sort import build_tracker
from processing import extract_parts, draw, dynamic_vector_formation, scale, new_fill_miss_parts
from feature_formation import Xs_formation, new_get_H, new_get_X, dynamic_Djoints
from model.cmu_model import get_testing_model

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim=306,num_heads=8,ff_dim=32, rate=0.1,**kwargs):
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        super(TransformerBlock, self).__init__()
    

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        base_config = super().get_config().copy()
        config={
            'att': self.att,
            'ffn': self.ffn,
            'layernorm1': self.layernorm1,
            'layernorm2': self.layernorm2,
            'dropout1': self.dropout1,
            'dropout2': self.dropout2,
        }
        return dict(list(base_config.items()) + list(config.items()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='input video file name')
    parser.add_argument('--classifier', type=str, required=True, help='which classifier you want to use (LSTM or Transformer)')
    parser.add_argument('--model', type=str, default='model/keras/model.h5', help='path to the weights file')
    parser.add_argument('--frame_ratio', type=int, default=1, help='analyze every [n] frames')
    parser.add_argument('--process_speed', type=int, default=4,
                            help='Int 1 (fastest, lowest quality) to 4 (slowest, highest quality)')
    parser.add_argument('--end', type=int, default=None, help='Last video frame to analyze')
    args = parser.parse_args()
    if args.classifier=='LSTM':
      path_to_classifier = 'RNN_Transformer_FCNN_models/LSTM_model'
      classifier = load_model(path_to_classifier)
    elif args.classifier=='Transformer':
      path_to_classifier = 'RNN_Transformer_FCNN_models/Transformer_model.h5'
      classifier = load_model(path_to_classifier,custom_objects={'TransformerBlock': TransformerBlock})
    keras_weights_file = args.model
    frame_rate_ratio = args.frame_ratio
    process_speed = args.process_speed
    ending_frame = args.end
    
    print('start processing...')
    column_names = []
    final_column_names = []
    for i in range(18):
      for j in range(i+1,18):
        column_names.append('Dist_'+str(i)+'_'+str(j))
    for i in range(18):
      for j in range(i+1,18):
        column_names.append('Angle_'+str(i)+'_'+str(j))
    column_names.append('PersonId')
    final_column_names.append('PersonId')
    column_names.append('FrameNum')
    final_column_names.append('FrameNum')
    column_names.append('Label')
    final_column_names.append('Label')
    df = pd.DataFrame(columns=column_names)
    final_df = pd.DataFrame(columns=final_column_names)
    
    video = args.video
    video_path = 'videos/Test/'
    video_file = video_path + video
    
    cfg = get_config()
    cfg.merge_from_file("tracker/configs/deep_sort.yaml")
    deepsort = build_tracker(cfg, use_cuda=True)
    
    model = get_testing_model()
    model.load_weights('model/keras/model.h5')
    params, model_params = config_reader()
    
    cam = cv2.VideoCapture(video_file)
    input_fps = cam.get(cv2.CAP_PROP_FPS)
    ret_val, orig_image = cam.read()
    video_length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Length of video:',video_length,'frames')
    if ending_frame is None:
        ending_frame = video_length
    
    output_path = 'videos/outputs_'+args.classifier+'/'
    output_format = '.mp4'
    video_output = output_path + video.split('.')[0] + output_format
    output_fps = input_fps #input_fps/frame_rate_ratio
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output, fourcc, output_fps, (640,480))
    
    scale_search = [1, .5, 1.5, 2]  # [.5, 1, 1.5, 2]
    scale_search = scale_search[0:4] #scale_search[0:process_speed]
    params['scale_search'] = scale_search
    
    i = 0
    prev_frame = {}
    saved_frames = {}
    classes =['Carry', 'ClapHands', 'PickUp', 'Push', 'Sit', 'Stand', 'Throw',
              'Walk', 'WaveHands']
    while (cam.isOpened()) and ret_val is True and i < ending_frame: 
      if i%frame_rate_ratio==0:
        print('Processing frame: ', i)
        resized = cv2.resize(orig_image, (640,480), interpolation = cv2.INTER_AREA)
        im = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
        body_parts, all_peaks, subset, candidate = extract_parts(im, params, model, model_params)
        tuples = []
        people = []
        cls_conf = []
        for m in range(len(all_peaks)):
          for c in all_peaks[m]:
            tuples.append(c)
        for p in subset:
          person = []
          max_conf = 0
          for idx in p[:18]:
            if int(idx)==-1:
              person.append(())
            else:
              person.append(tuples[int(idx)])
              if max_conf<tuples[int(idx)][2]:
                max_conf = tuples[int(idx)][2]
          people.append(person)
          cls_conf.append(max_conf)
        bbox_xywh = []
        for per in people:
          x_max,y_max,x_min,y_min = 0, 0, 640, 480
          for joi in per:
            if len(joi)==0:
              continue
            else:
              if x_max<joi[0]:
                x_max = joi[0]
              if x_min>joi[0]:
                x_min = joi[0]
              if y_max<joi[1]:
                y_max = joi[1]
              if y_min>joi[1]:
                y_min = joi[1]
          bbox_xywh.append([(x_max-x_min)/2+x_min,(y_max-y_min)/2+y_min,x_max-x_min,y_max-y_min])
        bbox_xywh = np.array(bbox_xywh)
        cls_conf = np.array(cls_conf)
        outputs = deepsort.update(bbox_xywh, cls_conf, im)
        saved_frames[i] = (resized,outputs)
        if len(outputs)==0:
          print('Number of people: 0')
          ret_val, orig_image = cam.read()
          i+= 1
          print('')
          continue
        elif (len(outputs)<len(people)) or (len(outputs)>len(people)) or (len(outputs[:,4])!=len(set(outputs[:,4]))):
          print('Discarded frame: ',i)
          ret_val, orig_image = cam.read()
          i+= 1
          print('')
          continue
        else:
          discard = False
          for r in range(len(people)):
            mask = outputs[:, 4] == r
            id_s = outputs[mask,5]
            if len(id_s)>1:
              continue
            id = int(id_s[0])
            final_vector = dynamic_vector_formation(people[r])
            final_vector = scale(640, 480,final_vector)
            if len(final_vector[1])==0 or (len(final_vector[8])==0 and len(final_vector[11])==0):
              continue
            if id in prev_frame:
              final_vector = new_fill_miss_parts(final_vector,prev_frame[id])
            for g in final_vector:
              if len(g)==0:
                g.append([None,None])
            if discard == False:
              diction = {}
              Xs = Xs_formation(final_vector)
              H = new_get_H(Xs)
              diction = new_get_X(Xs,H,diction)
              resultant = dynamic_Djoints(Xs,diction)
              for key, value in resultant.items():
                if value is None:
                  resultant[key] = 0
                elif math.isinf(value):
                  discard = True
              if discard == False:
                resultant['PersonId'] = id
                resultant['FrameNum'] = i
                resultant['Label'] = "unprocessed"
                df = df.append(resultant, ignore_index=True)
                prev_frame[id] = final_vector            
              else:
                continue
          for idx in df.PersonId.unique():
            if len(df[df["PersonId"]==idx])==8:
              input = np.array(df[df["PersonId"]==idx].iloc[:,:-3]).reshape(1,8,306)
              input = input.reshape(input.shape[0],-1)
              input = preprocessing.normalize(input)
              input = input.reshape(1,8,306)
              predict = classifier.predict(input)
              pose_idx = np.argmax(predict)
              label = classes[pose_idx]
              temp_df = df[df["PersonId"]==idx].iloc[:,-3:]
              temp_df = temp_df.replace('unprocessed', label)
              final_df = final_df.append(temp_df,ignore_index=True)
              df = df[df.PersonId != idx]
        ids = outputs[:,4]
        for key in list(prev_frame.keys()):
          if key in ids:
            continue
          else:
            prev_frame.pop(key)
        ret_val, orig_image = cam.read()
        i+=1
      else:  
        ret_val, orig_image = cam.read()
        i+=1
    action_labels = {}
    i = 0
    for frame_num in saved_frames: 
      if frame_num in final_df.FrameNum.unique():
        for idx in final_df[final_df['FrameNum']==frame_num].PersonId.unique():
          action_labels[idx] = final_df[(final_df['FrameNum']==frame_num)&(final_df['PersonId']==idx)]['Label'].values[0]
        bbox_tlwh = []
        bbox_xyxy = saved_frames[frame_num][1][:, :4]
        identities = saved_frames[frame_num][1][:, -1]
        ori_im = draw_boxes(saved_frames[frame_num][0], bbox_xyxy,action_labels,identities)
        out.write(ori_im)
      else:
        out.write(saved_frames[frame_num][0])
    out.release()
    cam.release()
    print('processing finished!')
