from collections import defaultdict, Counter
import re
import math
import numpy as np
import os
import zipfile
import pandas as pd
import tensorflow as tf
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from bert import modeling
import dill
import pickle
import itertools
from BertScoring import bertScore
from WlmOut import wlmOutput

if __name__ == "__main__":

  '''
  Download pre-trained BERT multilingual model and extract it.
  wget https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip
  unzip multi_cased_L-12_H-768_A-12.zip
  '''

  model_name = "temp_model" # Replace the model name with your model
  t_data = pd.read_csv("./Eval/" + model_name + "_ALL_DECODING.csv")
  t_data = t_data["Hypothesis_Prefix_LM_" + model_name]

  t_data_new = []
  for i in range(2950,3075):
    t_data_new.append(re.sub(r'[\s]+', ' ', t_data[i].strip()))

  sentences_after = []
  k=0
  for r in t_data_new:
    try:
      pro_list = []
      sent_matrix = bertScore(r)
      answer = [' '.join(perm) for perm in itertools.product(*sent_matrix)]
      print(answer)
      for i in answer:
        prob = wlmOutput(i)
        pro_list.append(prob)
      if(len(pro_list)!=0):
        sentences_after.append(answer[pro_list.index(max(pro_list))])
      else:
        sentences_after.append(r)
      print(sentences_after[k])
      answer = []
    except():
      continue
    k+=1

  df = pd.DataFrame(sentences_after,columns=['Prefix_Bert'])
  path = "Spell Corrector BERT/bert_final.csv"
  df.to_csv(path)