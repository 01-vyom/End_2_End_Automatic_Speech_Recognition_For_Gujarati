from collections import defaultdict, Counter
import re
import math
import numpy as np
import os
import psutil
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
import Model
import SpellCorrector
from TokenGenerator import maskedId
from GenerateId import generateId

BERT_VOCAB = 'PATH_TO/multi_cased_L-12_H-768_A-12/vocab.txt'
BERT_INIT_CHKPNT = 'PATH_TO/multi_cased_L-12_H-768_A-12/bert_model.ckpt'

tokenization.validate_case_matches_checkpoint(False,BERT_INIT_CHKPNT)
tokenizer = tokenization.FullTokenizer(vocab_file=BERT_VOCAB, do_lower_case=False)

def bertScore(string):
  """
  Function to generate the output list consisting top K replacements for each word in the sentence using BERT.
  """

  corrector = SpellCorrector()
  temp1 = []
  temp2 = []
  temp3 = []
  con = list(string.split(" "))
  tf.reset_default_graph()
  sess = tf.InteractiveSession()
  model = Model()
  sess.run(tf.global_variables_initializer())
  var_lists = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'bert')
  for word in con:
    possible_states = corrector.edit_candidates(word,fast=False)
    if len(possible_states) == 1:
      word = possible_states[0]

    if word in possible_states:
      temp1.append([word])
      continue

    text = string
    text_mask = text.replace(word, '**mask**')
    print(text_mask)

    cls = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'cls')

    replaced_masks = [text_mask.replace('**mask**', state) for state in possible_states]

    # print(replaced_masks)

    val = math.ceil(len(replaced_masks)/5)
    m = 0
    n = 5
    for i in range(0,val):
      
      rep_new = replaced_masks[m:n]

      tokens = tokenizer.tokenize(rep_new[0])

      input_ids = [maskedId(tokens, i) for i in range(len(tokens))]

      tokens_ids = tokenizer.convert_tokens_to_ids(tokens)

      ids = [generateId(mask) for mask in rep_new]

      tokens, input_ids, tokens_ids = list(zip(*ids))

      indices, ids = [], []
      for i in range(len(input_ids)):
        indices.extend([i] * len(input_ids[i]))
        ids.extend(input_ids[i])
      masked_padded = tf.keras.preprocessing.sequence.pad_sequences(ids,padding='post')
      preds = sess.run(tf.nn.log_softmax(model.logits), feed_dict = {model.X: masked_padded})
      preds = np.reshape(preds, [masked_padded.shape[0], masked_padded.shape[1], 119547])
      indices = np.array(indices)
      scores = []

      for i in range(len(tokens)-1):
        filter_preds = preds[indices == i]
        total = np.sum([filter_preds[k, k + 1, x] for k, x in enumerate(tokens_ids[i])])
        scores.append(total)
      prob_scores = np.array(scores) / np.sum(scores)

      probs = list(zip(possible_states, prob_scores))
      for i in probs:
        temp3.append(i)
      
      m+=5
      n+=5
    temp3.sort(key = lambda x: x[1])
    list(temp3)
    j = 0
    for i in temp3: 
      if (j != 3):
        temp2.append(i[0])
      if (j == 3):
        break
      j = j + 1
    if len(temp2)!=0:
      temp1.append(temp2)
    else:
      temp1.append([word])
    temp2 = []
    temp3 = []
  sess.close()
  return temp1