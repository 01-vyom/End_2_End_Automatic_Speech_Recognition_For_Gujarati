# Importing libraries
import os
import zipfile
import pandas as pd
from string import ascii_lowercase
import IPython.display as ipd
import argparse
import logging
import pickle
import dill
import warnings

warnings.filterwarnings("ignore")
import sys
import tensorflow as tf
import numpy as np
import re
import librosa
import seaborn as sns

sns.set()
import IPython.display as ipd
from scipy.io import wavfile  # for audio processing
from string import ascii_lowercase
import matplotlib.pyplot as plt
from __future__ import division
from __future__ import print_function
import codecs
from glob import glob
import difflib
from difflib import SequenceMatcher
from collections import defaultdict, Counter
import librosa.display

dill._dill._reverse_typemap["ClassType"] = type
