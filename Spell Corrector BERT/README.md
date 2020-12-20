# Spell Corrector BERT

This folder contains the official implementation of Spell Corrector BERT mentioned in the paper. 

<img src="https://upload.wikimedia.org/wikipedia/commons/c/c4/Spell_Corrector_BERT.png" width=800>

-------------------------------------------------------

## Requirements

To install requirements:

```shell
pip install -r bert-requirements.txt
```
Download pre-trained BERT multilingual model and extract it:

```shell
wget https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip
unzip multi_cased_L-12_H-768_A-12.zip
```

Supported Environment:
- Linux OS
- Python-3.6
- TensorFlow-1.15.0
- CUDA-10.1
- CUDNN-8.0

## Execution

To execute the Spell Corrector BERT algorithm mentioned in the paper, run:

```shell
python ./Spell Corrector/init.py
```

Note:

- Do verify the comments in the [init.py](https://github.com/01-vyom/End_2_End_Automatic_Speech_Recognition_For_Gujarati/blob/main/Spell%20Corrector%20BERT/init.py) file before execution.

- The output of the algorithm would be stored as `bert_final.csv` file.

## Adaptation to other languages

To adapt this algorithm for other languages you need to make 4 changes:

1. In the file [CreateDict.py](https://github.com/01-vyom/End_2_End_Automatic_Speech_Recognition_For_Gujarati/blob/main/Spell%20Corrector%20BERT/CreateDict.py) replace the [GujWikiCorpusCount.csv](https://github.com/01-vyom/End_2_End_Automatic_Speech_Recognition_For_Gujarati/blob/main/Data%20Files/GujWikiCorpusCount.csv) file with your language specific word count file.

2. In the file [WlmDict.py](https://github.com/01-vyom/End_2_End_Automatic_Speech_Recognition_For_Gujarati/blob/main/Spell%20Corrector%20BERT/WlmDict.py) replace the [gujdata.txt](https://github.com/01-vyom/End_2_End_Automatic_Speech_Recognition_For_Gujarati/blob/main/Data%20Files/gujdata.txt) file with your corpus.

3. In the files [WlmDict.py](https://github.com/01-vyom/End_2_End_Automatic_Speech_Recognition_For_Gujarati/blob/main/Spell%20Corrector%20BERT/WlmDict.py) and [WlmOut.py](https://github.com/01-vyom/End_2_End_Automatic_Speech_Recognition_For_Gujarati/blob/main/Spell%20Corrector%20BERT/WlmOut.py) replace the variable `total` with the total count of words in your corpus.

4. In the file [EditDistance.py](https://github.com/01-vyom/End_2_End_Automatic_Speech_Recognition_For_Gujarati/blob/main/Spell%20Corrector%20BERT/EditDistance.py) replace the `letters` variable of edit_step function with the characters specific to your language.