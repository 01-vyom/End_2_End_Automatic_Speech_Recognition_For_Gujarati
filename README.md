# End-to-End Automatic Speech Recognition For Gujarati
## ICON 2020: 17th International Conference on Natural Language Processing

### [[Paper]](https://drive.google.com/file/d/1u-X61pTSxoCEF-xC9IX7UVJiG482LeHR/view) | [[Long Oral Talk]](https://youtu.be/RO4BBpe61h8)

[Deepang Raval](https://www.linkedin.com/in/deepang-raval-8528b816b/)<sup>1</sup> | [Vyom Pathak](https://www.linkedin.com/in/01-vyom/)<sup>1</sup> | [Muktan Patel](https://www.linkedin.com/in/muktan-patel/)<sup>1</sup> | [Brijesh Bhatt](https://scholar.google.com/citations?user=aEkOFcUAAAAJ)<sup>1</sup>

[Dharmsinh Desai University, Nadiad](https://ddu.ac.in)<sup>1</sup>

We present a novel approach for improving the performance of an End-to-End speech recognition system for the Gujarati language. We follow a deep learning based approach which includes Convolutional Neural Network (CNN), Bi-directional Long Short Term Memory (BiLSTM) layers, Dense layers, and Connectionist Temporal Classification (CTC) as a loss function. In order to improve the performance of the system with the limited size of the dataset, we present a combined language model (WLM and CLM) based prefix decoding technique and Bidirectional Encoder Representations from Transformers (BERT) based post-processing technique. To gain key insights from our Automatic Speech Recognition (ASR) system, we proposed different analysis methods. These insights help to understand our ASR system based on a particular language (Gujarati) as well as can govern ASR systems' to improve the performance for low resource languages. We have trained the model on the Microsoft Speech Corpus, and we observe a 5.11% decrease in Word Error Rate (WER) with respect to base-model WER.

Complete proceedings can be found [here](https://www.iitp.ac.in/~ai-nlp-ml/icon2020/resources/ICON2020-Proceedings.pdf).

<!-- If you find this work useful in your research, please cite using the following BibTeX: BIB Here -->

## Setup

### System & Requirements

- Linux OS
- Python-3.6
- TensorFlow-2.2.0
- CUDA-11.1
- CUDNN-7.6.5

### Setting up repository

  ```shell
  git clone https://github.com/01-vyom/End_2_End_Automatic_Speech_Recognition_For_Gujarati.git
  python -m venv asr_env
  source $PWD/asr_env/bin/activate
  ```

### Installing Dependencies

Change directory to the root of the repository.

  ```shell
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

## Running Code

Change directory to the root of the repository.

### Training

To train the model in the paper, run this command:

```shell
python ./Train/train.py
```

Note:

- If required change the variables `PathDataAudios` and `PathDataTranscripts` to point to appropriate path to audio files and path to trascript file, in [Train/feature_extractor.py](https://github.com/01-vyom/End_2_End_Automatic_Speech_Recognition_For_Gujarati/blob/main/Train/feature_extractor.py) file. 
- If required change the variable `currmodel` in [Train/train.py](https://github.com/01-vyom/End_2_End_Automatic_Speech_Recognition_For_Gujarati/blob/main/Train/train.py) file to change the model name that is being saved.



### Evaluation

#### Inference

To inference using the model trained, run:

```shell
python ./Eval/inference.py
```

Note:

- Change the variables `PathDataAudios` and `PathDataTranscripts` to point to appropriate path to audio files and path to trascript file for testing.
- To change the name of the model for inferencing, change the variable `model`, and to change the name of file for testing, change `test_data` variable. 
- The output will be a `.pickle` of  references and hypothesis with a model specific name stored in the `./Eval/` folder.


#### Decoding

To decode the infered output, run:

```shell
python ./Eval/decode.py
```

Note:

- To select a model specific `.pickle` change the `model` variable.
- The output will be stored in `./Eval/`, specific to a model with all types of decoding and actual text.

#### Post-Processing

For post-processing the decoded output, follow the steps mentioned in this [README](https://github.com/01-vyom/End_2_End_Automatic_Speech_Recognition_For_Gujarati/blob/main/Spell%20Corrector%20BERT/README.md).
### System Analysis

To perform the system analysis, run:

```shell
python ./System Analysis/system_analysis.py
```

Note:

- To select a model specific decoding `.csv` file to analyze, change the `model` variable.

- To select a specific type of column (hypothesis type) to perform analysis, change the `type` variable. The output files will be saved in `./System Analysis/`, specific to a model and type of decoding.


## Results

Our algorithm achieves the following performance:

| Technique name                          | WER(%) reduction |
| --------------------------------------- | ---------------- |
| Prefix with LMs'                        | 2.42             |
| Prefix with LMs' + Spell Corrector BERT | 5.11             |

Note:

- These reductions in WER are w.r.t. the Greedy Decoding.

## Acknowledgement

The prefix decoding code is based on [1](https://github.com/corticph/prefix-beam-search) and [2](https://github.com/githubharald/CTCDecoder) open-source implementations. The code for Bert based spell corrector is adapted from this [open-source implementation](https://github.com/huseinzol05/NLP-Models-Tensorflow)

Licensed under the [MIT License](LICENSE.md).
