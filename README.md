# [HMS-Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification)
This repository contains a series of experiments that improved the classification performance of EEG-Spectrogram Data in the Kaggle competition [HMS - Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification).

## Description

* The Data consists of 50-second long EEG samples plus matched spectrograms covering a 10-minute window centered at the same time and labeled the central 10 seconds.
* Each of these samples belongs to one of six categories: Seizure, LPD, GPD, LRDA, GRDA, or Other is determined by expert voters.
* The vote count for each sample varies among several experts, ranging from 1 to 28.
* The Competition Criterion is KLDivergence Loss between the predicted probability and the observed target.

I primarily focused on utilizing **Spectrogram Image Data**, employing both [CNN](https://github.com/Cranjis-McB/HMS-Harmful-Brain-Activity-Classification/tree/main/CNN) and [Transformer](https://github.com/Cranjis-McB/HMS-Harmful-Brain-Activity-Classification/tree/main/Transformers) based approaches to enhance the Performance.

## Configuration
For most of the experiments, I have followed the same configuration as described below.

* **Model**: Efficientnets
* **Fold**: StratifiedGroupKFold (5 Folds)
* **Epochs**: 6
* **Eval_per_epoch**: 2
* **Optimizer**: AdamW
* **Learning Rate**: 1e-3 (For CNN)/ 1e-4 (For Transformers)
* **Scheduler**: One Cycle Policy with MaxLR: 1e-3 (For CNN)/ 1e-4 (For Transformers)
* **Loss**: KLDiv Loss

## [CNN](https://github.com/Cranjis-McB/HMS-Harmful-Brain-Activity-Classification/blob/main/CNN)

### 1. [Baseline Model](https://github.com/Cranjis-McB/HMS-Harmful-Brain-Activity-Classification/blob/main/CNN/baseline.ipynb)

Our baseline model processes a spectrogram image composed of four panels stacked vertically: LL, LP, RL, and RP.

| Input | OOF-CV | Public LB |
|-----------------|-----------------|-----------------|
| Spectrogram Images | 0.7287 | 0.45 |

### 2. [Global Spectrogram Features](https://github.com/Cranjis-McB/HMS-Harmful-Brain-Activity-Classification/blob/main/CNN/spectrogram_stat_image-nb.ipynb)


In this approach, rather than directly using the images, we extract the following statistics from four panel images and utilize them as input for our CNN:

```python
 X_min = np.min([LL, LP, RL, RP])
 X_max = np.max([LL, LP, RL, RP])
 X_mean = np.mean([LL, LP, RL, RP])
 X_var = X_max - X_min
```

These can be seen as **global spectrogram features**. These derived statistics are then utilized as input features for our Convolutional Neural Network (CNN).

 | Input | OOF-CV | Public LB |
|-----------------|-----------------|-----------------|
| Global Features | 0.7324 | 0.46 |

### 3. Ensemble of [1](https://github.com/Cranjis-McB/HMS-Harmful-Brain-Activity-Classification/blob/main/CNN/baseline.ipynb) + [2](https://github.com/Cranjis-McB/HMS-Harmful-Brain-Activity-Classification/blob/main/CNN/spectrogram_stat_image-nb.ipynb)

Ensemble can be performed in multiple ways; 1. **Model Ensemble**; where we take the weighted sum of the 2 models to get the final output. 2. **Input Feature Ensemble**; where we concat the input features from 1 and 2 and then train the model.

```python
 # 1. Model Ensemble
 model = 0.5 * model_1 + 0.5 * model_2

 # 2. Input Ensemble
 input = np.hstack([baseline_features, global_features])
```

 | Type | OOF-CV | Public LB |
|-----------------|-----------------|-----------------|
| Model Ensemble | NA | 0.42 |
| Input Feature Ensemble | 0.7027 | 0.43 |

### 4. EEG Spectrograms

Instead of using Kaggle-provided spectrograms, we generated Spectrograms from EEG Data as described in [this notebook](https://www.kaggle.com/code/cdeotte/how-to-make-spectrogram-from-eeg).

Note that for the baseline model, we concatenated percentile features along with the Input Features. that gave us a good **0.04 boost on CV and 0.01 boost on LB**.
```python
 # Percentiles
 X_20p = np.percentile(X, q=20, axis=0)
 X_40p = np.percentile(X, q=40, axis=0)
 X_60p = np.percentile(X, q=60, axis=0)
 X_80p = np.percentile(X, q=80, axis=0)
 X_median = np.vstack([X_20p, X_40p, X_60p, X_80p])
 
 input_img = np.hstack([input_img, X_median])
```

 | Inputs | OOF-CV | Public LB |
|-----------------|-----------------|-----------------|
| [Baseline + Percentiles](https://github.com/Cranjis-McB/HMS-Harmful-Brain-Activity-Classification/blob/main/CNN/eeg-percentiles-nb.ipynb) | 0.7104 | 0.45 |
| [Global Features](https://github.com/Cranjis-McB/HMS-Harmful-Brain-Activity-Classification/blob/main/CNN/eeg-global-features-nb.ipynb) | 0.7537 | 0.46 |
| Model Ensemble | NA | 0.42 |

### 5. Kaggle + EEG Ensemble

This is the ensemble of the models yielded in 3 and 4.

 | Type | OOF-CV | Public LB |
|-----------------|-----------------|-----------------|
| Kaggle Ensemble | NA | 0.42 |
| EEG Ensemble | NA | 0.42 |
| Kaggle + EEG Ensemble | NA | 0.38 |

### 6. Vote-Weighted KLDiv Loss

So far we were only using KL-Divergence Loss as a Cost function; ignoring the total expert votes for a given sample.

**The idea here is that samples with more votes are more reliable.** So we modify the cost function to take account of the total number of votes along with KLDiv-Loss. we modify the cost function to:


<p align="center">
  <span style="color:#333;">
    Loss = KLDiv * torch.log(total_votes + 1)
  </span>
</p>


This alone gave us **a total of 0.02 boost in CV and 0.02 boost in LB**. we further added percentile features as described in 4 and used [Same-Class Cutmix Augmentation](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/479446) to get **an additional 0.03 boost in CV and 0.01 boost** in LB over baseline described in 1.

| Input | OOF-CV | Public LB |
|-----------------|-----------------|-----------------|
| [Spectrogram + Percentiles](https://github.com/Cranjis-McB/HMS-Harmful-Brain-Activity-Classification/blob/main/CNN/vote-weighted-kldiv-loss-cutmix-nb.ipynb)| 0.6767 | 0.42 |
| [Global Features](https://github.com/Cranjis-McB/HMS-Harmful-Brain-Activity-Classification/blob/main/CNN/vote-weighted-kldiv-loss-global-cutmix-nb.ipynb) | 0.6971 | 0.42 |
| Ensemble | NA | 0.40 |

Table: Kaggle Spectrograms

### 7. Global Normalization

So far we have been normalizing the spectrogram images according to their mean and variance as shown below.

```python
# Normalization
m = np.nanmean(img.flatten())
s = np.nanstd(img.flatten())
img = (img - m) / (s + ep)
```
Instead of doing this, we derived the mean and standard deviation from the training data and used them for normalization. This gave us a **a good 0.04 boost in CV and 0.02 boost in LB.** (Thanks to Sandeep Anna for suggesting this idea.)

| Input | OOF-CV | Public LB |
|-----------------|-----------------|-----------------|
| [Spectrograms](https://github.com/Cranjis-McB/HMS-Harmful-Brain-Activity-Classification/blob/main/CNN/effb4_global_normalization_nb_(1).ipynb)| 0.6355 | 0.40 |
| [Global Features](https://github.com/Cranjis-McB/HMS-Harmful-Brain-Activity-Classification/blob/main/CNN/effb2-global-cutmix-nb.ipynb) | 0.6566 | 0.41 |
| Ensemble | NA | 0.38 |

Table: Kaggle Spectrograms

### 8. Mosaic Warmup + xloss

* **Mosaic Warmup**: We combined 4 spectrogram images into one image and labeled them as the average of their labels. we use these images and labels as warmup training for 3-epochs.
* **xloss:** we further change the loss function to

  <p align="center">
  <span style="color:#333;">
    Loss = KLDiv * torch.clamp(total_votes , 10)
  </span>
</p>

| Input | OOF-CV | Public LB |
|-----------------|-----------------|-----------------|
| [Spectrograms](https://github.com/Cranjis-McB/HMS-Harmful-Brain-Activity-Classification/blob/main/CNN/effb4_global_normalization_nb_(1).ipynb)| 0.6290 | 0.37 |
| [Global Features](https://github.com/Cranjis-McB/HMS-Harmful-Brain-Activity-Classification/blob/main/CNN/effb2-global-cutmix-nb.ipynb) | 0.6402 | 0.39 |
| Ensemble | NA | 0.36 |

Table: Kaggle Spectrograms


## References
* [HMS-Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification)
* [Audio Spectrogram Transformer](https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer)
* [EfficientNetB0 Starter - [LB 0.43]](https://www.kaggle.com/code/cdeotte/efficientnetb0-starter-lb-0-43?scriptVersionId=159911317)
* [How To Make Spectrogram from EEG](https://www.kaggle.com/code/cdeotte/how-to-make-spectrogram-from-eeg)
* [HMS-HBAC: ResNet34d Baseline [Training]](https://www.kaggle.com/code/ttahara/hms-hbac-resnet34d-baseline-training)
* [Same-Class-CutMix: The Only Data Aug that worked for me](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/479446)
* [Hard samples are more important - [LB 0.37]](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/477461)
