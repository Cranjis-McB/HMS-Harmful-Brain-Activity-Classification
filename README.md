# [HMS-Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification)
This repository contains a series of experiments that improved the classification performance of EEG-Spectrogram Data in the Kaggle competition [HMS - Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification).

## Description

* The Data consists of 50-second long EEG samples plus matched spectrograms covering a 10-minute window centered at the same time and labeled the central 10 seconds.
* Each of these samples belongs to one of six categories: Seizure, LPD, GPD, LRDA, GRDA, or Other is determined by expert voters.
* The vote count for each sample varies among several experts, ranging from 1 to 27.
* The Competition Criterion is KLDivergence Loss between the predicted probability and the observed target.

I primarily focused on utilizing **Spectrogram Image Data**, employing both [CNN](https://github.com/Cranjis-McB/HMS-Harmful-Brain-Activity-Classification/tree/main/CNN) and [Transformer](https://github.com/Cranjis-McB/HMS-Harmful-Brain-Activity-Classification/tree/main/Transformers) based approaches to enhance the Performance.

## Configuration
For most of the experiments, I have followed the same configuration as described below.

### For CNN
* **Model**: Efficientnet-b2
* **Epochs**: 6
* **Eval_per_epoch**: 2
* **Optimizer**: Adam
* **Learning Rate**: 1e-3
* **Scheduler**: One Cycle Policy with MaxLR: 1e-3
* **Loss**: KLDiv Loss


## References
* [HMS-Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification)
* [Audio Spectrogram Transformer](https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer)
* [EfficientNetB0 Starter - [LB 0.43]](https://www.kaggle.com/code/cdeotte/efficientnetb0-starter-lb-0-43?scriptVersionId=159911317)
* [How To Make Spectrogram from EEG](https://www.kaggle.com/code/cdeotte/how-to-make-spectrogram-from-eeg)
* [HMS-HBAC: ResNet34d Baseline [Training]](https://www.kaggle.com/code/ttahara/hms-hbac-resnet34d-baseline-training)
* [Same-Class-CutMix: The Only Data Aug that worked for me](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/479446)
* [Hard samples are more important - [LB 0.37]](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/477461)
