# [HMS-Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification)
This repository contains a series of experiments that improved the classification performance of EEG-Spectrogram Data in the Kaggle competition [HMS - Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification).

## Description

* The Data consists of 50-second long EEG samples plus matched spectrograms covering a 10-minute window centered at the same time and labeled the central 10 seconds.
* Each of these samples belongs to one of six categories: Seizure, LPD, GPD, LRDA, GRDA, or Other is determined by expert votes.
* The vote count for each sample varies among several experts, ranging from 1 to 27.
* The Competition Criterion is KLDivergence Loss between the predicted probability and the observed target.

I primarily focused on utilizing **Spectrogram Image Data**, employing both [CNN](https://github.com/Cranjis-McB/HMS-Harmful-Brain-Activity-Classification/tree/main/CNN) and [Transformer](https://github.com/Cranjis-McB/HMS-Harmful-Brain-Activity-Classification/tree/main/Transformers) based approaches to enhance the Performance.
