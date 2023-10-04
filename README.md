# MediaBiasinNews

This repository contains the source code for the paper titled "Reducing Media Bias in News Headlines", which aims to detect and mitigate media bias in news.

## Project Overview

Our project consists of several key components:

1. **Text Cleaning**: The first step in our pipeline involves cleaning and preprocessing the raw text data obtained from news articles. This ensures that the subsequent analysis is based on standardized and consistent input.

2. **Media Bias Detector Training**: We have developed a machine learning model to detect media bias in news articles. This model is trained on labeled data to distinguish between biased and non-biased content. The training process involves extracting informative features from the text.

3. **Media Bias Detector Inference**: Once the model is trained, it can be used for inference. Given a new news article, the bias detector assigns a bias score, helping us identify the level of bias in the content.

4. **PMI Analysis**: We utilize Pointwise Mutual Information (PMI) to construct lists of biased and non-biased words. PMI measures the association between words, and this analysis helps us identify terms that are indicative of bias.

5. **Word Replacement**: To mitigate bias in news articles, we have developed a replacement file that suggests alternative words or phrases for biased terms. This step allows for the creation of more balanced and objective content.

