# Notes
As part of sprint 1, literature review is conducted to explore and understand what/how some of the existing ML solutions out there that could be adopted/referenced to current project.

Below are some notable solutions explored:


## Stanford Research Paper
[Sentiment Analysis for Amazon Reviews](https://cs229.stanford.edu/proj2018/report/122.pdf)
- Author explored models not only include traditional algorithms such as naive
bayes, linear supporting vector machines, K-nearest neighbor, but also deep learning metrics such as Recurrent Neural
Networks and convolutional neural networks
- Dataset comes from Consumer Reviews of Amazon
Products with 34660 data points in total
- In general, all models perform better with traditional input features than with glove input features
- Specifically,
LSTM generates the most accurate predictions over all other
models
- Summary of results

|          Models         | Training Acc. | Test Acc. |
|:-----------------------:|:-------------:|:---------:|
|      Multinomial NB     |     75.1%     |   70.6%   |
|        Linear SVM       |     83.4%     |   69.6%   |
|         RBF SVM         |     69.7%     |   69.2%   |
|          KNN-4          |     61.7%     |   61.7%   |
|          KNN-5          |     65.5%     |   65.4%   |
|          KNN-6          |     64.9%     |   64.6%   |
|           LSTM          |     73.5%     |   71.5%   |
|   Gaussian NB w/ Glove  |     52.2%     |   52.4%   |
|   Linear SVM w/ Glove   |     68.7%     |   68.6%   |
|      KNN-4 w/ Glove     |     58.1%     |   57.6%   |
|      KNN-5 w/ Glove     |     62.6%     |   62.2%   |
|      KNN-6 w/ Glove     |     61.3%     |   61.6%   |
|      LSTM w/ Glove      |     70.1%     |   70.2%   |
| LSTM w/ Glove(Resample) |     85.6%     |   65.6%   |

## Analytics Vidhya
[Amazon Product review Sentiment Analysis using BERT](https://www.analyticsvidhya.com/blog/2021/06/amazon-product-review-sentiment-analysis-using-bert/)

- model is trained on a pre-trained BERT model
- author's reason: BERT models have replaced the conventional RNN based LSTM networks which suffered from information loss in large sequential text
- BERT can easily understand the context of a word in a sentence based on previous words in the sentences due to its bi-directional approach
- Model performance:
  - with 2 epochs, author's models give 94.73% and 92% accuracy on the Training and validation set respectively

## Research Gate
[Sentimental Analysis of Amazon Product Reviews Using Machine Learning Approach](https://www.researchgate.net/publication/348159352_Sentimental_Analysis_of_Amazon_Product_Reviews_Using_Machine_Learning_Approach)
- Author explored naive bayes classifier (based on conditional probability) and support vector machine (discriminative algorithm)
- For preprocessing, author used Term Frequency- Inverse Document Frequency
  - tf-idf: a statistical measurement of word count in a document and how important word is in collection of documents
- Accuracy of naive bayes is 84.72%
- Accuracy of SVM is 86.59%
- Recall, precision were also recorded in the paper

## Others 
To be considered during iterative modelling phase:

- [A Complete Sentiment Analysis Algorithm in Python with Amazon Product Review Data: Step by Step](https://towardsdatascience.com/a-complete-sentiment-analysis-algorithm-in-python-with-amazon-product-review-data-step-by-step-2680d2e2c23b)
- [Sentiment classification on
Amazon reviews using machine
learning approaches](https://www.diva-portal.org/smash/get/diva2:1241547/FULLTEXT01.pdf)
- [Amazon Review - Machine Learning Project](https://t-lanigan.github.io/amazon-review-classifier/)