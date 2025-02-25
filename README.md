# A-Novel-Hybrid-Method-for-Detection-of-Online-Harassment-

MODULES:

• Data Selection

• Data Preprocessing

• Sentiment analysis

• Vectorization

• Data splitting

• Classification

• Performance metrics

4.2 MODULES DESCRIPTION:

4.2.1: DATA SELECTION:

· The input data was collected from dataset repository.

· In this project, the cyberbullying tweets dataset is used for detecting offensive and non-offensive tweets.

· The dataset which contains the information about the user name and tweets label.


4.2.2: DATA PREPROCESSING:

· Data pre-processing is the process of removing the unwanted data from the dataset.

· Pre-processing data transformation operations are used to transform the dataset into a structure suitable for machine learning.

· This step also includes cleaning the dataset by removing irrelevant or corrupted data that can affect the accuracy of the dataset, which makes it more efficient.

· Missing data removal

· Encoding Categorical data

· Missing data removal: In this process, the null values such as missing values and Nan values are replaced by 0.

· Missing and duplicate values were removed and data was cleaned of any abnormalities.

· Encoding Categorical data: That categorical data is defined as variables with a finite set of label values.

· That most machine learning algorithms require numerical input and output variables.

4.2.3 NLP TECHNIQUES:

• NLP is a field in machine learning with the ability of a computer to understand, analyze, manipulate, and potentially generate human language.

• Cleaning (or pre-processing) the data typically consists of a number of steps:

• Remove punctuation: Punctuation can provide grammatical context to a sentence which supports our understanding.

• Tokenization: Tokenizing separates text into units such as sentences or words. It gives structure to previously unstructured text. eg: Plata o Plomo-> ‘Plata’,’o’,’Plomo’.

• Stemming: Stemming helps reduce a word to its stem form.

• Sentiment analysis: In this step, we can analyse the sentiment into positive, neutral and negative by using the sentiment analyser (polarity score).

• Sentiment analysis works by breaking a message down into topic chunks and then assigning a sentiment score to each topic.


4.2.4: DATA SPLITTING:

· During the machine learning process, data are needed so that learning can take place.

· In addition to the data required for training, test data are needed to evaluate the performance of the algorithm in order to see how well it works.

· In our process, we considered 70% of the dataset to be the training data and the remaining 30% to be the testing data.

· Data splitting is the act of partitioning available data into two portions, usually for cross-validator purposes.

· One Portion of the data is used to develop a predictive model and the other to evaluate the model's performance.

· Separating data into training and testing sets is an important part of evaluating data mining model
Typically, when you separate a data set into a training set and testing set, most of the data is used for training, and a smaller portion of the data is used for testing.


4.2.4: CLASSIFICATION:

• In this step, we have to implement the two different machine learning algorithms such as Decision tree and naives bayes.With the help of machine learning algorithms, we have to analyse the cyberbullying cases.based on both give a short content to be kept on my resume project section 

4.2.5: RESULT GENERATION:

The Final Result will get generated based on the overall classification and prediction. The performance of this proposed approach is evaluated using some measures like,

· Accuracy

Accuracy of classifier refers to the ability of classifier. It predicts the class label correctly and the accuracy of the predictor refers to how well a given predictor can guess the value of predicted attribute for a new data.

AC= (TP+TN)/ (TP+TN+FP+FN)

· Precision

Precision is defined as the number of true positives divided by the number of true positives plus the number of false positives.

Precision=TP/ (TP+FP)


· Recall

Recall is the number of correct results divided by the number of results that should have been returned. In binary classification, recall is called sensitivity. It can be viewed as the probability that a relevant document is retrieved by the query.

Recall=TP/ (TP+FN)
