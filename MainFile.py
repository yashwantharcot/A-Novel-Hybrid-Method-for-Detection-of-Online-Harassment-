#========================= IMPORT PACKAGES ===========================
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import naive_bayes
from sklearn.metrics import classification_report,accuracy_score
import re
from sklearn.feature_extraction.text import TfidfTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier 
import numpy as np

#==================== DATA SELECTION =========================

from tkinter.filedialog import askopenfilename
filename = askopenfilename()
print("-----------------------------------------")
print("============ Data Selection =============")
print("-----------------------------------------")
data=pd.read_csv(filename)
print(data.head(10))
print()

#================== PREPROCESSING =============================

#=== checking missing values ===
print("-----------------------------------------")
print("========= Checking missing values  ======")
print("-----------------------------------------")
print(data.isnull().sum())
print()


#drop duplicates
data.drop_duplicates(inplace = True)


#drop unnecessary columns because its not required
print("----------------------------------------------")
print("========= Before drop unwanted columns  ======")
print("----------------------------------------------")
print()

print(data.shape)
print("----------------------------------------------")
print("========= After drop unwanted columns  ======")
print("----------------------------------------------")
print()

data_1=data.drop(['annotation'], axis = 1)
print(data_1.shape)


#========================= NLP TECHNIQUES ============================
#Corpus bag of words
corpus = []

#regular expressions and convert lower case
for i in range (0, len(data)):                               
    review = re.sub('[A-Z^a-z]',' ',data['content'][i])       
    review = review.lower()                                 
    review = review.split()                                 
    review = ' '.join(review)                              
    corpus.append(review)                                 


#====================== VECTORIZATION ==============================
#count vectorization

bow_transformer =  CountVectorizer()               
bow_transformer = bow_transformer.fit(corpus)      
print(len(bow_transformer.vocabulary_))            
messages_bow = bow_transformer.transform(corpus)  

tfidf_transformer = TfidfTransformer().fit(messages_bow)

#================== SENTIMENT ANALYSIS =================================
#positive, negative and neutral

analyzer = SentimentIntensityAnalyzer()
data_1['compound'] = [analyzer.polarity_scores(x)['compound'] for x in data_1['content']]
data_1['neg'] = [analyzer.polarity_scores(x)['neg'] for x in data_1['content']]
data_1['neu'] = [analyzer.polarity_scores(x)['neu'] for x in data_1['content']]
data_1['pos'] = [analyzer.polarity_scores(x)['pos'] for x in data_1['content']]

#Labelling
data_1['comp_score'] = data_1['compound'].apply(lambda c: 0 if c >=0 else 1)


#====================== DATA SPLITTING ================================
X_train, X_test, y_train, y_test = train_test_split(data_1['content'],data_1['comp_score'], random_state=100)

print("Total number of rows in dataset:", data.shape)
print()
print("Total number of rows in training data:", X_train.shape)
print()
print("Total number of rows in testing data:", X_test.shape)


#CountVectorizer method
vector = CountVectorizer(stop_words = 'english', lowercase = True)

#Fitting the training data 
training_data = vector.fit_transform(X_train)

#Transform testing data 
testing_data = vector.transform(X_test)

#================= CLASSIFICATION =================================
#Naive Bayes

print("================================")
print()
print("Naive Bayes")

#initialize the model
Naive = naive_bayes.MultinomialNB()

#fitting the model
Naive.fit(training_data, y_train)

#predict the model
nb_pred = Naive.predict(testing_data)


print()
print("Performances analysis for Naives bayes")
print()
Result_nb=accuracy_score(nb_pred,y_test)*100
print("Accuracy of naive bayes:",Result_nb,'%')
print()
print("Classification Report")
print(classification_report(nb_pred,y_test))


# #decision tree

print("================================")
print()
print("Decision tree")

#initialize the model
dt = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=100, min_samples_leaf=1)

#fitting the model
dt.fit(training_data, y_train)

#predict the model
dt_prediction=dt.predict(testing_data)

print()
print("Performances analysis for decision tree")
print()
Result_dt=accuracy_score(y_test, dt_prediction)*100
print("Accuracy of decision tree:",Result_dt,'%')
print()
print("Classification Report")
print(metrics.classification_report(y_test,dt_prediction))


#pie graph
plt.figure(figsize = (6,6))
counts = data_1['comp_score'].value_counts()
plt.pie(counts, labels = counts.index, startangle = 90, counterclock = False, wedgeprops = {'width' : 0.6},autopct='%1.1f%%', pctdistance = 0.55, textprops = {'color': 'black', 'fontsize' : 15}, shadow = True,colors = sns.color_palette("Paired")[3:])
plt.text(x = -0.35, y = 0, s = 'Total Tweets: {}'.format(data.shape[0]))
plt.title('Distribution of Tweets', fontsize = 14);
plt.show()
#====================== ALGORITHM COMPARISON ==================================

# #algorithm comparion 
if(Result_nb>Result_dt):
    print("==========================")
    print()
    print("Naives bayes algorithm is efficient")
    print()
    print("=========================")
else:
    print("==========================")
    print()
    print("Decision tree is efficient")
    print()
    print("=========================")


pred=int(input("Enter the prediction Index Number:"))

if nb_pred[pred] == 1:
    
    print('***********************************')
    print()
    print('-- Cyberbulling Cases --')
    print()
    print('***********************************')
    
else:
    
    print('***********************************')
    print()  
    print('-- Non Cyberbullying Cases  --')
    print()
    print('***********************************')
    
    
# #================== COMPARISON GRAPH =================================
objects = ('Naive Bayes', 'Decision tree')
y_pos = np.arange(len(objects))
performance = [Result_nb,Result_dt]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Algorithm comparison')
plt.show()