
# coding: utf-8

'''
 My own interest in statistics comes from an interest in health studies so this dataset appealed to me. 
 This dataset contained a number of features describing patients with diabetetes and whether or not they 
 were readmitted, and if they were readmitted if they were readmitted within 30 days or longer. Once the data 
 was cleaned I used two different models, a RandomForest Classification model and a Naive Bayes model and compared their results.
 '''

#look through all these imports, remove the ones that aren't used. 
import requests
import pandas as pd
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import sklearn.metrics as skm
import pylab as pl
from sklearn import cross_validation 
from sklearn import preprocessing
from sklearn import svm
import zipfile
import StringIO
from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt
from sklearn.lda import LDA
from sklearn.naive_bayes import MultinomialNB


'''
The CSV was downloaded as a zip file and then read into a Pandas dataframe. I chose using Pandas Dataframes 
due to their ease of use, which I felt outweighted the value of working with a SQL database considering the size of the dataset.
'''

Download the ZIP file and then open the csv. 
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip"
r = requests.get(url)
z = zipfile.ZipFile(StringIO.StringIO(r.content))
load the dataframe as df, this is the raw data and not what will be used for the models
df = pd.read_csv(z.open('dataset_diabetes/diabetic_data.csv'))

'''
I then examined the data. I first made sure it that the file was imported into a dataframe correctly. 
I next examined the first rows of the data using the head function. This gave me an immediete sense of the 
features, along with what ones might be useful, and what features would have to be converted. A second file 
containing the meaning of some categorical feature values was in the zip file and was consulted. This is not 
technically neccesary for any of the models, but is useful from a real world perspective to know what 
exactly I am working with. I next used Pandas' built in Describe function to examine the data. I found 
that several of the features would not be useful, such as weight which contained only question marks. 
The majority of the features were categorical, and many had multiple non-ordered values, so I knew I would
 have to convert them into dummy variables. Other variables were integer values and I was able to see the m
 ean value as well as the variance and other measures of distribution.
'''

#examine the dataframe
print type(df)
print df.head()
print df.describe()

"""
Once I saw the data I knew that I would want to predict the result of the readmitted, so I examined the frequency of the different values contained in the feature. Slightly over half of the patients were not readmitted to the hospital, and the majority who were readmitted were readmitted after 30 days. Since readmittance at all was more interesting to me then how long until they were readmitted I decided to combine the values and simply categorize patients as readmitted or not readmitted. 
"""

#get the count of it
df['readmitted'].value_counts()

'''
Next I had to clean the data. I created a new, empty dataframe that I named X. 
I then examined each of the features in the original dataset one by one, looking at what 
type of data the feature contained values they had, and the frequency of those values. Features 
where one value dominated to a high degree were discarded as not being useful for the data. 
An example of this was the feature 'glipizide-metformin' where only 13 individuals had a 
value other than No for the feature. Other variables were incorporated into the new dataframe for analysis. 

Some variables, such as the number of medications that a patient was taking I was able to just 
use without any cleaning or processing. For other features, such as the patient's gender, I
 only had to change it from a string value into a dummy variable where the strings were replaced by either a one or a zero. 

For most of the variables more processing then this was neccesary. These were categorical 
variables with multiple non-ordinal levels in the form of strings. For these values I first checked 
the frequency of the different values. If one value dominated the feature then I did not use the feature 
as I did not think it would be useful as discussed above. If one value did not dominate the feature I then 
examined it to see how many values were frequent enough in the feature to be used. In many cases, only one or 
two values of the feature were common. In these cases I created dummy variables for only the values that were 
common, and all other values of the feature were grouped together as 0 on all dummy variables. An example of 
this was for the admission type feature. This feature could take on 25 different values for the reason for admission, 
however three values dominated the feature so I used only these three values. To save space I have commented out the 
different frequencies after I used them.
'''


#Datacleaning, all of the features here
#looking through the data and examining what variables I am going to include, and how to include them
#If I am including then add it to the X dataframe in clean form
X = pd.DataFrame() #X will be the df used in the analysis. 

#print df['race'].value_counts() #After I examine the count I comment out the code, will see below
#race is mostly caucasian, then AA, create dummy variables for White and Black
X['white'] = (df['race'] == 'Caucasian').astype(int)
X['black'] = (df['race'] == 'AfricanAmerican').astype(int)

#print df['gender'].value_counts()
#female is most common, setting that as dummy 
X['female'] = (df['gender'] == 'Female').astype(int)

#print df['age'].value_counts()
#most of them are above 50, so under fifty is just going to be 0 in all dummies
   
X['sixty'] = (df['age'] == '[50-60)').astype(int)
X['seventy'] = (df['age'] == '[60-70)').astype(int)
X['eighty'] = (df['age'] == '[70-80)').astype(int)
X['ninety'] = (df['age'] == '[80-90)').astype(int)
X['hundred'] = (df['age'] == '[90-100)').astype(int)




 
#print df['admission_type_id'].value_counts()
#1, 2, and 3 are the most common so selecting them. 0's on all 3 are everything else
X['emergency'] = (df['admission_type_id'] == 1).astype(int)
X['urgent'] = (df['admission_type_id'] == 2).astype(int)               
X['elective'] = (df['admission_type_id'] == 3).astype(int)

#print df['discharge_disposition_id'].value_counts()
#1, 3, and 6 are the most common
X['dishome'] = (df['discharge_disposition_id'] == 1).astype(int)
X['dissnf'] = (df['discharge_disposition_id'] == 3).astype(int)
X['dishhs'] = (df['discharge_disposition_id'] == 6).astype(int)

#print df['admission_source_id'].value_counts()
#most common are 7, 1, 17, 4, 6, and 2. 17 is null, let's take 7 and 1
X['admitER'] = (df['admission_source_id'] == 7).astype(int)
X['admitphys'] = (df['admission_source_id'] == 1).astype(int)

#print df['time_in_hospital'].value_counts()
#not sure what these mean, they are not in the codebook, but I assume ordinal so can be left alone? Might be days in hospital
X['time_in_hospital'] = df['time_in_hospital']


#print df['payer_code'].value_counts()
#MC is by far the most common after ?, but not sure what it means. Medicaid?
X['MC'] = (df['payer_code'] == 'MC').astype(int)

#print df['medical_specialty'].value_counts() #most ?
X['MedSpecInternal'] = (df['medical_specialty'] == 'InternalMedicine').astype(int)
X['MedSpecER'] = (df['medical_specialty'] == 'Emergency/Trauma').astype(int)
X['MedSpecFamGP'] = (df['medical_specialty'] == 'Family/GeneralPractice').astype(int)
X['MedSpecCard'] = (df['medical_specialty'] == 'Cardiology').astype(int)
X['MedSpecSurg'] = (df['medical_specialty'] == 'Surgery-General').astype(int)

#print df['num_lab_procedures'].value_counts()#can just use these
X['NumLabProc'] = df['num_lab_procedures']

#print df['num_procedures'].value_counts()#can just use this
X['NumProc'] = df['num_procedures']

X['NumMed'] = df['num_medications']

X['NumOutpatient'] = df['number_outpatient']

X['NumEmerg'] = df['number_emergency']

X['NumInpat'] = df['number_inpatient']

#print df['diag_1'].value_counts
#diag 1, 2, and 3 don't really make sense, discarding

X['NumDiag'] = df['number_diagnoses']

#print df['max_glu_serum'].value_counts()
#most are None, so I don't think this should be used

#print df['A1Cresult'].value_counts()
#90%ish are None, so I think this would be a confounder to include

#print df['metformin'].value_counts()
#just using No and Steady
X['MetNo'] = (df['metformin'] == 'No').astype(int)
X['MetSteady'] = (df['metformin'] == 'Steady').astype(int)

#print df['repaglinide'].value_counts()
#just using Steady??
X['RepSteady'] = (df['repaglinide'] == 'Steady').astype(int)

#print df['nateglinide'].value_counts()
#under a thousand have value, not using this

#print df['chlorpropamide'].value_counts()
#under 100 have value

#print df['glimepiride'].value_counts()
X['GlimSteady'] = (df['glimepiride'] == 'Steady').astype(int)

#print df['acetohexamide'].value_counts()
X['AcetSteady'] = (df['acetohexamide'] == 'Steady').astype(int)

#print df['glipizide'].value_counts() #most popular so far
X['GlipSteady'] = (df['glipizide'] == 'Steady').astype(int)

#print df['glyburide'].value_counts()
X['GlySteady'] = (df['glyburide'] == 'Steady').astype(int)

#print df['tolbutamide'].value_counts() #least popular

#print df['pioglitazone'].value_counts()
X['PiogSteady'] = (df['pioglitazone'] == 'Steady').astype(int)

#print df['rosiglitazone'].value_counts()
X['RosigSteady'] = (df['rosiglitazone'] == 'Steady').astype(int)

#print df['acarbose'].value_counts() #too many Nos

#print df['miglitol'].value_counts() #too many no's

#print df['troglitazone'].value_counts() #too many nos

#print df['tolazamide'].value_counts() #not using

#print df['examide'].value_counts() #all no

#print df['citoglipton'].value_counts() #all no

#print df['insulin'].value_counts()
#using all of these
X['InsulinNo'] = (df['insulin'] == 'No').astype(int)
X['InsulinSteady'] = (df['insulin'] == 'Steady').astype(int)
X['InsulinDown'] = (df['insulin'] == 'Down').astype(int)
X['InsulinUp'] = (df['insulin'] == 'Up').astype(int)

#print df['glyburide-metformin'].value_counts() #under 1000 non No's not using

#print df['glipizide-metformin'].value_counts() #only 13 non Nos

#print df['glimepiride-pioglitazone'].value_counts() #1 non no

#print df['metformin-rosiglitazone'].value_counts() #only 2 non no's

#print df['metformin-pioglitazone'].value_counts() #only 1 non no

#print df['change'].value_counts() 
X['change'] = (df['change'] == 'Ch').astype(int)

#print df['diabetesMed'].value_counts() 
X['DiabetesMed'] = (df['diabetesMed'] == 'Yes').astype(int)

#print X.shape
#101,766 rows, 45 columns
#print X.describe()
#everything seems ok


#create new column NR for not readmitted
df['NR'] = (df['readmitted'] == 'NO').astype(int)
y = df['NR']



# Next I split the data into a training and validating set. I planned on using 5-fold cross validation methods for the model so did not create a seperate test dataset. I chose this to conserve my data. I reserved 30% of the data as a validation set. 

# In[4]:

#train test split. 
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = .3)


# I then created the random forest model to be used. I ran into the initial problem that my computer's RAM could be overloaded by the model. Because of this I had limits on some of the finetuning of the model that I could do. This presented some constraints to the modelling process. I fit the model using a 5-fold cross validation approach. 

# In[5]:

#this can easily overload my computers ram and crash python, fine tuned it to this as the max
forest = RandomForestClassifier(n_estimators=300, oob_score=True, n_jobs = 1) 


# In[6]:

scores = cross_validation.cross_val_score(forest, X_train, y_train, cv=5)
print scores
#this is the score to predict readmitance, averaged around 63%


# After I used Cross validation to set the parameters for the forest I then trained the forest on all of the training data. This is the model that will be teted against the test data set at the end of the project.

# In[8]:

forest.fit(X_train, y_train)


# I then wanted to see the feature importance. To do this I used the existed feature_importances_ that is built into the random forest module. I then plotted a histogram of the feature importance.


fet_ind = np.argsort(forest.feature_importances_)[::-1]
fet_imp = forest.feature_importances_[fet_ind]

fig = plt.figure(figsize=(8,4));
ax = plt.subplot(111)
plt.bar(np.arange(len(fet_imp)), fet_imp, width=1, lw=2)
plt.grid(False)
ax.set_xticks(np.arange(len(fet_imp))+.5)
ax.set_xticklabels(X.columns.values[fet_ind], rotation =90)
plt.xlim(0, len(fet_imp))


# As you can see above the most important features were the number of lab procedures performed, the number of 
#medications the patient was on, the time in the hospital, and the number of diagnoses. These are all things 
#that we would expect to be found in sicker individuals, so it would make sense that they would be important 
#features. They are also somewhat disapointing as they could show that the people being readmitted were 
#readmitted for a problem relating to one of their other diagnoses or an infection aquired in the hospital. 

# Next I created a Naive Bayes model once again using cross validation.I first compared a Gaussian and 
#Multinomial model and found that the Multinomial model performed better. After setting the paramaters
#using cross validation I trained the model on the entire training data set.


#random gaussian naive bayes estimator, just to see what happens
# Instantiate the estimator
mnb = MultinomialNB(alpha = 1, )
# Fit the estimator to the data, leaving out the last five samples
scores = cross_validation.cross_val_score(mnb, X_train, y_train, cv=5)
# Use the model to predict the last several labels
print scores



mnb.fit(X_train, y_train)


# Finally I compared the two models against the test dataset to compare their performance.


print "The Random Forest Score is: " + str(forest.score(X_test, y_test))
print "The Naive Bayes Score is: " + str(mnb.score(X_test, y_test))


# The Random Forest model outperformed the Naive Bayes model however neither model did very well. In 
#the future I would like to test out additional models on the data, as well as spend more time proccessing 
#the data in the hopes of finding a better classification model.
