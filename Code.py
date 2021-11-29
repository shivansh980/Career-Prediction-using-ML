#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing necessary modules and libraries
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import random


# In[ ]:


#path to the csv file
path = 'data.csv'
df = pd.read_csv(path)


# In[ ]:


#object of label encoder
le = LabelEncoder()


# In[ ]:


#reducing number of classes by reducing job roles

df = df.replace(to_replace =["Business Intelligence Analyst", "Business Systems Analyst","CRM Business Analyst","E-Commerce Analyst","Systems Analyst","Systems Security Administrator"], value ="Analyst")

df=df.replace(to_replace =["CRM Technical Developer","Mobile Applications Developer","Web Developer"],value="Applications Developer")

df=df.replace(to_replace=["Data Architect","Database Administrator","Database Developer","Database Manager"],value="Database Engineer")

df=df.replace(to_replace=["Design & UX","UX Designer"],value="Applications Developer")

df=df.replace(to_replace=["Information Security Analyst","Information Technology Auditor","Information Technology Manager"],value="Analyst")

df=df.replace(to_replace=["Network Engineer","Network Security Administrator","Network Security Engineer"],value="Technical Engineer")

df=df.replace(to_replace=["Portal Administrator","Programmer Analyst","Project Manager","Quality Assurance Associate"],value="Project Manager")

df=df.replace(to_replace=["Software Developer","Software Quality Assurance (QA) / Testing","Software Systems Engineer","Solutions Architect"],value="Software Engineer")

df=df.replace(to_replace=["Technical Services/Help Desk/Tech Support","Technical Support"],value="Technical Engineer")


# In[ ]:


#removing other symbols and changing them to underscore
df.columns = [c.replace(' ', '_') for c in df.columns]

df.columns = [c.replace('?', '') for c in df.columns]

df.columns = [c.replace('-', '_') for c in df.columns]

df.columns = [c.replace('/', '_') for c in df.columns]


# In[ ]:


#merging and dropping unncessary features
sum_column = df["talenttests_taken"] + df["olympiads"]

df["talenttests_taken"] = sum_column

df.drop('hackathons', axis=1, inplace=True)

df.drop('olympiads', axis=1, inplace=True)

df.drop("Interested_Type_of_Books",axis=1,inplace=True)

sum_column = df["Extra_courses_did"] + df["certifications"]

df["Extra_courses_did"] = sum_column

df.drop('certifications', axis=1, inplace=True)

df.drop('Taken_inputs_from_seniors_or_elders',axis=1, inplace=True)

df.drop('interested_in_games',axis=1, inplace=True)

df.drop('Job_Higher_Studies',axis=1, inplace=True)

df.drop('In_a_Realtionship',axis=1, inplace=True)

df.drop('Gentle_or_Tuff_behaviour',axis=1, inplace=True)


# In[ ]:


#converting objects with other data types to integer data type 
df.can_work_long_time_before_system=le.fit_transform(df.can_work_long_time_before_system)

df.self_learning_capability=le.fit_transform(df.self_learning_capability)

df.Extra_courses_did=le.fit_transform(df.Extra_courses_did)

df.workshops=le.fit_transform(df.workshops)

df.talenttests_taken=le.fit_transform(df.talenttests_taken)

df.reading_and_writing_skills=le.fit_transform(df.reading_and_writing_skills)

df.memory_capability_score=le.fit_transform(df.memory_capability_score)

df.Interested_subjects=le.fit_transform(df.Interested_subjects)

df.interested_career_area=le.fit_transform(df.interested_career_area)

df.Type_of_company_want_to_settle_in=le.fit_transform(df.Type_of_company_want_to_settle_in)

df.Salary_Range_Expected=le.fit_transform(df.Salary_Range_Expected)

df.Management_or_Technical=le.fit_transform(df.Management_or_Technical)

df.Salary_work=le.fit_transform(df.Salary_work)

df.hard_smart_worker=le.fit_transform(df.hard_smart_worker)

df.worked_in_teams_ever=le.fit_transform(df.worked_in_teams_ever)

df.Introvert=le.fit_transform(df.Introvert)

df.Suggested_Job_Role=le.fit_transform(df.Suggested_Job_Role)


# In[ ]:


#changing various values to high med and low in columns containing a great range of values
df.percentage_in_Algorithms=df.percentage_in_Algorithms.floordiv(10)

df.loc[df.percentage_in_Algorithms < 6 , "percentage_in_Algorithms"] = 1

df.loc[df.percentage_in_Algorithms > 8 , "percentage_in_Algorithms"] = 3

df.loc[df.percentage_in_Algorithms >=6 , "percentage_in_Algorithms"] = 2

df.Acedamic_percentage_in_Operating_Systems=df.Acedamic_percentage_in_Operating_Systems.floordiv(10)

df.loc[df.Acedamic_percentage_in_Operating_Systems < 6 , "Acedamic_percentage_in_Operating_Systems"] = 1

df.loc[df.Acedamic_percentage_in_Operating_Systems > 8 , "Acedamic_percentage_in_Operating_Systems"] = 3

df.loc[df.Acedamic_percentage_in_Operating_Systems >=6 , "Acedamic_percentage_in_Operating_Systems"] = 2

df.Percentage_in_Programming_Concepts=df.Percentage_in_Programming_Concepts.floordiv(10)

df.loc[df.Percentage_in_Programming_Concepts < 6 , "Percentage_in_Programming_Concepts"] = 1

df.loc[df.Percentage_in_Programming_Concepts > 8 , "Percentage_in_Programming_Concepts"] = 3

df.loc[df.Percentage_in_Programming_Concepts >=6 , "Percentage_in_Programming_Concepts"] = 2

df.Percentage_in_Software_Engineering=df.Percentage_in_Software_Engineering.floordiv(10)

df.loc[df.Percentage_in_Software_Engineering < 6 , "Percentage_in_Software_Engineering"] = 1

df.loc[df.Percentage_in_Software_Engineering > 8 , "Percentage_in_Software_Engineering"] = 3

df.loc[df.Percentage_in_Software_Engineering >=6 , "Percentage_in_Software_Engineering"] = 2

df.Percentage_in_Computer_Networks=df.Percentage_in_Computer_Networks.floordiv(10)

df.loc[df.Percentage_in_Computer_Networks < 6 , "Percentage_in_Computer_Networks"] = 1

df.loc[df.Percentage_in_Computer_Networks > 8 , "Percentage_in_Computer_Networks"] = 3

df.loc[df.Percentage_in_Computer_Networks >=6 , "Percentage_in_Computer_Networks"] = 2

df.Percentage_in_Electronics_Subjects=df.Percentage_in_Electronics_Subjects.floordiv(10)

df.loc[df.Percentage_in_Electronics_Subjects < 6 , "Percentage_in_Electronics_Subjects"] = 1

df.loc[df.Percentage_in_Electronics_Subjects > 8 , "Percentage_in_Electronics_Subjects"] = 3

df.loc[df.Percentage_in_Electronics_Subjects >=6 , "Percentage_in_Electronics_Subjects"] = 2

df.Percentage_in_Computer_Architecture=df.Percentage_in_Computer_Architecture.floordiv(10)

df.loc[df.Percentage_in_Computer_Architecture < 6 , "Percentage_in_Computer_Architecture"] = 1

df.loc[df.Percentage_in_Computer_Architecture > 8 , "Percentage_in_Computer_Architecture"] = 3

df.loc[df.Percentage_in_Computer_Architecture >=6 , "Percentage_in_Computer_Architecture"] = 2

df.Percentage_in_Communication_skills=df.Percentage_in_Communication_skills.floordiv(10)

df.loc[df.Percentage_in_Communication_skills < 6, "Percentage_in_Communication_skills"] = 1

df.loc[df.Percentage_in_Communication_skills > 8 , "Percentage_in_Communication_skills"] = 3

df.loc[df.Percentage_in_Communication_skills >=6 , "Percentage_in_Communication_skills"] = 2

df.Percentage_in_Mathematics=df.Percentage_in_Mathematics.floordiv(10)

df.loc[df.Percentage_in_Mathematics < 6 , "Percentage_in_Mathematics"] = 1

df.loc[df.Percentage_in_Mathematics > 8 , "Percentage_in_Mathematics"] = 3

df.loc[df.Percentage_in_Mathematics >=6 , "Percentage_in_Mathematics"] = 2


# In[ ]:


#changing various values to high med and low in columns containing a great range of values
df.Hours_working_per_day=df.Hours_working_per_day.floordiv(3)

df.Logical_quotient_rating=df.Logical_quotient_rating.floordiv(2)

df.coding_skills_rating=df.coding_skills_rating.floordiv(2)

df.public_speaking_points=df.public_speaking_points.floordiv(2)

df.workshops=df.workshops.floordiv(3)

df.Interested_subjects=df.Interested_subjects.floordiv(2)

df.Type_of_company_want_to_settle_in=df.Type_of_company_want_to_settle_in.floordiv(3)


# In[ ]:


#getting a list of size equal to df
points = list(range(len(df)))


# In[ ]:


#dividing training and testing datasets
trainSize = int(0.9*len(df))
testSize  = int(0.1*len(df))


# In[ ]:


trainingSet = random.sample(points,trainSize)

for x in trainingSet:
    points.remove(x)


# In[ ]:


#preparing the dataset by dropping the column to be predicted from the dataset
testSet = points
TrainingSet = df.drop(testSet)
TestingSet   = df.drop(trainingSet)


# In[ ]:


#naming column to be predicted as TrainingX and TrainingY
TrainingY = TrainingSet['Suggested_Job_Role']
TestingY = TestingSet['Suggested_Job_Role']


# In[ ]:


TrainingX = TrainingSet.drop('Suggested_Job_Role',axis="columns")
TestingX = TestingSet.drop('Suggested_Job_Role',axis="columns")


# In[ ]:


#MLP classifier to train data
classifier = MLPClassifier(max_iter=10000).fit(TrainingX, TrainingY)


# In[ ]:


#getting the accuracy of the model 
print(classifier.score(TestingX, TestingY))


# In[ ]:


#predicting for TestingX to calculate the confusion matrix
arr=classifier.predict(TestingX)


# In[ ]:


#using inbuilt, predicted arr, and known values to find the confusion matrix
cm=confusion_matrix(TestingY, arr)


# In[ ]:


#the confusion matrix
confusion_matrix(TestingY, arr)


# In[ ]:


#calculating the class wise accuracies from the confusion matrix
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


# In[ ]:


#class wise accuracies are stores on the diagonals of the matrix. Representing the array. 
cm.diagonal()


# In[ ]:


#Repeating same process for 80:20 data ratio
points1 = list(range(len(df)))


# In[ ]:


trainSize = int(0.8*len(df))
testSize  = int(0.2*len(df))


# In[ ]:


trainingSet = random.sample(points1,trainSize)

for x in trainingSet:
    points1.remove(x)


# In[ ]:


testSet = points1
TrainingSet = df.drop(testSet)
TestingSet   = df.drop(trainingSet)


# In[ ]:


TrainingY = TrainingSet['Suggested_Job_Role']
TestingY = TestingSet['Suggested_Job_Role']


# In[ ]:


TrainingX = TrainingSet.drop('Suggested_Job_Role',axis="columns")
TestingX = TestingSet.drop('Suggested_Job_Role',axis="columns")


# In[ ]:


classifier = MLPClassifier(max_iter=10000).fit(TrainingX, TrainingY)


# In[ ]:


print(classifier.score(TestingX, TestingY))


# In[ ]:


arr=classifier.predict(TestingX)


# In[ ]:


cm=confusion_matrix(TestingY, arr)


# In[ ]:


confusion_matrix(TestingY, arr)


# In[ ]:


cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


# In[ ]:


cm.diagonal()


# In[ ]:


#Repeating same process for 80:20 data ratio
points3 = list(range(len(df)))


# In[ ]:


trainSize = int(0.6*len(df))
testSize  = int(0.4*len(df))


# In[ ]:


trainingSet = random.sample(points3,trainSize)

for x in trainingSet:
    points3.remove(x)


# In[ ]:


testSet = points3
TrainingSet = df.drop(testSet)
TestingSet   = df.drop(trainingSet)


# In[ ]:


TrainingY = TrainingSet['Suggested_Job_Role']
TestingY = TestingSet['Suggested_Job_Role']


# In[ ]:


TrainingX = TrainingSet.drop('Suggested_Job_Role',axis="columns")
TestingX = TestingSet.drop('Suggested_Job_Role',axis="columns")


# In[ ]:


classifier = MLPClassifier(max_iter=10000).fit(TrainingX, TrainingY)


# In[ ]:


print(classifier.score(TestingX, TestingY))


# In[ ]:


arr=classifier.predict(TestingX)


# In[ ]:


cm=confusion_matrix(TestingY, arr)


# In[ ]:


confusion_matrix(TestingY, arr)


# In[ ]:


cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


# In[ ]:


cm.diagonal()


# In[ ]:





# In[ ]:




