# -*- coding: utf-8 -*-
"""
Date: Created on Thu Aug 27 11:22:06 2020
Name: Loan approval Modelling Project.
Description: Decision 
1. The main aim of this project is to provide a predictive result on whether the customers is eligible for loan sanction.
2. Loan status feature has been as a target class label.
3. Here Entropy has been used for Attribute selection Mechanism
@Classification Model : Decision Tree
Python version: 3.7
@author:Aakash
"""

# Importing the required packages
import pandas as pd
import pydotplus
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image

# This method is responsible for fetching the data's from the input excel sheet to implement the project 
def importdata():
    try:
        missing_values = ["n/a", "na", "--", " "]
        balanceData = pd.read_csv("LoanApplicantData.csv", na_values = missing_values)
        # Printing the dataswet shape 
        print ("\n-----------------------------------------------------------------------------------------------------------------------")
        print ("\nDataset Shape(Number of Rows in input data sheet, Number of columns in input data sheet): ", balanceData.shape)  
        # Printing the dataset obseravtions 
        print ("\nDataset Head (5 sample records): \n\n",balanceData.head())
        print ("\n-----------------------------------------------------------------------------------------------------------------------\n")
        return balanceData
    except IOError:
        print ("\n\n-----------------------------------------------------------------------------------------------------------------------")
        print ("\n----Could not read file. There is a file not found exception-------")
        print ("\n-----------------------------------------------------------------------------------------------------------------------\n")
        
# This Method handles data preprocessing techniques such as Data cleaning, Data Transformation to provide quality data
def dataPreprocessorMethod(data):
    #Filling n/a data with max_value_count 
    term_count = data["Loan_Amount_Term"].value_counts().idxmax()
    data["Loan_Amount_Term"].fillna(term_count, inplace=True)    
    
    max_dependent = data["Dependents"].value_counts().idxmax()
    data["Dependents"].fillna(max_dependent, inplace=True)
    
    is_max_Married = data["Married"].value_counts().idxmax()
    data["Married"].fillna(is_max_Married, inplace=True)
    
    max_Gender = data["Gender"].value_counts().idxmax()
    data["Gender"].fillna(max_Gender, inplace=True)
    
    is_max_SelfEmployed = data["Self_Employed"].value_counts().idxmax()
    data["Self_Employed"].fillna(is_max_SelfEmployed, inplace=True)
    
    #Filling n/a data with mean
    median = data["LoanAmount"].median()
    data["LoanAmount"].fillna(median, inplace=True)
    
    #Drop row those doesn't have credit History
    data.dropna(subset=["Credit_History"], inplace=True)
    data["Credit_History"].isnull().sum()
    
    targetVariable = data.iloc[:,-1]
    feature = data.iloc[:,:-1].values
    feature = data.drop(columns=['Loan_ID', 'Loan_Status'])
    
    # Data transformation
    encode = LabelEncoder()
    feature.iloc[:,0] = encode.fit_transform(feature.iloc[:,0])
    feature.iloc[:,1] = encode.fit_transform(feature.iloc[:,1])
    feature.iloc[:,2] = encode.fit_transform(feature.iloc[:,2])
    feature.iloc[:,3] = encode.fit_transform(feature.iloc[:,3])
    feature.iloc[:,4] = encode.fit_transform(feature.iloc[:,4])
    feature.iloc[:,10] = encode.fit_transform(feature.iloc[:,10])
    feature_cols = feature.iloc[:,:].columns
    
    return feature, targetVariable, feature_cols

# This definition helps us splitting the LoanApplicantData.csv input records into Training set and test set of data.
def splitDataset (feature,targetVariable):
    # 70% training and 30% test
    X_train, X_test, y_train, y_test = train_test_split(feature, targetVariable, test_size=0.3, random_state=1) 
    return X_train, X_test, y_train, y_test

# Decision making method
def prediction(X_test, clf_object): 
    y_pred = clf_object.predict(X_test)  
    return y_pred 

# This method calculates the Accuracy and generates Report and provide the Confusion Matrix
def cal_accuracy(y_test, prediction):
    print("\n\nConfusion Matrix: \n\n", confusion_matrix(y_test, prediction)) 
    print("\nAccuracy : ", accuracy_score(y_test,prediction)*100) 
    print("\nReport : \n", classification_report(y_test, prediction))
    print ("\n-----------------------------------------------------------------------------------------------------------------------\n")
        
def main():   
    # Building Phase
    data = importdata()
    feature, targetVariable, feature_cols = dataPreprocessorMethod(data)
    X_train, X_test, y_train, y_test = splitDataset(feature,targetVariable)
    
    
    # Create Decision Tree classifer object before using Entropy
    clf = DecisionTreeClassifier()
    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)
    y_pred = prediction(X_test, clf)
    print ("\n-----------------------------------------------------------------------------------------------------------------------")
    print ("\nDecision tree is created and evaluated. Refer LoanDetail_DecisionTree.png file for decision tree created.")
    print ("\nAccuracy details are as follows:") 
    cal_accuracy(y_test, y_pred)
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True,feature_names = feature_cols,class_names=['N','Y'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png('LoanDetail_DecisionTree.png')
    Image(graph.create_png())
       
    
    #Optimizing Decision Tree Performance
    # Create Decision Tree classifer object after using Entropy
    clf1 = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    # Train Decision Tree Classifer
    clf1 = clf1.fit(X_train,y_train)
    #Predict the response for test dataset
    y_preds = prediction(X_test,clf1)
    print ("\n-----------------------------------------------------------------------------------------------------------------------")
    print ("\nDecision tree is refined. Refer LoanDetail_DecisionTree_Refined.png file for the refined decision tree.")
    print ("\nAccuracy details after refining the decision tree are as follows:") 
    cal_accuracy(y_test, y_preds)
    dot_data = StringIO()
    export_graphviz(clf1, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True,feature_names = feature_cols,class_names=['N','Y'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png('LoanDetail_DecisionTree_Refined.png')
    Image(graph.create_png())
    
    print ("End Of Program!!!!!")
    print ("\n-----------------------------------------------------------------------------------------------------------------------")
    
# Calling main function 
if __name__=="__main__": 
    main()
# End of program
