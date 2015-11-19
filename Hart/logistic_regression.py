'''
Project : Ensembling two independent logisitic classifier

Classifier I : Logistic classifier for breast cancer model
Classifier II: Logisitic classifier for Titanic survival model

Both the classifiers are built using sklearn
Inputs for the sklearn.logistic regression are :
Regularization = 0.5 [to minimize overfitting]
Training size = 50% of the given data
Test size  50% of the given data
Random Sampling = 10 [To avoid the extraction of sequential sampling]
Hard iterations = 900 
The weight vector is a hyperplane . It is an argmax of product of p(Y/X)
Accuracy = 95% for breast cancer model

Result:
+----------+----------+----------+----------+----------+----------+----------+
|   Name   | Passenge | Cancer-  |   Sex    |  Breast  | Titanic  |    Is    |
|          |   r_Id   |    Id    |          |  Cancer  | Survival |  alive?  |
+==========+==========+==========+==========+==========+==========+==========+
+----------+----------+----------+----------+----------+----------+----------+
+----------+----------+----------+----------+----------+----------+----------+
|  "Whabee |     1239 |  1000045 |   female | malignan |     Died |       No |
|     Mrs. |          |          |          |        t |          |          |
|   George |          |          |          |          |          |          |
| Joseph ( |          |          |          |          |          |          |
| Shawneen |          |          |          |          |          |          |
|   e Abi- |          |          |          |          |          |          |
|   Saab)" |          |          |          |          |          |          |
+----------+----------+----------+----------+----------+----------+----------+
|   "Giles |     1240 |  1000055 |     male |       No | Survived |      Yes |
|      Mr. |          |          |          |   Breast |          |          |
|   Ralph" |          |          |          |   cancer |          |          |
+----------+----------+----------+----------+----------+----------+----------+
| "Walcrof |     1241 |  1000065 |   female |   Benign |     Died |       No |
|  t Miss. |          |          |          |          s|          |          |
|  Nellie" |          |          |          |          |          |          |
+----------+----------+----------+----------+----------+----------+----------+
Logical Explanation:
If there is a male survivor in Titanic wrek, he isn't tested for breast cancer prediction (assuming females has breast cancer)
If someone is survived and has benign (low severity Breast cancer) then she has higher probability to survive. Hence, is_alive case is yes
If someone is dead in Titanic wreck then he/she has low probability to survive. Hence is_alive case is No
If someone is survived in Titanic wreck but has maignant breast cancer then the probability is low to survive. is_alive is No
'''
'''
Libraries used : Sklearn, numpy and texttable
'''
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import scipy
import texttable as table

def correlate_data_sets(cancer_test_data, cancer_model, titanic_test_data, titanic_model, traveller_info, cancer_id):
    '''
    All the test datas are sliced to cancer test dataset to achieved uniformity
    Now test dataset of cancer and titanic are of same size (same rows)
    '''
    np.random.shuffle(titanic_test_data)
    np.random.shuffle(traveller_info)
    np.random.shuffle(cancer_id)
    titanic_test_data = titanic_test_data[:cancer_test_data.shape[0], :]
    traveller_info = traveller_info[:cancer_test_data.shape[0], :]
    cancer_id = cancer_id[:cancer_test_data.shape[0], :]
    tab = table.Texttable()
    rows = [[]] 
    '''
    For each record the prediction model built predicts the result
    Prediction is nothing but a linear multiplication of the weight vector and the test set
    If the result is greater than 1 then titanic survival model predicts "yes" 
    and breast cancer predicts "malignant" (both are independent model)
    '''
    for each_record in range(cancer_test_data.shape[0]):
        sex = traveller_info[each_record][4]
        name = traveller_info[each_record][2] + traveller_info[each_record][3]
        titanic_survival = titanic_model.predict(titanic_test_data[each_record])
        '''
        If sex is female then the cancer rate model is called.
        Else only titanic model is called.
        '''
        if sex == "female":
            cancer_rate = cancer_model.predict(cancer_test_data[each_record])
            if int(cancer_rate) == 2:
                cancer_severity = "Benign"
            else:
                cancer_severity = "malignant"
            if int(cancer_rate) == 4 and int(titanic_survival[0]) == 1:
                is_alive= "No"
            else:
                is_alive= "Yes"
        else:
            is_alive = "Yes"
            cancer_severity = "No Breast cancer"
        if int(titanic_survival[0]) == 0:
            is_alive = "No"
        # All the values are appended into a single row.
        rows.append([name, traveller_info[each_record][0], cancer_id[each_record][0], sex, cancer_severity, "Survived" if int(titanic_survival[0]) == 1 else "Died" , is_alive])    
    tab.add_rows(rows)
    tab.set_cols_align(['r','r','r','r','r','r','r'])
    tab.header(['Name', 'Passenger_Id' , 'Cancer-Id' , 'Sex', 'Breast Cancer', 'Titanic Survival', 'Is alive?'])
    print tab.draw()
    

def main():
    '''
    Breast Cancer data set
    '''
    # Get the breast cancer data
    cancer_data = np.loadtxt("breast-cancer-wisconsin.data", delimiter=',', dtype=str)
    # All the missing values are subsitutes to 0.0
    cancer_data[cancer_data == "?"] = 0.0
    # Extract the cancer ids from the given input
    cancer_id = cancer_data[:, :1]
    # Extract the features from the given input
    input_matrix = cancer_data[:, 1:-1]
    # Extract the output labels
    labels = cancer_data[:, -1]
    # Instantiation of Logistic Regression
    # Regularization to avoid overfitting
    logistic_classifier = LogisticRegression(C=0.5, max_iter = 900)
    # Splitting the datas into training and testing
    # Could have split into training , test and cross-valdation to avoid overfitting.
    train_set, test_set, train_class_label, test_class_label = train_test_split(input_matrix, labels, train_size = 0.5, test_size=0.5, random_state=10)
    # To ease, all the values are converted to float
    train_set=np.array(train_set,dtype=float)
    test_set=np.array(test_set,dtype=float)
    train_class_label=np.array(train_class_label,dtype=float)
    test_class_label=np.array(test_class_label,dtype=float)
    '''Train a machine learning model with the given training set'''
    logistic_classifier.fit(train_set, train_class_label)
    '''
    Titanic Data set
    '''
    titanic_data = np.loadtxt("train.csv", delimiter=',', dtype=str)
    titanic_data[titanic_data == "?"] = 0.0
    titanic_data[titanic_data == ""] = 0.0
    labels = titanic_data[1:, 1]
    # To Ease, all the string columns are removed so that the logistic regression model can be built easily
    # Columns removed are : Passenger Id, Name, Pclass, Embarkment, Sex, Cabin
    # Traveller info contains the information of the passenger's name, id and sex
    titanic_data = titanic_data[1:, 2:-1]
    titanic_data = scipy.delete(titanic_data, [1,2,3,7,9], 1)
    titanic_data=np.array(titanic_data,dtype=float)
    titanic_logistic_classifier = LogisticRegression(C=0.5, max_iter = 900)
    titanic_logistic_classifier.fit(titanic_data, labels)
    
    # Test set of titanic data set
    titanic_test_set = np.loadtxt("test.csv", delimiter=',', dtype=str)
    titanic_test_set[titanic_test_set == "?"] = 0.0
    titanic_test_set[titanic_test_set == ""] = 0.0
    
    # Slice the features from the input
    # To Ease, all the string columns are removed so that the logistic regression model can be built easily
    # Columns removed are : Passenger Id, Name, Pclass, Embarkment, Sex, Cabin
    # Traveller info contains the information of the passenger's name, id and sex
    traveller_info = titanic_test_set[1:, :5]
    titanic_test_set = titanic_test_set[1:, 1:]
    titanic_test_set = scipy.delete(titanic_test_set, [1,2,3,7,9], 1)
    titanic_test_set=np.array(titanic_test_set,dtype=float)
    # Calling the function correlate date
    correlate_data_sets(test_set, logistic_classifier, titanic_test_set, titanic_logistic_classifier, traveller_info, cancer_id)
if __name__ == "__main__":
    main()
        