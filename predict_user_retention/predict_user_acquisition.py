import pandas as pd
import numpy as np
import datetime 
from numpy import array

from sklearn import (linear_model,tree)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt
import graphviz

from  scipy.sparse import coo_matrix, hstack

#load data
user_engagement = pd.read_csv('takehome_user_engagement-intern.csv')
users = pd.read_csv('takehome_users-intern.csv')

#data exploration
print list(users)
print list(user_engagement)
print users.shape, user_engagement.shape
#print users.head, user_engagement.head

#generate adopted_user, who log into the product on three separate days in at least one seven-day period, in users data
users['adopted_user'] = 0

def isAdoptedUser(userID) :
    #userID never logged in
    if userID not in user_engagement['user_id'] :
        return False
    
    user_log = user_engagement.loc[user_engagement['user_id'] == userID]
    #return False, if userID had logged in fewer than three times
    if len(user_log) < 3 :
        return False
    
    #create a set of logged-in days given the userID
    user_log_days = set()
    for time in user_log['time_stamp'] :
        day = pd.Timestamp(time).date()
        user_log_days.add(day)
    
    #return False, if the number of days between the first and last logged-in dates is fewer than 3
    firstDay = min(user_log_days)
    lastDay = max(user_log_days)
    if (lastDay - firstDay).days < 3 :
        return False

    #return False, if userID had logged in fewer than three days
    if len(user_log_days) < 3 :
        return False

    #return True, if userID logged in the product at least three times in a 7 day period
    user_log_days =  sorted(user_log_days)
    for i in range(len(user_log_days)-2) :
        if (user_log_days[i+2] - user_log_days[i]).days <= 7 :
            return True

    return False

#if a user is an adopted user, set 'adopted_user' to be 1for user in users['object_id'] :
for user in users['object_id']:
    if isAdoptedUser(user) :
        users.loc[users['object_id'] == user, 'adopted_user'] = 1
print users['adopted_user'] 
print users['adopted_user'].value_counts()


#modeling of early user adoption

#create dummy variables for creation_source
create_source_values = array(users['creation_source'])
print "creation source unique values"
print users['creation_source'].value_counts()
#integer encode
l_enc = LabelEncoder()
integer_encoded_create_source = l_enc.fit_transform(create_source_values)
print create_source_values, integer_encoded_create_source
#binary encode
enc = OneHotEncoder()
integer_encoded_create_source = integer_encoded_create_source.reshape(len(integer_encoded_create_source), 1)
onehot_encoded_create_source = enc.fit_transform(integer_encoded_create_source)
print "one hot encoded creation_source's shape"
print onehot_encoded_create_source.shape

#create dummy variables for email_domain
email_domain_values = array(users['email_domain'])
#integer encode
integer_encoded_email_domain = l_enc.fit_transform(email_domain_values)
#binary encode
integer_encoded_email_domain = integer_encoded_email_domain.reshape(len(integer_encoded_email_domain), 1)
onehot_encoded_email_domain = enc.fit_transform(integer_encoded_email_domain)
#print onehot_encoded_email_domain.shape

#get predicting and response variables X, y
X = hstack([users.loc[:, ['last_session_creation_time','opted_in_to_mailing_list','enabled_for_marketing_drip','org_id','invited_by_user_id']], onehot_encoded_create_source]).toarray()
y = users['adopted_user']
X_header = ['last_session_creation_time','opted_in_to_mailing_list','enabled_for_marketing_drip','org_id','invited_by_user_id','Guest_Invite','Org_Invite','Personal_Projects','SighUp','SignUp_Google_Auth']

#replace nan with zero 
X = pd.DataFrame(data=X)
X.fillna(0, inplace = True)

#split train test data for X, y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
print list(X), X.shape, X_train.shape

#logsitic regression on X_train, y_train
logistic = linear_model.LogisticRegression()
logsitic = logistic.fit(X_train, y_train)
print logistic.coef_

y_pred_log = logistic.predict(X_test)
print logistic.score(X_test, y_test)
confusion_matrix = confusion_matrix(y_test, y_pred_log)
print confusion_matrix
print classification_report(y_test,y_pred_log)


#decision tree
decisionTree = tree.DecisionTreeClassifier(max_depth=5)
decisionfit = decisionTree.fit(X_train, y_train)
y_pred_tree = decisionfit.predict(X_test)
print "decision tree accuracy rate"
print decisionfit.score(X_test, y_test)

#roc curve
tree_roc_auc = roc_auc_score(y_test, decisionfit.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, decisionTree.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label = 'decision tree (area = %0.2f)' % tree_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')

#dot_data = tree.export_graphviz(decisionfit, out_file = 'tree.dot', feature_names = X, class_names = y, filled = True, rounded = True, special_characters = True) 
dot_data = tree.export_graphviz(decisionfit, out_file = 'tree.dot',feature_names =X_header)
graph = graphviz.Source(dot_data) 

#graph.render("decision tree")


#randomForest
rfc = RandomForestClassifier(max_depth = 3 , random_state = 0)
rfc.fit(X_train,y_train)
y_pred_rfc = rfc.predict(X_test)
#roc curve
rfc_roc_auc = roc_auc_score(y_test, rfc.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, rfc.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label = 'Random Forest (area = %0.2f)' % rfc_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


#gradient boosting
gb = GradientBoostingClassifier(n_estimators = 10)
gb.fit(X_train,y_train)
y_pred_gb = gb.predict_proba(X_test)[:,1]
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_pred_gb)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_gb, tpr_gb, label='GBT')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
