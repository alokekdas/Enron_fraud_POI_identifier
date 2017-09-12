# ---------------------------------------
# File: poi_id_randomf.py
#
# Classifier: Random Forest
#
# Input: final_project_dataset.pkl
#
# Output: my_dataset.pkl
#         my_feature_list.pkl
#         my_classifier.pkl
# ---------------------------------------
import sys
import pickle
import numpy as np
from matplotlib import pyplot as plt

# Data spliting
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

# Feature scaling
from sklearn.preprocessing import MinMaxScaler

# Feature selection
from sklearn.feature_selection import SelectKBest, f_classif

# Principal Component Analysis
from sklearn.decomposition import PCA

# Classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Pipeline
from sklearn.pipeline import Pipeline

# grid search
from sklearn.model_selection import GridSearchCV

# Performance metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Helper functions
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier

# Explore the dataset
# --------------------
# Load the dataset
# --------------------
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
print "Number of records in the dataset:", len(data_dict)
# --------------------
# What are the features available in the dataset?
# --------------------
one_dict = data_dict[list(data_dict)[0]]
print "No. of features:", len(one_dict)
list_of_all_features = list(one_dict)
print "The features are:"
print list_of_all_features
# -----------------------
# How many POIs are there in the dataset?
# -----------------------
total_no_of_pois = 0
for key in data_dict:
    if data_dict[key]['poi'] == 1:
        total_no_of_pois +=1
print "Total number of POIs:", total_no_of_pois
# -----------------------
# How many non-POIs are there in the dataset?
# -----------------------
total_no_of_non_pois = 0
for key in data_dict:
    if data_dict[key]['poi'] == 0:
        total_no_of_non_pois +=1
print "Total number of non-POIs:", total_no_of_non_pois
print
# -------------------------------------
# Which features do not have any values
# -------------------------------------
print "Features with count of null values:"
print
print "{0:18s} \t{1:21s}".format("Feature", "Null value count")
print "{0:18s} \t{1:21s}".format("-------", "----------------")
for a_feature in list_of_all_features:
    nan_value = 0
    for key in data_dict:
        feature_val = data_dict[key][a_feature]
        if feature_val == "NaN":
           nan_value += 1
    print "{0:25s} \t{1:3d}".format(a_feature, nan_value)

# ---------------------------
# Feature Selection
# ----------------------------
features_list = ['poi','salary', 'bonus', 
                 'deferral_payments', 'total_payments', 
                 'exercised_stock_options', 'restricted_stock', 
                 'restricted_stock_deferred',
                 'total_stock_value', 
                 'expenses', 'director_fees', 'deferred_income', 
                 'long_term_incentive', 
                 'to_messages', 'from_messages',
                 'from_poi_to_this_person',
                 'from_this_person_to_poi' ] 
# --------------------------------
# Outlier detection and removal
# --------------------------------
print "Checking for any outliers ...."
print
# Create numpy arrays out of the selected features in the dataset
data = featureFormat(data_dict, features_list, sort_keys = True)

for point in data:
    plt.scatter(point[1], point[2])
plt.xlabel("Salary")
plt.ylabel("Bonus")
plt.title("Salary vs. Bonus")
plt.show()

# Find out who this outlier is:
sorted_data = sorted(data, key=lambda x:x[1], reverse=True)
outlier_salary = int(sorted_data[0][1])
outlier_bonus = sorted_data[0][2]

for key in data_dict:
    if data_dict[key]["salary"] == outlier_salary:
        print "Outlier key = {}, Outlier salary = {}".format(key, data_dict[key]["salary"])
        print data_dict[key]
# ---------------------------
# Remove the outlier
# ---------------------------
print "Removing the record of 'TOTAL' from the dataset ..."
del data_dict['TOTAL']
print "New samples in the dataset:", len(data_dict)

# Extract the feature set again after removing the outlier
data = featureFormat(data_dict, features_list, sort_keys = True)
# How does the plot now look like?
for point in data:
    plt.scatter(point[1], point[2])
plt.xlabel("Salary")
plt.ylabel("Bonus")
plt.title("Salary vs. Bonus")
plt.show()

# Are there any more outliers?
print "Who are the people who got > 1M salary or > 7M bonus?:"
for key in data_dict:
    key_salary = data_dict[key]['salary']
    key_bonus  = data_dict[key]['bonus']
    if key_salary != 'NaN' and key_bonus != 'NaN':
       if key_salary > 1000000 or key_bonus > 7000000:
          print key, key_salary, key_bonus

# -------------------------------
# Any non-person in the dataset? Such entities will typically
# have no salary, bonus, restricted stock and possibly no 
# e-mail address
# -------------------------------
for key in data_dict:
    key_salary = data_dict[key]['salary']
    key_bonus = data_dict[key]['bonus']
    key_restricted_stock = data_dict[key]['restricted_stock']
    key_email_address = data_dict[key]['email_address']
    if key_salary == 'NaN' and \
       key_bonus == 'NaN' and \
       key_restricted_stock == 'NaN' and \
       key_email_address == 'NaN':
       print key

# ------------------
# There is a Travel Agency in the dataset
# ------------------
print data_dict['THE TRAVEL AGENCY IN THE PARK']
# ----------------------------------
# This record contains some financial information. We will retain
# this record in the dataset for analysis
# -----------------------------------

# -------------------------------
# New feature creation
# -------------------------------
my_dataset = data_dict
for key in my_dataset:
    # ----
    # Calculate fraction of messages from this person to poi
    # ----
    key_fr_msgs = my_dataset[key]["from_messages"] 
    key_fr_msgs_to_poi = my_dataset[key]["from_this_person_to_poi"]
    if key_fr_msgs_to_poi != 'NaN' and key_fr_msgs != 'NaN':
       fraction_msgs_to_poi = key_fr_msgs_to_poi/float(key_fr_msgs)
    else:
       fraction_msgs_to_poi = 0
    my_dataset[key]["fraction_from_this_person_to_poi"] = fraction_msgs_to_poi
    # -----
    # Calculate fraction of messages from poi to this person
    # -----
    key_to_msgs = my_dataset[key]["to_messages"] 
    key_to_msgs_fr_poi = my_dataset[key]["from_poi_to_this_person"]
    if key_to_msgs_fr_poi != 'NaN' and key_to_msgs != 'NaN':
       fraction_msgs_fr_poi = key_to_msgs_fr_poi/float(key_to_msgs)
    else:
       fraction_msgs_fr_poi = 0
    my_dataset[key]["fraction_to_this_person_from_poi"] = fraction_msgs_fr_poi

# Add these two new features in the features list
features_list.append('fraction_from_this_person_to_poi' )
features_list.append('fraction_to_this_person_from_poi' )
print features_list

# Extract the feature set again now that the outlier has 
# been removed and two new features have been added.
data = featureFormat(my_dataset, features_list, sort_keys = True)
print data[0]

# Do the fraction of e-mail messages to/from poi indicate
# anything obviously important?  We will plot them to visualize.
for point in data:
    if point[0] == 1:
       point_color = 'red'
    else:
       point_color = 'blue'
    plt.scatter(point[17], point[18], color=point_color)
type1 = plt.scatter(0.2, 0.2, marker = 'o', color='blue')
type2 = plt.scatter(0.2, 0.2, marker = 'o', color='red')
plt.xlabel("Fraction of messages from this person to poi")
plt.ylabel("Fraction of messages to this person from poi")
plt.legend((type1, type2), ('non-poi', 'poi'), loc='upper right', scatterpoints=1)
plt.title("Fraction of messages to/from poi")
plt.show()

### Extract features and labels from dataset for local testing
print "Extract features and labels from dataset for local testing"
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# ----------------------------------------
# Random Forest classifier
# ----------------------------------------
scaler = MinMaxScaler()
skb = SelectKBest()
pca = PCA()
randomf = RandomForestClassifier()

Pipe = Pipeline(steps=[('scaling',scaler),("SKB", skb),("PCA", pca),("randforest", randomf)])

param_grid = {"SKB__k":[10, 12, 16, 18], "PCA__n_components":[6, 8, 10], "PCA__whiten":[True], 
              "randforest__n_estimators":[8, 10, 12], "randforest__max_features":[2, 4]}

# -------------------------------------------------
# StratifiedShuffleSplit to create 100 folds with 30% test set
# --------------------------------------------------
stashsp = StratifiedShuffleSplit(n_splits=100, test_size=0.3, random_state=42)

gs = GridSearchCV(Pipe, param_grid=param_grid, scoring='f1', n_jobs=1, cv=stashsp)
gs.fit(features, labels)
print "Best parameters from GridSearchCV are:", gs.best_params_
print "Best estimator found by grid search:", gs.best_estimator_
clf = gs.best_estimator_

# ------------------------------------------
# At the tail end of the pipe
# ------------------------------------------
# Dump the classifier, dataset and features list
dump_classifier_and_data(clf, my_dataset, features_list)
# ---------------------------------------------
# Evaluate the performance of the model
# --------------------------------------------
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.2, random_state=42)
pred = clf.predict(features_test)
print classification_report(labels_test, pred)
print confusion_matrix(labels_test, pred)
test_classifier(clf, my_dataset, features_list, folds=100)
