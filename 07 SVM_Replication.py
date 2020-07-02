# Support Vector Model
# Vincent Weiss
# 14.06.2020

# Overview
#0 imports
#1.0 define preprocessing functions
#2.0 Figure 5
        # 2.1 Data Selection + settings
        # 2.2 training and extracting the separating line
        # 2.3 plot results figure 5
#3.0 Table 4 and 5
        #3.1 set parameters and preprocessing
        #3.2 perform cross validation and print out results
#4.0 Figure 6 and Table 3
        #4.1 set parameters and preprocessing
        #4.2 fit classifier, make predictions and transform predictions to probabilities
        #4.3 Figure 6: probability regimes over time
        #4.4 Table 3: ex post probability of large negative events, for each regime
#5.0 Figure 4: 'heatmap' of prediction values
#6.0 weighted precision and recall, with shuffling
        #6.1 settings and preprocessing
        #6.2 iterate through all kfold.split indices
        #6.3 Report weighted Recall and Precision

#0 ____________________________________________________
# imports
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm


# turn of some warnings not needed
pd.set_option('mode.chained_assignment', None)
# and make everything visible
pd.set_option("display.max_rows", None, "display.max_columns", None)

MSCI_Data=pd.read_csv("DATA_SVM_JV_Preprocessed.csv")
MSCI_Data.set_index('Unnamed: 0', inplace=True)
MSCI_Data.index.name = None

#1.0 ____________________________________________________
# define preprocessing functions

# select either 1, 12, or 60-month classification, drop the other classifications
# and rewrite classifications as 1 and 0
def load_MSCI(Month):
    MSCI_Data_ = MSCI_Data[['Date',\
                            'Total Return Index',\
                            'Price/Book Ratio',\
                            'Dividend Yield',\
                            str(Month)+' Month Return',\
                            str(Month)+' Month Class']]

    MSCI_Data_.dropna(inplace=True)

    # rewrite classifications:
    # a period of low returns is classified with 1 (positive event) (previously -1)
    # a period of normal/high returns is classified with 0 (negative event) (previously 1)
    i=0
    while i < len(MSCI_Data_):
        if MSCI_Data_[str(Month)+' Month Class'].iloc[i]==-1:
            MSCI_Data_[str(Month)+' Month Class'].iloc[i]=1
        else:
            MSCI_Data_[str(Month)+' Month Class'].iloc[i]=0
        i+=1
    return MSCI_Data_


# select only relevant information, output train and test data subsets 
def Train_Test_Data(MSCI_Data_, Month, share, scale):
    # used for in cases where no cross validation was used
    Train_Length = int(round(len(MSCI_Data_)*share))
    Test_Length = int(len(MSCI_Data_)-Train_Length)

    # save the full range without shuffling-> not for training but for prediction only
    X_Full = MSCI_Data_[['Price/Book Ratio', 'Dividend Yield']].iloc[:]
    Y_Full = MSCI_Data_[[str(Month)+' Month Class']].iloc[:]
    
    # return 100% of the MSCI_Data_ but randomized
    MSCI_Data_=MSCI_Data_.sample(frac=1).reset_index(drop=True)
    
    X_Train = MSCI_Data_[['Price/Book Ratio', 'Dividend Yield']].iloc[0:Train_Length]
    Y_Train = MSCI_Data_[[str(Month)+' Month Class']].iloc[0:Train_Length]
    X_Test = MSCI_Data_[['Price/Book Ratio', 'Dividend Yield']].iloc[Train_Length:]
    Y_Test = MSCI_Data_[[str(Month)+' Month Class']].iloc[Train_Length:]

    if scale == True:
    # the scaler can be used, but was not used to construct the some of the plots
    #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
        scaler = StandardScaler()
        scaler.fit(X_Train)
        X_Train=scaler.transform(X_Train)
        scaler = StandardScaler()
        scaler.fit(X_Test)
        X_Test=scaler.transform(X_Test)
        scaler = StandardScaler()
        scaler.fit(X_Full)
        X_Full=scaler.transform(X_Full)
    return (X_Train, X_Test, Y_Train, Y_Test, X_Full)


# select only relevant information, output train and test data subsets
def K_Fold_Train_Test_Data(MSCI_Data_, Month):
    
    MSCI_Data_=MSCI_Data_.sample(frac=1).reset_index(drop=True)
    
    # Entire X (Input) includes P/B and divY
    X_Kfold= MSCI_Data_[['Price/Book Ratio', 'Dividend Yield']]
    
    # Entire Y (Output) consists of binary classification for respective time horizon
    Y_Kfold = np.array(MSCI_Data_[[str(Month)+' Month Class']])

    # Standardize the input data by removing mean and transform to unit variance
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    scaler = StandardScaler()
    scaler.fit(X_Kfold)
    X_Kfold=scaler.transform(X_Kfold)
    
    # return X and Y for the K_fold case
    return (X_Kfold, Y_Kfold)

#2.0 ____________________________________________________
# Figure 5

#2.1 ____________________________________________________
# Data Selection + settings

# m: investment horizon : 60, 12 or 1
# Train_fraction: share of training data
m=60
Train_fraction = 0.9

# Define Data used for plotting (this section is not used for Kfold training and fitting)
MSCI_here=load_MSCI(Month=m)
X_Train, X_Test, Y_Train, Y_Test, X_Full = \
    Train_Test_Data(MSCI_here, Month=m, share = Train_fraction, scale = False)

# fit model
# define classifier settings
clf = svm.SVC(C=1, kernel= 'linear')

#2.2 ____________________________________________________
# training and extracting the separating line & contour plot values

# fit the classifier
clf.fit(X_Train, np.ravel(Y_Train))

# contourplot data
# Produce probabilistic output (Platt(2000))
Calibration = CalibratedClassifierCV(base_estimator=clf, method='sigmoid', cv='prefit')
Calibration_Fit = Calibration.fit(X_Train, np.ravel(Y_Train))
cmap = cm.get_cmap('jet')

# meshgrid with coordinates from P/B and DivY
#https://gawron.sdsu.edu/python_for_ss/course_core/book_draft/text/linear_classifier_svm.html
color_xx, color_yy = np.meshgrid(np.linspace(0, 4.5, num = 100),\
                                 np.linspace(1, 6, num=100))

# np.c_ : Translates slice objects to concatenation along the second axis. (short form)
Z =Calibration_Fit.predict_proba(np.c_[color_xx.ravel(), color_yy.ravel()])

# choose the first column only (Probability for class one and not probability for class 0)
Z=Z[:,[1]]
Z = Z.reshape(color_xx.shape)

# extract the separating hyperlane (line in this case)
#https://scikit-learn.org/stable/auto_examples/svm/plot_svm_margin.html
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]

margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
yy_down = yy - np.sqrt(1 + a ** 2) * margin
yy_up = yy + np.sqrt(1 + a ** 2) * margin

#2.3 ____________________________________________________
# plot results
cmap = cm.get_cmap('jet')

plt.figure(figsize=(12, 6))

# ommitted, but can be used if needed
#plt.plot(xx, yy, 'k-')
# margin to the right
#plt.plot(xx, yy_down, 'k--')
# margin to the left
#plt.plot(xx, yy_up, 'k--')

plt.contour(color_xx, color_yy, Z, levels = 10, linestyles = 'solid', colors='black')
plt.contourf(color_xx, color_yy, Z,levels = 10, cmap=cmap)

# True classification and P/B and DivY taken from original dataframe
plt.scatter(MSCI_here[MSCI_here['60 Month Class']==0]['Price/Book Ratio'],\
            MSCI_here[MSCI_here['60 Month Class']==0]['Dividend Yield'], s=10, color='magenta')

plt.scatter(MSCI_here[MSCI_here['60 Month Class']==1]['Price/Book Ratio'],\
            MSCI_here[MSCI_here['60 Month Class']==1]['Dividend Yield'], s=30, color='aqua', marker='x')

plt.ylim(1, 6)
plt.xlim(0.51,4.5)
plt.title('SVM, Linear Kernel, Decision Boundary', fontsize=15, fontweight='bold')
plt.xlabel('P/B', fontsize=15)
plt.ylabel('divY', fontsize=15)

def format_functiony(y, pos):
    y_formatted = str(y/100)
    return y_formatted

ax=plt.gca()
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_functiony))
ax.tick_params(direction='in')
plt.show()

#3.0 ____________________________________________________
#Table 4 and 5

#3.1 ____________________________________________________
# set parameters and preprocessing
# m: investment horizon : 60, 12 or 1
# num_folds: number of folds
# kernel: 'linear' or 'rbf' or 'poly'
# gamma: 0.4 or 1.4

m=60
num_folds = 3
kernel = 'poly'
gamma = 1.4

# for K-fold cross validation the input is the full range of data
MSCI_here=load_MSCI(Month=m)
X_Kfold, Y_Kfold = K_Fold_Train_Test_Data( MSCI_here, Month=m)

clf = svm.SVC(C=1, kernel= kernel, gamma = gamma)

#3.2 ____________________________________________________
# perform cross validation and print out results
# change settings above if needed

result_dict = cross_validate(clf, X_Kfold,\
                             np.ravel(Y_Kfold),\
                             cv=num_folds, scoring=('accuracy','f1','roc_auc', 'precision', 'recall'))

Accuracy=result_dict['test_accuracy']
F1 = result_dict['test_f1']
ROC = result_dict['test_roc_auc']
Precision = result_dict['test_precision']
Recall = result_dict['test_recall']

print('\n_______________________________\nCross Validation Results:\n')
print('Accuracy:    '+str(round(Accuracy.mean(),3))+'\n'+\
      'F1:          '+str(round(F1.mean(),3))+'\n'+\
      'ROC:         '+str(round(ROC.mean(),3))+'\n'+\
      'Precision:   '+str(round(Precision.mean(),3))+'\n'+\
      'Recall:      '+str(round(Recall.mean(),3)))

#4.0 ____________________________________________________
# Figure 6 and Table 3

#4.1 ____________________________________________________
# set parameters and preprocessing

m=60
Train_fraction = 0.9
clf = svm.SVC(C=1, kernel= 'linear')

# Define Data used for plotting (this section is not used for Kfold training and fitting)
MSCI_here=load_MSCI(Month=m)
X_Train, X_Test, Y_Train, Y_Test, X_Full = \
    Train_Test_Data(MSCI_here, Month=m, share = Train_fraction, scale = True)

#Regimes
# calibrate output using platt's sigmoid approach

#4.2 ____________________________________________________
# fit classifier, make predictions and transform predictions to probabilities
clf.fit(X_Train, np.ravel(Y_Train))
Calibration = CalibratedClassifierCV(base_estimator=clf, method='sigmoid', cv='prefit')
Calibration_Fit = Calibration.fit(X_Train, np.ravel(Y_Train))
Probabilities = Calibration_Fit.predict_proba(X_Full)

#4.3 ____________________________________________________
# Figure 6: probability regimes over time

# custom list of x_ticks, one tick every 60 months
list_xticks=[]
MSCI_Data_Plot=MSCI_here['Date']
i = 0
while i<len(MSCI_here):
    MSCI_Data_Plot.iloc[i]=MSCI_here['Date'].iloc[i][0:7]
    list_xticks.append(MSCI_here['Date'].iloc[i])
    i+=60

plt.figure(figsize=(12, 6))

# Posterior Probability regimes
i=0
Regimes = []
while i<len(MSCI_here):
    
    # 1/10 < 0.15?  ->  2/10 < 0.15?  -> save 2 as Regime for this observation
    regime_count = 1
    while (regime_count/10)<Probabilities[i,1]:
        regime_count+=1
        
    Regimes.append(regime_count)
    i+=1

plt.plot(MSCI_Data_Plot.iloc[:len(Regimes)], np.array(Regimes))

plt.xticks(list_xticks)
plt.title(\
    'Posterior Probability Regimes Dec 1974 - \Mar 2015 estimated with Platt\'s (2000) Sigmoid Approach', \
    fontsize=11, fontweight='bold')
plt.ylabel('Probability Regimes', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.locator_params(axis='y', nbins=10)
plt.xlim(0, len(Regimes)-1)
ax= plt.gca()
ax.tick_params(direction='in')
plt.show()

#4.4 ____________________________________________________
# Table 3: ex post probability of large negative events, for each regime

MSCI_Data['Posterior Regime']='Nan'
MSCI_Data['Posterior Regime'].iloc[0:len(Regimes)]=Regimes

Regime_List=[]
Ex_Post_List=[]
Number_Total_Observations=[]

i=1
while i<11:
    
    # start by selecting regime one: iff posterior regime in the dataset is equal to the regime that
    # is currently looked for, then add this regime and the classification to the new dataframe
    Single_R=MSCI_Data[MSCI_Data['Posterior Regime']==i][['Posterior Regime','60 Month Class']]
    
    # further examine the Single_R dataframe
    # count Positive classifications in Single_R and divide by the length
    # != 0 to avoid dividing by 0
    if len(Single_R) != 0:
        Singel_Ex_Post=len(Single_R[Single_R['60 Month Class']==-1])/len(Single_R)
    else:
        Single_Ex_Post = 'nan'
    
    # store the results for regime i
    Regime_List.append(i)
    Ex_Post_List.append(Singel_Ex_Post)
    Number_Total_Observations.append(len(Single_R))
    i+=1

Table_2 = pd.DataFrame (columns = \
                            ['Posterior Probability Regimes',\
                             'Ex-Post Probability',\
                             'Number of Total Realizations'])

print('\n_______________________________\nPosterior Probability and Ex-Post Probability Table: \n')
Table_2['Posterior Probability Regimes']=Regime_List
Table_2['Ex-Post Probability']=Ex_Post_List
Table_2['Number of Total Realizations']=Number_Total_Observations
display(Table_2)

#5.0 ____________________________________________________
# Figure 4: 'heatmap' of prediction values

cmap = cm.get_cmap('jet_r')

# iterate through all inputs and predictions
# PB is .iloc[i][2]
# DivY is .iloc[i][3]
# color is assigned from the Probabilities output

i=0
while i < len(X_Full):
    plt.scatter(MSCI_here.iloc[i][2],\
                MSCI_here.iloc[i][3],\
                color=cmap(Probabilities[i][0]-0.01), s=10)
    i+=1  

# change some formats
ax=plt.gca()
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_functiony))
ax.tick_params(direction='in')
plt.ylim(1, 6)
plt.xlim(0.51,4.5)
plt.title('SVM, Linear Kernel, Probabilities', fontsize=15, fontweight='bold')
plt.xlabel('P/B', fontsize=15)
plt.ylabel('divY', fontsize=15)

def format_functiony(y, pos):
    y_formatted = str(y/100)
    return y_formatted

ax=plt.gca()
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_functiony))
ax.tick_params(direction='in')

plt.show()

#6.0 ____________________________________________________
# weighted precision and recall, with shuffling
# accuracy should be the same as in the above so it is not reported

#6.1 ____________________________________________________
# settings and preprocessing
num_folds = 10
kfold = KFold(n_splits=num_folds, shuffle=True)

# set parameters
# m: investment horizon : 60, 12 or 1
# num_folds: number of folds
# kernel: 'linear' or 'rbf' or 'poly'
# gamma: 0.4 or 1.4

m=60
kernel = 'linear'
gamma = 1.4

# for K-fold cross validation the input is the full range of data
MSCI_here=load_MSCI(Month=m)
X_Kfold, Y_Kfold = K_Fold_Train_Test_Data( MSCI_here, Month=m)

P_Pred =0
TP = 0
TP_list=[]
FP = []
Pos = []
Pos_Pred = []

#6.2 ____________________________________________________
# iterate through all kfold.split indices
# store the relevant values for recall and precision

for train_index, test_index in kfold.split(X_Kfold, Y_Kfold):
    clf = svm.SVC(C=1, kernel= kernel, gamma = gamma)
    clf.fit(np.array(X_Kfold)[train_index], np.ravel(np.array(Y_Kfold)[train_index]))
    Y_Pred=clf.predict(np.array(X_Kfold)[test_index])
    
    TP_This = 0
    FP_This = 0
    Pos_This = 0
    Pos_Pred_This = 0
    
    i=0
    while i < len(Y_Pred):
        
        if Y_Pred[i] == 1:

            # count the positive predictions in test set in fold i
            # additionally add them to the total number of positive predictions of all folds
            P_Pred +=1
            Pos_Pred_This+=1
            if np.array(Y_Kfold)[test_index][i] == 1:
                
                # count true positives in fold i 
                # additionally add to the total true positives
                TP_This+=1
                TP +=1

            elif np.array(Y_Kfold)[test_index][i] == 0:
                
                    # count false positives from test set in this fold
                    FP_This+=1
        
        if np.array(Y_Kfold)[test_index][i] == 1:
            
            # count the values which belong to the positive class in fold i
            Pos_This +=1
        i+=1

    FP.append(FP_This)
    Pos.append(Pos_This)
    Pos_Pred.append(Pos_Pred_This)
    TP_list.append(TP_This)

#6.3 ____________________________________________________
# Report weighted Recall and Precision
# Recall = TP/P
# make some exceptions when dividing through 0    
if np.array(Y_Kfold)[train_index].sum() != 0:
    Recall = TP/np.array(Y_Kfold)[train_index].sum()
else:
    Recall ='nan'

if P_Pred != 0:
    Precision = TP/P_Pred   
else:
    Precision = 'nan'

print('\n_______________________________\nWeighted precision and recall, with shuffling: \n')
print('Precision: '+str(Precision)+'\n'+      'Recall:    '+str(Recall))
print('Positive events for each fold: '+str(Pos)+'\n'+      'True Positives for each fold:  '+str(TP_list))

