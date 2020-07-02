# Neural Network Model
# Vincent Weiss
# 14.06.2020

# Overview
#0 Imports
#1.0 Preprocessing
        #1.1 Preprocessing functions
        #1.2 Calling preprocessing functions and set some model parameters
        #1.3 Creating Time windows
#2.0 Evaluation
        #2.1 defining measures for evaluation
        #2.2 Miscellaneous
        #2.3 The K-fold NN loop
#3.0 Plotting performance measures for different epochs
        #3.1 Calculate average performance measures
#4.0 Plots & tables
        #4.1 A 3D plot, dimensions: sigmoid output(or classification), P/B, DivY
        #4.2 Graph of the colormap used in the paper
        #4.3 Figure 9 and Figure 10: 'Heatmaps' for model classification
        #4.4 Figure 11: Regimes over time
        #4.5 Table 6: ex post probabilities

#0 _____________________________________
# Imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from IPython.display import display
from keras import layers
from keras import models
from keras.utils.vis_utils import plot_model
from keras import backend as K
from keras import Input
from keras.models import Model
from keras.activations import softplus, tanh
from keras.layers import Dense, LSTM, Conv1D, Dropout, Flatten, concatenate, AveragePooling1D, MaxPooling1D
from keras.optimizers import SGD, Adam, RMSprop, Adamax
from keras.metrics import AUC, Precision, Recall, TruePositives, FalsePositives
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D 

# turn of some warnings not needed
pd.set_option('mode.chained_assignment', None)
# and make everything visible
pd.set_option("display.max_rows", None, "display.max_columns", None)

# https://www.machinecurve.com/index.php/2020/02/18/how-to-use-k-fold-cross-validation-with-keras/
from sklearn.model_selection import KFold

MSCI_Data=pd.read_csv("DATA_SVM_JV_Preprocessed.csv")
MSCI_Data.set_index('Unnamed: 0', inplace=True)
MSCI_Data.index.name = None

#1.0 ____________________________________________________
# Preprocessing: defining classification and selecting data for Input X and output Y

#1.1 ____________________________________________________
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

    # rewrite classifications: a period of low returns is classified with 1 (positive event) (previously -1)
    #                          a period of normal/high returns is classified with 0 (negative event) (previously 1)
    i=0
    while i < len(MSCI_Data_):
        if MSCI_Data_[str(Month)+' Month Class'].iloc[i]==-1:
            MSCI_Data_[str(Month)+' Month Class'].iloc[i]=1
        else:
            MSCI_Data_[str(Month)+' Month Class'].iloc[i]=0
        i+=1
    return MSCI_Data_

# not used in this file but can be used when manually setting validation set
def Train_Test_Data(MSCI_Data_, Month, share):
    Train_Length = int(round(len(MSCI_Data_)*share))
    Test_Length = int(len(MSCI_Data_)-Train_Length)

    X_Train = MSCI_Data_[['Price/Book Ratio', 'Dividend Yield']].iloc[0:Train_Length]
    Y_Train = MSCI_Data_[[str(Month)+' Month Class','negative']].iloc[0:Train_Length]
    X_Test = MSCI_Data_[['Price/Book Ratio', 'Dividend Yield']].iloc[Train_Length:]
    Y_Test = MSCI_Data_[[str(Month)+' Month Class','negative']].iloc[Train_Length:]
    #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    scaler = StandardScaler()
    scaler.fit(X_Train)
    X_Train=scaler.transform(X_Train)

    # Standardize the input data by removing mean and transform to unit variance
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    scaler = StandardScaler()
    scaler.fit(X_Test)
    X_Test=scaler.transform(X_Test)
    return (X_Train, X_Test, Y_Train, Y_Test)

def K_Fold_Train_Test_Data(MSCI_Data_, Month):
    
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

#1.2 ____________________________________________________
# Call preprocessing functions and define settings

# set some variables
# m can be either 1, 12, or 60
# time_length: - needs to be larger than 4 (due to kernel size of Conv1D layer) and should not be too big
#              - is a 'lookback' period, e.g. for 60 the model is fed 60 observations at once
#              - should not be too large (e.g. 60 means the first 59 observations from Dec 1974 are lost)
# num_epochs: number of epochs of the NN (8 to 10 recommended)
# num_folds: how many partitions are in the cross validation? (10 recommended)
# opt: which optimizer is used in the NN?
# CNN1_size and LSTM1_size: which output size do the CNN1 and LSTM1 layers have?

m=60
time_length=60
num_epochs = 10
num_folds = 10
opt=Adam(lr=0.01)
CNN1_size = 16
LSTM1_size = 20

# call functions from above
MSCI_here=load_MSCI(Month=m)
X_Kfold, Y_Kfold = K_Fold_Train_Test_Data( MSCI_here, Month=m)

#1.3 ____________________________________________________
# Creating Time windows

# The NN is fed time windows, several observations at once
# Time windows of time_length are created and stored to an array
# We loose some observations unfortunately (time_length = z  => z-1 observations lost)
X = []
for i in range(time_length, len(X_Kfold)+1):
    X.append(X_Kfold[i-time_length:i])
X=np.array(X)

# obviously, Y looses some observations as well!
Y = Y_Kfold[time_length-1:]

#2.0 ____________________________________________________
# Evaluation

#2.1 ____________________________________________________
# defining measures for evaluation

# Empty lists, which will be filled in each fold and epoch
# used for accuracy measures later on
acc = []
val_acc = []
precision = []
val_precision = []
recall = []
val_recall = []
auc = []
val_auc =[]
Recall_Regime_Weights=[]
TP_Regimes =[]

#2.2 ____________________________________________________
# Misceanellous

# Keep track on the fold numbers while in K-fold loop: use a counter
Fold_count = 1
# number of positive classifications
P=MSCI_here[MSCI_here[str(m)+' Month Class']==1][str(m)+' Month Class'].count()
# counter of positive predictions
P_Pred = 0
# kfold creates a list of splits that is used for different cross validation partitions
kfold = KFold(n_splits=num_folds, shuffle=False)

#2.3 ____________________________________________________
# The K-fold loop

# kfold.split generates indices to split data into training and test set
# for all resulting splits (all folds) the loop is repeated
#https://stackabuse.com/time-series-analysis-with-lstm-using-pythons-keras-library/
for train_index, test_index in kfold.split(X, Y):

    # keep track of fold numbers
    print('\n__________________________\n\n Fold Number: '+str(Fold_count)+'\n __________________________\n')
    
    # define inputs for training and validation
    X_features_set = X[train_index]
    Y_features_set = Y[train_index]
    x_features_set = X[test_index]
    y_features_set = Y[test_index]
    input_shape = (time_length, 2)
    network=None

    #Loop part 2 ____________________________________________________
    # Constructing the NN

    Input_object = Input(shape =(time_length, 2), name='Input', dtype='float32')

    # CNN branch
    CNN1 = Conv1D(CNN1_size, kernel_size=4, activation='relu',\
                  padding='same',name='CNN_branch_HL1', input_shape=input_shape)(Input_object)
    CNN_Pool = AveragePooling1D(pool_size=10, padding='same',name='CNN_branch_HL2')(CNN1)
    CNN_Flat = Flatten(name='CNN_branch_HL3')(CNN_Pool)
    
    # LSTM branch
    LSTM1 = LSTM(LSTM1_size, recurrent_dropout=0.1, return_sequences=False, activation='relu',\
                 name='LSTM_branch_HL1', input_shape=input_shape)(Input_object)
    
    # Concatenate CNN and LSTM so results can interpreted as one
    Concatenated = concatenate([CNN_Flat, LSTM1], name='Concatenation_Layer')
    
    # output layer
    Dense1 = Dense(1, activation='sigmoid', name='Output')(Concatenated)

    network = Model([Input_object], Dense1)

    #Loop part 3 ____________________________________________________
    # Compiling and fitting the network
    
    # define optimizer, loss function and some metrics to be reported
    network.compile(optimizer=opt,loss='binary_crossentropy',\
                    metrics=[Precision(name = 'precision'),\
                             Recall(thresholds=0.5, name='recall'),\
                             AUC(name='auc'),'accuracy',\
                             TruePositives(name = 'true_positives'),\
                             FalsePositives(name = 'false_positives')])
    
    history=network.fit(X_features_set, Y_features_set, epochs=num_epochs,\
                        validation_data=(x_features_set, y_features_set))

    #Loop part 4 ____________________________________________________
    # save the evaluation of the current fold
    
    # needed for next fold
    Fold_count+=1
    
    # Positive classifications in this fold's validation set, used for calculating recall weights
    P_This_Fold=Y[test_index].sum()
    
    # calculate the weight of this regime and store to list
    Recall_Current_Regime_Weight = P_This_Fold/P    
    Recall_Regime_Weights.append(Recall_Current_Regime_Weight)
    
    # store validation true positives for all epochs of this fold
    TP_Regimes.append(history.history['val_true_positives'])
    
    # update the positive predicted values of this fold:
    # add validation true positives and validation true positives
    # but add the mean of the last 3 epochs (i.e. when the model is trained already)
    # in the end, P_Pred is the total of positive predicted values (on average) of all folds
    P_Pred= P_Pred + np.array(history.history['val_true_positives'][-3:]).mean()\
            + np.array(history.history['val_false_positives'][-3:]).mean()
    
    # save some performance measures of all epochs in this fold
    acc.append(history.history['accuracy'])
    val_acc.append(history.history['val_accuracy'])
    precision.append(history.history['precision'])
    val_precision.append(history.history['val_precision'])
    recall.append(history.history['recall'])
    val_recall.append(history.history['val_recall'])
    auc.append(history.history['auc'])
    val_auc.append(history.history['val_auc'])

# figure 7
plot_model(network, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#3.0 ____________________________________________________
# Plotting performance measures for different epochs
# Total of 4 plots

# range needed for plotting
epochs = range(1, len(acc[0]) + 1)

# qualitative colorlist
colorlist = ['blue','cyan','purple','pink','orange','red','brown','gray','olive','green']

# Figure 8
plt.figure(figsize=(12,6))
# Keep in mind acc[i] is one fold, consisting of n Epochs (n = 10 usually)
# 'bo' -> dots for training set
# 'b'  -> line for validation set
i=0
while i <  num_folds:
    plt.plot(epochs, acc[i] , 'bo', label='Training acc', color = colorlist[i])
    plt.plot(epochs, val_acc[i], 'b', label='Validation acc', color=colorlist[i])
    ax= plt.gca()
    ax.tick_params(direction='in')
    i+=1
plt.title('Training and validation accuracy')

# as above
plt.figure(figsize=(12,6))
i=0
while i <  num_folds:
    plt.plot(epochs, precision[i] , 'bo', label='Precision', color = colorlist[i])
    plt.plot(epochs, val_precision[i], 'b', label='val_precision', color=colorlist[i])
    i+=1
plt.title('Training and validation precision')

# as above
plt.figure(figsize=(12,6))
i=0
while i <  num_folds:
    plt.plot(epochs, recall[i] , 'bo', label='Recall', color = colorlist[i])
    plt.plot(epochs, val_recall[i], 'b', label='val_Recall', color=colorlist[i])
    i+=1
plt.title('Training and validation recall')

# as above
plt.figure(figsize=(12,6))
i=0
while i <  num_folds:
    plt.plot(epochs, auc[i] , 'bo', label='auc', color = colorlist[i])
    plt.plot(epochs, val_auc[i], 'b', label='val_auc', color=colorlist[i])
    i+=1
plt.title('Training and validation auc')
plt.show()

#3.1 ____________________________________________________
# Calculate average performance measures
# Table 7

Recall_Regime_Weights = np.array(Recall_Regime_Weights)

# empty lists to be filled
val_TP_Regimes = []
val_acc_last = []
val_precision_last = []
val_recall_last = []
val_auc_last = []
precision_last =[]
accuracy_last =[]

# Last Epochs: how many of the last Epochs to include in the average result
# values should be smaller than num_epochs, and larger or equal than 1
LE = 3

# loop through all folds (val_precision is a list of list, first dimension is the number of folds)
i=0
while i < num_folds:
    
    # add the mean of the last n epochs; repeat for each fold
    val_TP_Regimes.append(np.array(TP_Regimes[i][-LE:]).mean())
    val_acc_last.append(np.array(val_acc[i][-LE:]).mean())
    val_precision_last.append(np.array(val_precision[i][-LE:]).mean())
    val_recall_last.append(np.array(val_recall[i][-LE:]).mean())
    val_auc_last.append(np.array(val_auc[i][-LE:]).mean())   
    precision_last.append(np.array(precision[i][-LE:]).mean())
    accuracy_last.append(np.array(acc[i][-LE:]).mean())
    i += 1

# dot product of val_precision last and the weights (True Positives of fold i / Total of Positive predictions)
weighted_Precision = np.dot(np.array(val_precision_last).transpose(), np.array(val_TP_Regimes)/P_Pred)
                            
# just for comparison with weighted Precision, not reported in paper
unweighted_precision_last = np.array(val_precision_last).mean()
                            
# no weight adjustment needed for accuracy, since folds are equally sized
val_acc_last = np.array(val_acc_last).mean()

# dot product of val_recall_last and the weights (Positives of fold i / Total of Positives)
weighted_Recall_last = np.dot(np.array(val_recall_last).transpose(), Recall_Regime_Weights)
unweighted_Recall_last = np.array(val_recall_last).mean()                  

val_auc_last = np.array(val_auc_last).mean()

# report train performance to check for overfitting
train_precision_last = np.dot(np.array(precision_last).transpose(), np.array(val_TP_Regimes)/P_Pred)
train_accuracy_last = np.array(accuracy_last).mean()

print('\n_______________________________\nCross Validation Results:\n')
print('Validation accuracy:            '+str(val_acc_last)+'\n'+\
      'Validation Weighted Precision:  '+ str(weighted_Precision)+'\n'+\
      'Validation Weighted Recall:     '+str(weighted_Recall_last)+'\n'+\
      'Validation AUC:                 '+str(val_auc_last)+'\n'+\
      'Training Accuracy               '+str(train_accuracy_last)+'\n'+\
      'Training Precision              '+str(train_precision_last))

#4.0 ____________________________________________________
# Plots & Tables

# use the last network settings to predict train and test
# last network settings -> last epoch on the last fold
Predict_Train=network.predict(X_features_set)
Predict_Test=network.predict(x_features_set)

#4.1 ____________________________________________________
# A 3D plot, dimensions: sigmoid output(or classification), P/B, DivY
# I want to map the non standardized P/B and DivY on model output

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
Y_graph=np.array(Y_Kfold[train_index])
y_graph=np.array(Y_Kfold[test_index])

# I want to map the non standardized P/B and DivY on model output
# Therefore reference the MSCI_here (with non standardized values)
# Remember the first time_length-1 observations are lost for the model, due to construction of time windows
# Index= 59 -> 60th observation
i=0+time_length-1
while i < len(train_index):
    ax.scatter(MSCI_here.iloc[i,2], MSCI_here.iloc[i,5],MSCI_here.iloc[i,3],\
               color='blue', s=10, alpha=1, marker=matplotlib.markers.MarkerStyle('s', fillstyle='bottom'))
    i+=1

# before: only train data, now test data, but still the true classification is used
# .iloc[:,x] -> x=2 P/B, x=3 DivY, x=5 Classification
i=0
while i < len(test_index):
    ax.scatter(MSCI_here.iloc[i-1+len(train_index)+time_length,2],\
               MSCI_here.iloc[i-1+len(train_index)+time_length,5],\
               MSCI_here.iloc[i-1+len(train_index)+time_length,3],\
               color='cyan', s=15,marker=matplotlib.markers.MarkerStyle('s', fillstyle='top'))
    i+=1   

# Pred Train
# as above
i=0+time_length-1
while i < len(train_index):
    ax.scatter(MSCI_here.iloc[i,2],                #in Predict_Train we want to start from index 0#
               Predict_Train[i-time_length+1,0], \
               MSCI_here.iloc[i,3], \
               color='green', s=20, marker=matplotlib.markers.MarkerStyle('s', fillstyle='right'))
    i+=1
    
# Pred Test
# as above
i=0
while i < len(test_index):
    ax.scatter(MSCI_here.iloc[i-1+len(train_index)+time_length,2], \
               Predict_Test[i,0],\
               MSCI_here.iloc[i-1+len(train_index)+time_length,3],\
               color='red', s=25, marker=matplotlib.markers.MarkerStyle('s', fillstyle='left'))
    i+=1


# add empty plot to create labels    
#https://stackoverflow.com/questions/20505105/add-a-legend-in-a-3d-scatterplot-with-scatter-in-matplotlib
scatter1_proxy = Line2D([0],[0], linestyle="none", c='blue', marker = 's')
scatter2_proxy = Line2D([0],[0], linestyle="none", c='cyan', marker = 's')
scatter3_proxy = Line2D([0],[0], linestyle="none", c='green', marker = 's')
scatter4_proxy = Line2D([0],[0], linestyle="none", c='red', marker = 's')
ax.legend([scatter1_proxy, scatter2_proxy,scatter3_proxy,scatter4_proxy],\
          ['True: Train', 'True: Test','Pred: Train', 'Pred: Test'], numpoints = 1)

ax.set_xlabel('P/B', fontsize=15)
ax.set_ylabel('Classification', fontsize=15)
ax.set_zlabel('DivY', fontsize=15)
plt.title('3D Classification Representation', fontsize=15, fontweight='bold')
plt.show()

#4.2 ____________________________________________________
# Graph of the colormap used in the paper

w = 2
h = 4
d = 70
plt.figure(figsize=(w, h), dpi=d)
color_map_list = [[]]

i = 0
while i <=1:
    color_map_list[0].append(i)
    i+=0.01

ax= plt.gca()
ax.axes.get_xaxis().set_visible(False)
color_map = plt.imshow(np.array(color_map_list).transpose(), extent = (0,0.1,0,1), interpolation='nearest')

#4.3 ____________________________________________________
# Figure 9 and Figure 10
#'Heatmaps' for model classification

cmap = cm.get_cmap('jet')

# used later to format y-axis
def format_functiony(y, pos):
    y_formatted = str(y/100)
    return y_formatted

plt.figure(figsize=(10,5))

# P/B and DivY from MSCI Dataset, color "value" from model prediction
#Pred Train
i=0+time_length-1
while i < len(train_index):
    plt.scatter(MSCI_here.iloc[i,2], MSCI_here.iloc[i,3], color=cmap(Predict_Train[i-time_length+1]-0.01), s=10)
    i+=1

# Pred Test
i=0
while i < len(test_index):
    plt.scatter(MSCI_here.iloc[i-1+len(train_index)+time_length,2], \
                MSCI_here.iloc[i-1+len(train_index)+time_length,3],\
                color=cmap(Predict_Test[i,0]-0.01), s=10)
    i+=1  
    
ax=plt.gca()
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_functiony))
ax.tick_params(direction='in')
plt.ylim(1, 6)
plt.xlim(0.6,4.5)
plt.title('NN, Predicted Classifications', fontsize=15, fontweight='bold')
plt.xlabel('P/B', fontsize=15)
plt.ylabel('divY', fontsize=15)
plt.show()

plt.figure(figsize=(10,5))

# P/B and DivY from MSCI Dataset, color "value" from model true classification
cmap = cm.get_cmap('jet')

#True Values
i=0+time_length
while i < len(MSCI_here):
    # had to substract some small value (-0.01) from color value because
    # at the upper bound the color shown was wrong
    plt.scatter(MSCI_here.iloc[i,2], MSCI_here.iloc[i,3], color=cmap(MSCI_here.iloc[i,5]-0.01), s=10)
    i+=1

ax=plt.gca()
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_functiony))
ax.tick_params(direction='in')
plt.ylim(1, 6)
plt.xlim(0.6,4.5)
plt.title('True Classification', fontsize=15, fontweight='bold')
plt.xlabel('P/B', fontsize=15)
plt.ylabel('divY', fontsize=15)
plt.show()

#4.4 ____________________________________________________
# Figure 11: Regimes over time

# Drop the exact time from Date column, only keep year and month
i = 0
while i<len(MSCI_here):
    MSCI_here['Date'].iloc[i]=MSCI_here['Date'].iloc[i][0:7]
    i+=1

# custom list of x_ticks, one tick every 60 months
list_xticks = []
i = 0
while i < (len(X)):
    list_xticks.append(MSCI_here['Date'].iloc[i+time_length][0:7])
    i += 60

# Posterior Probability regimes
# need sigmoid outputs first
Predict_X = network.predict(X)
plt.figure (figsize=(12,6))

i=0
Regimes = []
while i<len(Predict_X):  
    regime_count = 1
    # 1/10 < 0.15?  ->  2/10 < 0.15?  -> save 2 as Regime for this observation
    while (regime_count/10)<Predict_X[i]:
        regime_count+=1
    Regimes.append(regime_count)
    i+=1

plt.plot(MSCI_here.iloc[time_length-1:,0], np.array(Regimes))
plt.xticks(list_xticks)
plt.title('Posterior Probability Regimes Dec 1974 - Mar 2015 estimated with Sigmoid Output Layer',\
          fontsize=11, fontweight='bold')
plt.ylabel('Probability Regimes', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.locator_params(axis='y', nbins=10)
plt.xlim(0, 483-60)
ax= plt.gca()
ax.tick_params(direction='in')
plt.ylabel('Probability Regimes', fontsize=12)
plt.show()

# Same graph as above, additionally with true class for comparison
i = 0
while i<len(MSCI_here):
    MSCI_here['Date'].iloc[i]=MSCI_here['Date'].iloc[i][0:7]
    i+=1

list_xticks = []
i = 0
while i < (len(X)):
    list_xticks.append(MSCI_here['Date'].iloc[i+time_length][0:7])
    i += 60

# Posterior Probability regimes over time (not reported in paper, only for comparison)
Predict_X = network.predict(X)
plt.figure (figsize=(12,6))
i=0
Regimes = []
while i<len(Predict_X):
    regime_count = 1
    while (regime_count/10)<Predict_X[i]:
        regime_count+=1
    Regimes.append(regime_count)
    i+=1

plt.plot(MSCI_here.iloc[time_length-1:,0], np.array(Regimes))
plt.plot(MSCI_here.iloc[time_length-1:,0], MSCI_here.iloc[time_length-1:,5]*10)
plt.title('Posterior Probability Regimes (blue) and True classification (organge)',\
          fontsize=11, fontweight='bold')
plt.ylabel('Probability Regimes/ Classification', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.locator_params(axis='y', nbins=10)
plt.xlim(0, 483-60)
ax= plt.gca()
ax.tick_params(direction='in')
plt.xticks(list_xticks)
plt.show()

#4.5 ____________________________________________________
# Table 6: Ex Post Probabilities

MSCI_here['Posterior Regime']='Nan'
MSCI_here['Posterior Regime'].iloc[0+time_length-1:len(Regimes)+time_length-1]=Regimes

Regime_List=[]
Ex_Post_List=[]
Number_Total_Observations=[]
i=1
while i<11:
    # start by selecting regime one: iff posterior regime in the dataset is equal to the regime that
    # is currently looked for, then add this regime and the classification to the new dataframe
    Single_R=MSCI_here[MSCI_here['Posterior Regime']==i][['Posterior Regime','60 Month Class']]
    
    # further examine the Single_R dataframe
    # count True Positive values in Single_R and divide by the length
    # != 0 to avoid dividing by 0
    if len(Single_R) != 0:
        Single_Ex_Post=len(Single_R[Single_R['60 Month Class']==1])/len(Single_R)
    else:
        Single_Ex_Post = 'nan'
    
    # store the results for regime i
    Regime_List.append(i)
    Ex_Post_List.append(Single_Ex_Post)
    Number_Total_Observations.append(len(Single_R))
    i+=1
    
Table_2 = pd.DataFrame (columns = ['Posterior Probability Regimes','Ex-Post Probability',\
                                   'Number of Total Realizations'])
Table_2['Posterior Probability Regimes']=Regime_List
Table_2['Ex-Post Probability']=Ex_Post_List
Table_2['Number of Total Realizations']=Number_Total_Observations
print('\n_______________________________\nPosterior Probability Regimes and Ex-Post Probability:\n')
display(Table_2)