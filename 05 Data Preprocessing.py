# Data Preprocessing
# Vincent Weiss
# 14.06.2020

# Overview
#Part A: Preprocess + classification
#0 imports
#1 load, rename and drop unwanted line
#2 calculate returns, return quantiles and classify data, save to csv

#Part B: Dataset Statistics (Table 1)
#3 load, rename and drop unwanted line
#4 use dummy numbers to build dataframes, one for P/B and one for DivY
#5 output both dataframes

import pandas as pd
import numpy as np
from scipy.stats import jarque_bera
from IPython.display import display

# turn of some warnings not needed
pd.set_option('mode.chained_assignment', None)
# and make everything visible
pd.set_option("display.max_rows", None, "display.max_columns", None)



#Part A: Preprocess + classification
# 1 ____________________________________________________
# load, rename and drop unwanted line
raw = pd.read_excel("DATA_SVM_JV.xlsx", 'Sheet1') 
MSCI_Data=raw.rename(columns={"MSCI World ": "Date",\
                              "NDDUWI": "Total Return Index",\
                              'MSCI WORLD U$ - PRICE/BOOK RATIO': 'Price/Book Ratio',\
                              'MSCI WORLD U$ - DIVIDEND YIELD': 'Dividend Yield'})
MSCI_Data.drop(index=0, inplace=True)


# 2 ____________________________________________________
# calculate returns, return quantiles and classify data, save



# Calculate discrete index returns: when investing in period t, getting returns in t+1/t+12/t+60
# (index t+1 / index t)-1; (index t+12 / index t)-1; (index t+60 / index t)-1
# shift(-1) makes downward shift
MSCI_Data['1 Month Return']=\
    MSCI_Data['Total Return Index'].shift(-1)/MSCI_Data['Total Return Index']-1
MSCI_Data['12 Month Return']=\
    MSCI_Data['Total Return Index'].shift(-12)/MSCI_Data['Total Return Index']-1
MSCI_Data['60 Month Return']=\
    MSCI_Data['Total Return Index'].shift(-60)/MSCI_Data['Total Return Index']-1


# calculate quantiles for each investment horizon
Quantile_1Month=MSCI_Data['1 Month Return'].quantile(q=0.25)
Quantile_12Month=MSCI_Data['12 Month Return'].quantile(q=0.25)
Quantile_60Month=MSCI_Data['60 Month Return'].quantile(q=0.25)


# set the default classification as -1 (low return period)
MSCI_Data['1 Month Class']=-1
MSCI_Data['12 Month Class']=-1
MSCI_Data['60 Month Class']=-1


# if returns are higher than respective 25% quantiles
# -> change classification to 1 (normal or high return period)
# repeat for all investment horizons
i=0
while i < len(MSCI_Data):
    if MSCI_Data['1 Month Return'].iloc[i]>=Quantile_1Month:
        MSCI_Data['1 Month Class'].iloc[i]=1
    if MSCI_Data['12 Month Return'].iloc[i]>=Quantile_12Month:
        MSCI_Data['12 Month Class'].iloc[i]=1
    if MSCI_Data['60 Month Return'].iloc[i]>=Quantile_60Month:
        MSCI_Data['60 Month Class'].iloc[i]=1
    i += 1



# save as csv
MSCI_Data.to_csv('DATA_SVM_JV_Preprocessed.csv', index=True)



#Part B: Dataset Statistics (Table 1)
# 3 ____________________________________________________
# load, rename and drop unwanted line
raw = pd.read_excel("DATA_SVM_JV.xlsx", 'Sheet1')
MSCI_Data=raw.rename(columns={"MSCI World ": "Date",\
                              "NDDUWI": "Total Return Index",\
                              'MSCI WORLD U$ - PRICE/BOOK RATIO': 'Price/Book Ratio',\
                              'MSCI WORLD U$ - DIVIDEND YIELD': 'Dividend Yield'})
MSCI_Data.drop(index=0, inplace=True)



# 4 ____________________________________________________
# use dummy numbers to build dataframes, one for P/B and one for DivY
dummy = np.zeros((1,8))
MSCI_Overview_1 = pd.DataFrame (dummy, columns = \
    ['MIN','MEDIAN', 'MAX','MEAN','STD','SKEW','KURT','JBTest'])
MSCI_Overview_2 = pd.DataFrame (dummy, columns = \
    ['MIN','MEDIAN', 'MAX','MEAN','STD','SKEW','KURT','JBTest'])



# Jarque Bera test at significance of 0.05
singnificance = 0.05
if jarque_bera(MSCI_Data['Price/Book Ratio'])[1]<singnificance:
    JB_PB = 'rej. H0'
else:
    JB_PB = 'not rej. H0'
    
if jarque_bera(MSCI_Data['Dividend Yield'])[1]<singnificance:
    JB_divY = 'rej. H0'
else:
    JB_divY = 'not rej. H0'



# store statistics in P/B dataframe
MSCI_Overview_1['MIN']=np.round(MSCI_Data['Price/Book Ratio'].min(), 3)
MSCI_Overview_1['MEDIAN']=np.round(MSCI_Data['Price/Book Ratio'].median(), 3)
MSCI_Overview_1['MAX']=np.round(MSCI_Data['Price/Book Ratio'].max(), 3)
MSCI_Overview_1['MEAN']=np.round(MSCI_Data['Price/Book Ratio'].mean(), 3)
MSCI_Overview_1['STD']=np.round(MSCI_Data['Price/Book Ratio'].std(), 3)
MSCI_Overview_1['SKEW']=np.round(MSCI_Data['Price/Book Ratio'].skew(), 3)
MSCI_Overview_1['KURT']=np.round(MSCI_Data['Price/Book Ratio'].kurtosis()+3, 3)
MSCI_Overview_1['JBTest']=JB_PB


# store statistics in DivY dataframe
MSCI_Overview_2['MIN']=str(np.round(MSCI_Data['Dividend Yield'].min(), 3))+'%'
MSCI_Overview_2['MEDIAN']=str(np.round(MSCI_Data['Dividend Yield'].median(), 3))+'%'
MSCI_Overview_2['MAX']=str(np.round(MSCI_Data['Dividend Yield'].max(), 3))+'%'
MSCI_Overview_2['MEAN']=str(np.round(MSCI_Data['Dividend Yield'].mean(), 3))+'%'
MSCI_Overview_2['STD']=str(np.round(MSCI_Data['Dividend Yield'].std(), 3))+'%'
MSCI_Overview_2['SKEW']=np.round(MSCI_Data['Dividend Yield'].skew(), 3)
MSCI_Overview_2['KURT']=np.round(MSCI_Data['Dividend Yield'].kurtosis()+3, 3)
MSCI_Overview_2['JBTest']=JB_divY


# 5 ____________________________________________________
# output both dataframes
print('_____________________________________\nKey Statistics MSCI Dataset:')
print('Price to Book:\n'+str(MSCI_Overview_1)+'\n\nDividend Yield:\n'+str(MSCI_Overview_2))

