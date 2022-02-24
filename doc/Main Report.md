<h1><center>Causal Inference Algorithms Evaluation Report</center></h1>
<h2><center>Group 3: Catherine Gao, Eve Washington, Siyuan Sang, Zi Fang</center></h2>
<h3><center>April 7, 2021</center></h3>

## Project Overview

In this project, group 3 evaluates three causal inference algorithms to compute the average treatment effect (ATE) on two distict datasets and compare their computational efficiency and performance.

One dataset contains high dimensional data and another contains low dimensional data. We will use L1 penalized logistic regression to estimate the propensity scores for these two datasets, and apply the following three methods to calcualte ATE for each dataset:

| **Algorithm** | **Propensity Score Estimation** | 
|:-------------|:-------:|
| Propensity Scores Matching (full)    | L1 penalized logistic regression|
| Doubly Robust Estimations  | L1 penalized logistic regression| 
| Stratification | L1 penalized logistic regression| 

This report includes a description of each algorithm, code to reproduce the results, and a comparison of the models. 

## 2 Data Preparation
### 2.1 Load Required Packages

In both datasets, variable "Y" indicates the outcome variable and variable "A" indicates the treatment group assighnment. The remaining variables are covariates for consideration. 


```python
import numpy as np
import pandas as pd
import time
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# setting graph styles
sns.set(rc={'figure.figsize':(10,8)})
sns.set_theme(style='ticks')

# set seed
random_state = 2021
```

### 2.2 Load Data


```python
# load high dimensional data
highdim_data = pd.read_csv('../data/highDim_dataset.csv')

# load low dimensional data
lowdim_data = pd.read_csv('../data/lowDim_dataset.csv')
```


```python
print("The high dimensional data has",highdim_data.shape[0],"observations and", highdim_data.shape[1], "variables.")
print("The low dimensional data has",lowdim_data.shape[0],"observations and", lowdim_data.shape[1], "variables.")
```

    The high dimensional data has 2000 observations and 187 variables.
    The low dimensional data has 500 observations and 24 variables.


We use violin plots to visualize outcome values by treatment groups in both dataset to better understand the data. For the high dimensional data, we see that the outcome values of non-treated group mainly cluster between 0 to 50, but those of treated group below 0 and even have extreme values. For the low dimensional data, the outcome values are more evenly distributed between the treatment groups. 


```python
# visualize outcome values on high dimensional data by treatment group
sns.violinplot(x="A", y="Y", data=highdim_data)
plt.title('High Dimensional Data - Outcome values by Treatment Group', size=20)
plt.show()
```


    
![png](output_8_0.png)
    



```python
# visualize outcome values on low dimensional data by treatment group
sns.violinplot(x="A", y="Y", data=lowdim_data)
plt.title('Low Dimensional Data - Outcome values by Treatment Group', size=20)
plt.show()
```


    
![png](output_9_0.png)
    


### 2.3 Scale data for regression model
We will scale the features in the original dataset and combine with the outcome variables to create scaled datasets for regression models. Feature scaling is crucial because it helps to normalize the range of all features for distance calculation. 


```python
# function to scale the datasets
def scaled_data(data):
    x = data.drop(['A','Y'], axis = 1)
    y = data[["A"]]
    
    data_columns = data.columns.drop(['Y','A'])
    
    x_scaled = StandardScaler().fit_transform(x)
    
    data_scaled = pd.DataFrame(x_scaled, index = data.index, columns = data_columns)
    
    data_scaled['A'] = data['A']
    data_scaled['Y'] = data['Y']
    
    display(data_scaled.head())
    
    return data_scaled
```


```python
# scale the high dimentional dataset
highdim_scale_data = scaled_data(highdim_data)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>...</th>
      <th>V178</th>
      <th>V179</th>
      <th>V180</th>
      <th>V181</th>
      <th>V182</th>
      <th>V183</th>
      <th>V184</th>
      <th>V185</th>
      <th>A</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.015114</td>
      <td>0.482748</td>
      <td>-1.161393</td>
      <td>0.303352</td>
      <td>1.487812</td>
      <td>-1.171070</td>
      <td>-1.423520</td>
      <td>1.686961</td>
      <td>1.203321</td>
      <td>0.382352</td>
      <td>...</td>
      <td>-0.140538</td>
      <td>-0.061157</td>
      <td>-0.096027</td>
      <td>-0.051378</td>
      <td>-0.078727</td>
      <td>-0.092975</td>
      <td>-0.065612</td>
      <td>-0.053082</td>
      <td>0</td>
      <td>41.224513</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.015114</td>
      <td>-2.071474</td>
      <td>-1.650640</td>
      <td>-1.477143</td>
      <td>-0.512424</td>
      <td>-1.171070</td>
      <td>0.204290</td>
      <td>0.524392</td>
      <td>1.203321</td>
      <td>0.801943</td>
      <td>...</td>
      <td>-0.208702</td>
      <td>-0.099379</td>
      <td>-0.208780</td>
      <td>-0.051378</td>
      <td>-0.078727</td>
      <td>-0.092975</td>
      <td>-0.065612</td>
      <td>-0.053082</td>
      <td>0</td>
      <td>40.513875</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.015114</td>
      <td>-2.071474</td>
      <td>0.795598</td>
      <td>-1.922267</td>
      <td>-0.876103</td>
      <td>-0.415004</td>
      <td>-0.880917</td>
      <td>0.669713</td>
      <td>1.203321</td>
      <td>-0.037239</td>
      <td>...</td>
      <td>-0.447277</td>
      <td>-0.443385</td>
      <td>-0.434284</td>
      <td>-0.051378</td>
      <td>-0.078727</td>
      <td>-0.092975</td>
      <td>-0.065612</td>
      <td>-0.053082</td>
      <td>0</td>
      <td>38.495476</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.985111</td>
      <td>-2.071474</td>
      <td>-1.324475</td>
      <td>-1.477143</td>
      <td>-1.239782</td>
      <td>-1.171070</td>
      <td>-0.700049</td>
      <td>0.698777</td>
      <td>-0.831034</td>
      <td>-0.456829</td>
      <td>...</td>
      <td>-0.447277</td>
      <td>-0.443385</td>
      <td>-0.434284</td>
      <td>-0.051378</td>
      <td>-0.078727</td>
      <td>-0.092975</td>
      <td>-0.065612</td>
      <td>-0.053082</td>
      <td>0</td>
      <td>33.001889</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.985111</td>
      <td>0.482748</td>
      <td>-0.019815</td>
      <td>0.971038</td>
      <td>0.214934</td>
      <td>0.492274</td>
      <td>2.012968</td>
      <td>0.756906</td>
      <td>1.203321</td>
      <td>0.382352</td>
      <td>...</td>
      <td>-0.174620</td>
      <td>-0.137602</td>
      <td>-0.133612</td>
      <td>1.226231</td>
      <td>1.319316</td>
      <td>1.105593</td>
      <td>1.180743</td>
      <td>1.396247</td>
      <td>0</td>
      <td>37.043603</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 187 columns</p>
</div>



```python
# scale the low dimentional dataset
lowdim_scale_data = scaled_data(lowdim_data)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>...</th>
      <th>V15</th>
      <th>V16</th>
      <th>V17</th>
      <th>V18</th>
      <th>V19</th>
      <th>V20</th>
      <th>V21</th>
      <th>V22</th>
      <th>A</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.502205</td>
      <td>-0.352816</td>
      <td>-0.257883</td>
      <td>-0.266592</td>
      <td>-0.34195</td>
      <td>-0.465776</td>
      <td>-0.266412</td>
      <td>-0.649809</td>
      <td>-0.35092</td>
      <td>-0.159096</td>
      <td>...</td>
      <td>-0.868574</td>
      <td>-0.181992</td>
      <td>-0.739177</td>
      <td>-0.107309</td>
      <td>-0.285789</td>
      <td>-0.217729</td>
      <td>5.769470</td>
      <td>0.184432</td>
      <td>0</td>
      <td>30.486999</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.502205</td>
      <td>-0.352816</td>
      <td>-0.257883</td>
      <td>-0.266592</td>
      <td>-0.34195</td>
      <td>-0.465776</td>
      <td>-0.266412</td>
      <td>1.034631</td>
      <td>-0.35092</td>
      <td>-0.159096</td>
      <td>...</td>
      <td>-0.143710</td>
      <td>-0.181992</td>
      <td>0.443696</td>
      <td>-0.107309</td>
      <td>-0.285789</td>
      <td>-0.217729</td>
      <td>-0.335214</td>
      <td>2.374539</td>
      <td>0</td>
      <td>18.208417</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.502205</td>
      <td>-0.352816</td>
      <td>-0.257883</td>
      <td>-0.266592</td>
      <td>-0.34195</td>
      <td>-0.465776</td>
      <td>-0.266412</td>
      <td>-0.649809</td>
      <td>-0.35092</td>
      <td>-0.159096</td>
      <td>...</td>
      <td>0.979830</td>
      <td>-0.181992</td>
      <td>-0.739177</td>
      <td>-0.107309</td>
      <td>-0.285789</td>
      <td>-0.217729</td>
      <td>-0.335214</td>
      <td>-1.264176</td>
      <td>0</td>
      <td>13.485040</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.441468</td>
      <td>-0.352816</td>
      <td>-0.257883</td>
      <td>-0.266592</td>
      <td>-0.34195</td>
      <td>-0.465776</td>
      <td>-0.266412</td>
      <td>-0.649809</td>
      <td>-0.35092</td>
      <td>-0.159096</td>
      <td>...</td>
      <td>0.363695</td>
      <td>-0.181992</td>
      <td>1.271707</td>
      <td>-0.107309</td>
      <td>-0.285789</td>
      <td>-0.217729</td>
      <td>-0.335214</td>
      <td>-0.753260</td>
      <td>1</td>
      <td>25.699678</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.253654</td>
      <td>0.209949</td>
      <td>-0.150877</td>
      <td>-0.081459</td>
      <td>-0.34195</td>
      <td>0.445723</td>
      <td>0.162041</td>
      <td>0.493204</td>
      <td>1.15826</td>
      <td>-0.071411</td>
      <td>...</td>
      <td>0.767548</td>
      <td>-0.181992</td>
      <td>0.595779</td>
      <td>-0.107309</td>
      <td>1.061785</td>
      <td>0.282464</td>
      <td>-0.335214</td>
      <td>0.719985</td>
      <td>0</td>
      <td>23.752968</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>


## 3 Propensity Scores Estimation and Evaluation
Propensity score is the probability of assignment to the treatment group based on observed characteristics. It reduces each sample's set of covariates into a single score. 

We use L1 penalized logistic regression to estimate propensity scores for both data sets. To ensure more accurate results, we first tune the optimal hyperparameters for logistic regression, then use the best parameter to estimate propensity scores.

We repeat the above steps for both the high dimensional and low dimensional datasets.

### 3.1 Create Propensity Score Estimation Functions


```python
def best_param(data, random_state, param_grid, cv=10):
    '''
    Purpose: to find the best parameter "C" (coefficient of regularization strength) for the specific dataset
    
    Parameters:
    data - dataset to best tested on 
    random_state - set seed
    param_grid - set of parameter values to test on
    cv - number of folds for cross-validation
    
    '''

    x = data.drop(['A','Y'], axis = 1)  
    y = data[['A']].values.ravel()
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=random_state)
    
    
    model_cv = GridSearchCV(LogisticRegression(penalty='l1',solver = 'liblinear'), param_grid, cv=cv)
    model_cv.fit(x_train, y_train)
    
    print("The best tuned coefficient of regularization strength is",model_cv.best_params_.get('C'), 
          "with a testing accuracy of", model_cv.score(x_test, y_test))
    
    return model_cv.best_params_.get('C')

```


```python
def propensity_score(data, C=0.1, plot = True):
    '''
    Purpose: to estimate propensity score with L1 penalized logistic regression
    
    Parameters:
    data - dataset to estimate on 
    C - coeficient of regularization strength
    plot - print out visualization to show distribution of propensity scores
    
    Returns:
    1. ps for Propensity Score
    2. Visualization plot to show distribution of propensity scores
    
    '''
    
    T = 'A'
    Y = 'Y'
    X = data.columns.drop([T,Y])
    
    ps_model = LogisticRegression(random_state=random_state, penalty='l1',
                                  solver='liblinear').fit(data[X], data[T]) 
    
    ps = ps_model.predict_proba(data[X])[:,1] # we are interested in the probability of getting a "1"
    
    if plot:
        df_plot = pd.DataFrame({'Treatment':data[T], 'Propensity Score':ps})
        
        sns.histplot(data=df_plot, x = "Propensity Score", hue = "Treatment", element = "step")
        plt.title("Distribution of Propensity Score by Treatment Group", size=20)
        plt.show()
   
    return ps

```


```python
# setting parameters
param_grid = {"C":[0.01,0.05,0.1,0.3,0.5,0.7,1]}
```

### 3.2 Evaluate Propensity Scores for High Dimensional Data


```python
# use 10-fold cross-validation to tune for the best parameter for logistic regression
c_high = best_param(highdim_scale_data, random_state=random_state, param_grid=param_grid)
```

    The best tuned coefficient of regularization strength is 0.05 with a testing accuracy of 0.716



```python
# estimate propsensity scores
ps_high = propensity_score(highdim_scale_data, C = c_high)
```


    
![png](output_21_0.png)
    


### 3.3 Evaluate Propensity Scores for Low Dimensional Data


```python
# use 10-fold cross-validation to tune for the best parameter for logistic regression
c_low = best_param(lowdim_scale_data, random_state=random_state, param_grid=param_grid)
```

    The best tuned coefficient of regularization strength is 0.3 with a testing accuracy of 0.792



```python
ps_low = propensity_score(lowdim_scale_data, C = c_low)
```


    
![png](output_24_0.png)
    


The estimated propensity scores for both the high and low dimensional datasets are skewed to the right for the non-treatment group and spread out for the treatment group. This will create some challenges for propensity score matching.  

## 4. Algorithms and Evaluation




```python
# set true ATE
true_low = 2.0901
true_high = -54.8558
```

### 4.1 Propensity Scores Matching (Full)
Full matching creates a series of matched sets in an optimal way so that each matched set contains at least one treated individual and at least one control individual. 


```python
# append propensity score to original data
highdim_data['PS']=ps_high
lowdim_data['PS']=ps_low
```


```python
# create function to calculate PSM
def PSM(treated_df, control_df):
    
    #Get Distances
    treated_df.loc[:,'group']=None
    treated_df.loc[:,'control_Y']=None
    treated_df.loc[:,'D']=None
    for i in range(len(treated_df)):
        temp_d=[]
        for j in range(len(control_df)):
            temp_d.append(abs(treated_df.loc[i,'PS']-control_df.loc[j,'PS']))
        index=temp_d.index(np.min(temp_d))
        treated_df.loc[i,'control_Y']=control_df.loc[index,'Y']
        treated_df.loc[i,'D']=np.min(temp_d)
        
    #Split class
    #Here, we deide to create 5 subclasses
    r=(max(treated_df.loc[:,'D'])-min(treated_df.loc[:,'D']))/5
    for i in range(len(treated_df)):
        if treated_df.loc[i,'D'] <= min(treated_df.loc[:,'D'])+r:
            treated_df.loc[i,'group']=0
        elif treated_df.loc[i,'D'] > min(treated_df.loc[:,'D'])+r and treated_df.loc[i,'D'] <= min(treated_df.loc[:,'D'])+2*r:
            treated_df.loc[i,'group']=1
        elif treated_df.loc[i,'D'] > min(treated_df.loc[:,'D'])+2*r and treated_df.loc[i,'D'] <= min(treated_df.loc[:,'D'])+3*r:
            treated_df.loc[i,'group']=2
        elif treated_df.loc[i,'D'] > min(treated_df.loc[:,'D'])+3*r and treated_df.loc[i,'D'] <= min(treated_df.loc[:,'D'])+4*r:
            treated_df.loc[i,'group']=3
        else:
            treated_df.loc[i,'group']=4
            
    #Calculate ATE
    TE=[]
    for i in range(5):
        temp=treated_df[treated_df.loc[:,'group']==i]
        a=(temp.loc[:,'Y']-temp.loc[:,'control_Y']).mean()*len(temp)/len(treated_df)    
        TE.append(a)
    ATE=np.nanmean(TE)
    
    return ATE
```

#### Low dimensional data


```python
PS_low_start = time.time()
treated_low=lowdim_data[lowdim_data.A==1]
treated_low.reset_index(drop=True, inplace=True)
control_low=lowdim_data[lowdim_data.A==0]
control_low.reset_index(drop=True, inplace=True)
```


```python
PS_low_ATE = PSM(treated_low, control_low)
PS_low_end = time.time()
PS_low_accu = (1 - (abs(PS_low_ATE - true_low)/abs(true_low)))*100
PS_low_time = PS_low_end - PS_low_start
```


```python
# display results 
PS_low_result = pd.Series(data = ['PSM', 'Low', PS_low_time, PS_low_ATE, PS_low_accu], 
          index = ['Method','Data Type','Run Time','ATE','Accuracy'])

print(f'PSM method for low dimensional dataset:\n ATE = {PS_low_ATE:0.2f}\n Accuracy = {PS_low_accu:0.2f}\n PSM running time = {PS_low_time:0.2f}')

```

    PSM method for low dimensional dataset:
     ATE = 0.36
     Accuracy = 17.43
     PSM running time = 1.70


#### High dimensional data


```python
PS_high_start = time.time()
treated_high=highdim_data[highdim_data.A==1]
treated_high.reset_index(drop=True, inplace=True)
control_high=highdim_data[highdim_data.A==0]
control_high.reset_index(drop=True, inplace=True)
```


```python
PS_high_ATE = PSM(treated_high, control_high)
PS_high_end = time.time()
PS_high_accu = (1 - (abs(PS_high_ATE - true_high)/abs(true_high)))*100
PS_high_time = PS_high_end - PS_high_start
```


```python
# display results
PS_high_result = pd.Series(data = ['PSM', 'High', PS_high_time, PS_high_ATE, PS_high_accu], 
          index = ['Method','Data Type','Run Time','ATE', 'Accuracy'])

print(f'PSM method for high dimensional dataset:\n ATE = {PS_high_ATE:0.2f}\n Accuracy = {PS_high_accu:0.2f}\n PSM running time = {PS_high_time:0.2f}')
```

    PSM method for high dimensional dataset:
     ATE = -11.71
     Accuracy = 21.35
     PSM running time = 11.81


### 4.2 Doubly Robust Estimations
Doubly Robust estimation combines outcome regression model with weighting by propensity score model. Doubly robust estimation remains consistent even if either the outcome model or the propensity model is incorrect. 

#### Low Dimensional Data


```python
# reload data, add propensity score column and divide data into treat and control groups
lowdim_data = pd.read_csv('../data/lowDim_dataset.csv')
lowdim_data_new = pd.read_csv('../data/lowDim_dataset.csv')
lowdim_data_new['PS_low'] = pd.Series(ps_low, index=lowdim_data_new.index)
lowdim_treat = lowdim_data[lowdim_data['A'] == 1].reset_index(drop = True)
lowdim_control = lowdim_data[lowdim_data['A'] == 0].reset_index(drop = True)
```


```python
# fit regression model to treat and control group
xlow_treat = lowdim_treat.drop(['A','Y'],axis=1)
ylow_treat = lowdim_treat['Y']
lr_low_treat = LinearRegression().fit(xlow_treat, ylow_treat)

xlow_control = lowdim_control.drop(['A','Y'],axis=1)
ylow_control = lowdim_control['Y']
lr_low_control = LinearRegression().fit(xlow_control, ylow_control)
```


```python
# make prediction based on trained models and construct a full dataset 
xlow = lowdim_data_new.drop(['A','Y','PS_low'],axis=1)
lowdim_data_new['mtreat'] = lr_low_treat.predict(xlow)
lowdim_data_new['mcontrol'] = lr_low_control.predict(xlow)
```


```python
# perform Doubly Robust Estimation algorithm
DR_low_start = time.time()

DR_low_1 = 0
DR_low_0 = 0
    
for i in range(len(lowdim_data_new)):
    DR_low_1 = DR_low_1 + (lowdim_data_new['A'][i] * lowdim_data_new['Y'][i] - (lowdim_data_new['A'][i] - lowdim_data_new['PS_low'][i])*lowdim_data_new['mtreat'][i])/lowdim_data_new['PS_low'][i]
    DR_low_0 = DR_low_0 + ((1-lowdim_data_new['A'][i])* lowdim_data_new['Y'][i] + (lowdim_data_new['A'][i] - lowdim_data_new['PS_low'][i])*lowdim_data_new['mcontrol'][i])/(1-lowdim_data_new['PS_low'][i])
        
DR_low_ATE = (DR_low_1 - DR_low_0)/len(lowdim_data_new)
DR_low_accu = (1 - abs((DR_low_ATE -true_low)/true_low))*100
DR_low_end = time.time()
DR_low_time = DR_low_end - DR_low_start
```


```python
# print the ATE, accuracy and algorithm running time result
DR_low_result = pd.Series(data = ['Doubly Robust', 'Low', DR_low_time, DR_low_ATE, DR_low_accu], 
          index = ['Method','Data Type','Run Time','ATE', 'Accuracy'])

print(f'Doubly robust estimation method for low dimensional dataset:\n ATE = {DR_low_ATE:0.2f}\n Accuracy = {DR_low_accu:0.2f}\n DR running time = {DR_low_time:0.2f}')
```

    Doubly robust estimation method for low dimensional dataset:
     ATE = 2.09
     Accuracy = 99.76
     DR running time = 0.11


#### High dimensional data


```python
# reload data, add propensity score column and divide data into treat and control groups
highdim_data = pd.read_csv('../data/highDim_dataset.csv')
highdim_data_new = pd.read_csv('../data/highDim_dataset.csv')
highdim_data_new['PS_high'] = pd.Series(ps_high, index=highdim_data.index)
highdim_treat = highdim_data[highdim_data.A == 1].reset_index(drop = True)
highdim_control = highdim_data[highdim_data.A == 0].reset_index(drop = True)
```


```python
# fit regression model to treat and control group
xhigh_treat = highdim_treat.drop(['A','Y'],axis=1)
yhigh_treat = highdim_treat['Y']
lr_high_treat = LinearRegression().fit(xhigh_treat, yhigh_treat)

xhigh_control = highdim_control.drop(['A','Y'],axis=1)
yhigh_control = highdim_control['Y']
lr_high_control = LinearRegression().fit(xhigh_control, yhigh_control)
```


```python
# make prediction based on trained models and construct a full dataset 
xhigh = highdim_data_new.drop(['A','Y','PS_high'],axis=1)
highdim_data_new['mtreat'] = lr_high_treat.predict(xhigh)
highdim_data_new['mcontrol'] = lr_high_control.predict(xhigh)
```


```python
# perform Doubly Robust Estimation algorithm
DR_high_start = time.time()

DR_high_1 = 0
DR_high_0 = 0
    
for i in range(len(highdim_data_new)):
    DR_high_1 = DR_high_1 + (highdim_data_new['A'][i] * highdim_data_new['Y'][i] - (highdim_data_new['A'][i]- highdim_data_new['PS_high'][i])*highdim_data_new['mtreat'][i])/highdim_data_new['PS_high'][i]
    DR_high_0 = DR_high_0 + ((1-highdim_data_new['A'][i])* highdim_data_new['Y'][i] + (highdim_data_new['A'][i] - highdim_data_new['PS_high'][i])*highdim_data_new['mcontrol'][i])/(1-highdim_data_new['PS_high'][i])

DR_high_ATE = (DR_high_1 - DR_high_0)/len(highdim_data_new)
DR_high_accu = (1 - abs((DR_high_ATE - true_high)/true_high))*100
DR_high_end = time.time()
DR_high_time = DR_high_end - DR_high_start
```


```python
# save results and print the ATE, accuracy and algorithm running time result

DR_high_result = pd.Series(data = ['Doubly Robust', 'High', DR_high_time, DR_high_ATE, DR_high_accu], 
          index = ['Method','Data Type','Run Time','ATE', 'Accuracy'])

print(f'Doubly robust estimation method for high dimensional dataset:\n ATE = {DR_high_ATE:0.2f}\n Accuracy = {DR_high_accu:0.2f}\n DR running time = {DR_high_time:0.2f}')
```

    Doubly robust estimation method for high dimensional dataset:
     ATE = -57.04
     Accuracy = 96.02
     DR running time = 0.23


### 4.3 Stratification
We will rank and stratify 5 mutually exclusive subsets based on the propensity scores. Within each stratum, subjects have roughly similar values of the propensity scores.


```python
## Stratification

def stratification(data, prop):
    start = time.time()
    K = 5 # k, quintiles is reccomended
   # N = len(df.index)
    strata = [1,2,3,4,5]
    ATE = 0

    #split propensity scores into thier respective quintiles 
    prop_split = pd.qcut(prop, K)
    prop_split.categories = strata 
    
    
    #label the dataset and group accordingly
    quintiles = copy.copy(data)
    quintiles["strata"] = prop_split
    quintiles = quintiles[["A", "strata", "Y"]]

    #calucate the average Y
    quintiles = quintiles.groupby(["A", "strata"]).mean()

    for num in strata:
        ATE += quintiles.loc[pd.IndexSlice[(1, num)]] - quintiles.loc[pd.IndexSlice[(0, num)]] 
    
    #Divide by k
    ATE = ATE/K

    end = time.time()

    print("Estamated ATE: ", round(ATE.values[0], 2))
    print("Runtime: ", end-start)

    return(ATE, end-start)
```

#### Low Dimensional Data


```python
print("low")

S_low = stratification(lowdim_data, ps_low)

S_low_ATE = S_low[0]
S_low_ATE = S_low_ATE.values[0]
S_low_time = S_low[1]
S_low_accu = (1 - (abs(S_low_ATE - true_low)/abs(true_low)))*100
```

    low
    Estamated ATE:  2.38
    Runtime:  0.03311920166015625



```python
# save results and print the ATE, accuracy and algorithm running time result

S_low_result = pd.Series(data = ['Stratification', 'Low', S_low_time, S_low_ATE, S_low_accu], 
          index = ['Method','Data Type','Run Time','ATE', 'Accuracy'])

print(f'Stratification estimation method for low dimensional dataset:\n ATE = {S_low_ATE:0.2f}\n Accuracy = {S_low_accu:0.2f}\n DR running time = {S_low_time:0.2f}')
```

    Stratification estimation method for low dimensional dataset:
     ATE = 2.38
     Accuracy = 85.97
     DR running time = 0.03


#### High Dimensional Data


```python
print("high")

S_high = stratification(highdim_data, ps_high)

S_high_ATE = S_high[0]
S_high_ATE = S_high_ATE.values[0]
S_high_time = S_high[1]
S_high_accu = (1 - (abs(S_high_ATE - true_high)/abs(true_high)))*100
```

    high
    Estamated ATE:  -59.83
    Runtime:  0.033457040786743164



```python
# save results and print the ATE, accuracy and algorithm running time result

S_high_result = pd.Series(data = ['Stratification', 'High', S_high_time, S_high_ATE, S_high_accu], 
          index = ['Method','Data Type','Run Time','ATE', 'Accuracy'])

print(f'Stratification estimation method for high dimensional dataset:\n ATE = {S_high_ATE:0.2f}\n Accuracy = {S_high_accu:0.2f}\n DR running time = {S_high_time:0.2f}')

```

    Stratification estimation method for high dimensional dataset:
     ATE = -59.83
     Accuracy = 90.94
     DR running time = 0.03


## 5. Model Comparison and Conclusion

Propensity scores have been estimated using L1 penalized logistic regression. For algorithm testing, use ps_low for the low dimensional data and ps_high for the high dimensional data. 


```python
# store all final results into dataframe
result_table = pd.DataFrame([PS_low_result, DR_low_result, S_low_result, PS_high_result, DR_high_result, S_high_result])
result_table = result_table.round(2)
```


```python
# display results
result_table
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Method</th>
      <th>Data Type</th>
      <th>Run Time</th>
      <th>ATE</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PSM</td>
      <td>Low</td>
      <td>1.70</td>
      <td>0.36</td>
      <td>17.43</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Doubly Robust</td>
      <td>Low</td>
      <td>0.11</td>
      <td>2.09</td>
      <td>99.76</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Stratification</td>
      <td>Low</td>
      <td>0.03</td>
      <td>2.38</td>
      <td>85.97</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PSM</td>
      <td>High</td>
      <td>11.81</td>
      <td>-11.71</td>
      <td>21.35</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Doubly Robust</td>
      <td>High</td>
      <td>0.23</td>
      <td>-57.04</td>
      <td>96.02</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Stratification</td>
      <td>High</td>
      <td>0.03</td>
      <td>-59.83</td>
      <td>90.94</td>
    </tr>
  </tbody>
</table>
</div>



In summary, Propensity Score Matching has the longest run time and lowest accuracy of all three methods. This is due to the fact that the method goes through each sample and match the treated and non-treated unit based on propensity score, and yet the distributions of propensity scores for the non-treated group are skewed, resulting in some unmatched samples from the treated group. 

On the other hand, stratification provides the least run time on all three models with a relatively high accuracy. The stratification method groups subjects with similar propensity scores into mutually exclusive stratum. This method is powerful in our scenario because it reduces the impact of the extremely unbalanced distribution of propensity scores between the treated and non-treated group by creating relatively more balanced subgroups. 

Our best model is Doubly Robust Estimation, which returns an almost 100% accuracy on the low dimensional dataset and 96% accuracy on the high dimensional dataset. Doubly Robust estimation provides a simple way of combining linear regression with the propensity score to produce a doubly robust estimator, requiring only one of the models to be correct to identify the causal effect. Proven by our results, this algorithm outperforms the other models with its strong consistency and accuracy in prediction. 
