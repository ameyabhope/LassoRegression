#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('data.csv')

from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score

#Creating a Function for Lasso Regression
def lasso_regression(x_train,y_train,x_test,alpha):
    #Fitting the lasso model to our training data
    lasso_reg = Lasso(alpha=alpha, normalize=True, max_iter = 1e5)
    lasso_reg.fit(x_train,y_train)
    #Predicting the output
    y_pred = lasso_reg.predict(x_test)    
    #Comparing the obtained result with the test data
    y_true = np.where(y_pred> 0.5, 1, 0)
    #Calculating the accuracy
    accuracy = [accuracy_score(y_true ,y_test)*100]
    
        #returning the value of coefficeint and predicted output
    result = [accuracy]
    result.extend([y_true])
    result.extend(lasso_reg.coef_)
   
    return result



#independent variable(feature)
x = dataset.iloc[0:,2:-1]

#Independent variable(response vector): Whether Malignant or Benignant
y = dataset.iloc[0:,1]

#Encoding the dependent variable
from sklearn.preprocessing import LabelEncoder
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)

#Separating the Training Set and Test Set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 0)

alpha_lasso = [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1e-1, 0.5, 1]


#Initialize the dataframe to store coefficients
col = ['accuracy']+['y_true']+['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean',	'compactness_mean',	'concavity_mean', 'concavepoints_mean',	'symmetry_mean', 'fractal_dimension_mean','radius_se', 'texture_se','perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concavepoints_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concavepoints_worst', 'symmetry_worst','fractal_dimension_worst']
#col = ['coef_x_%d'%i for i in range(0,30)]
ind = ['alpha_%.2g'%alpha_lasso[i] for i in range(0,10)]
lasso_output = pd.DataFrame(index=ind, columns=col)

#Iterate over the 10 alpha values:
for i in range(10):
    lasso_output.iloc[i,] = lasso_regression(x_train,y_train,x_test,alpha_lasso[i])

acc = pd.Series.tolist(lasso_output.accuracy)

#Plotting the graph
plt.plot(alpha_lasso,acc)
plt.loglog()
plt.xticks([1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1e-1, 0.05, 10])
plt.xlabel('Alpha values')
plt.ylabel('Accuracy')
plt.title('Aplha Values v/s Accuracy')
plt.show()

#Exporting to Excel
lasso_output.to_csv('Output.csv')



    