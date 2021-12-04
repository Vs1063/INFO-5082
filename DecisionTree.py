import pandas as pd
import matplotlib as plt
import numpy as np
from sklearn import linear_model
from scipy.stats import norm

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from csv import reader
import csv
import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score




def process(path):

	dataset = pd.read_csv(path)
	X = dataset.iloc[:, 0:13].values
	y = dataset.iloc[:, 13].values

	X_train, X_test, y_train, y_test = train_test_split(X, y)

	DTmodel= DecisionTreeClassifier()
	DTmodel.fit(X_train, y_train)
	y_pred = DTmodel.predict(X_test)


	output=open("Output/DTpredict.csv","w")
	output.write("ID,Predicted Results" + "\n")
	for j in range(len(y_pred)):
	    output.write(str(j+1) + "," + str(y_pred[j]) + "\n")
	output.close()
	
	mse=mean_squared_error(y_test, y_pred)
	mae=mean_absolute_error(y_test, y_pred)
	r2=r2_score(y_test, y_pred)
	rmse = np.sqrt(mean_squared_error(y_test, y_pred))
	accuracy=accuracy_score(y_test,y_pred)

	output=open('Output/EvalDT.csv', 'w')
	output.write("Metrics,Value" + "\n")
	output.write("MSE" + "," +str(mse) + "\n")
	output.write("MAE" + "," +str(mae) + "\n")
	output.write("R-SQUARED" + "," +str(r2) + "\n")
	output.write("RMSE" + "," +str(rmse) + "\n")
	output.write("ACCURACY" + "," +str(accuracy) + "\n")
	output.close()
	
	
	df =  pd.read_csv('Output/EvalDT.csv')
	acc = df["Value"]
	alc = df["Metrics"]
	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
	explode = (0.1, 0, 0, 0, 0)  
	
	fig = plt.figure()
	plt.bar(alc, acc,color=colors)
	plt.xlabel('Metrics')
	plt.ylabel('Value')
	plt.title(' Decision Tree Evaluation Metrics Plot')
	fig.savefig('Output/DT.png') 
	plt.show(block=True)
	plt.close()

process("dataset.csv")