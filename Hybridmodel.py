import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def compute_prob(x, prob):
    new_x = []
    for row, prob in zip(x, prob):
        new_x.append(np.concatenate([row, prob]))

    return new_x


def rf_alg(train_x, test_x, train_y, test_y):
    model = RandomForestClassifier()
    model.fit(train_x, train_y)
    prediction = model.predict(test_x)
    return prediction


def dt_alg(train_x, test_x, train_y, test_y):
    model= tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    prediction = model.predict(test_x)
    return prediction


def hy_alg(train_x, test_x, train_y, test_y):
	rf = RandomForestClassifier()
	rf.fit(train_x, train_y)
	prob = rf.predict_proba(train_x)
	train_x = compute_prob(train_x, prob)
	clf = tree.DecisionTreeClassifier()
	clf.fit(train_x, train_y)
	prob = rf.predict_proba(test_x)
	test_x = compute_prob(test_x, prob)
	y_pred = clf.predict(test_x)

	output=open("Output/resultHybrid.csv","w")
	output.write("ID,Predicted Results" + "\n")
	for j in range(len(y_pred)):
	    output.write(str(j+1) + "," + str(y_pred[j]) + "\n")
	output.close()


	mse=mean_squared_error(test_y, y_pred)
	mae=mean_absolute_error(test_y, y_pred)
	r2=r2_score(test_y, y_pred)
	rmse = np.sqrt(mean_squared_error(test_y, y_pred))
	accuracy=accuracy_score(test_y,y_pred)

	output=open('Output/EvalHy.csv', 'w')
	output.write("Parameter,Value" + "\n")
	output.write("MSE" + "," +str(mse) + "\n")
	output.write("MAE" + "," +str(mae) + "\n")
	output.write("R-SQUARED" + "," +str(r2) + "\n")
	output.write("RMSE" + "," +str(rmse) + "\n")
	output.write("ACCURACY" + "," +str(accuracy) + "\n")
	output.close()
	
	
	df =  pd.read_csv('Output/EvalHy.csv')
	acc = df["Value"]
	alc = df["Parameter"]
	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
	explode = (0.1, 0, 0, 0, 0)  
	
	fig = plt.figure()
	plt.bar(alc, acc,color=colors)
	plt.xlabel('Parameter')
	plt.ylabel('Value')
	plt.title('Hybrid Evaluation Metrics Plot')
	fig.savefig('Output/Hy.png') 
	plt.show(block=True)
	plt.close()
	return y_pred

def compute_acc(test_y, prediction):
    match = 0
    for actual, predicted in zip(test_y, prediction):
        if actual == predicted:
            match += 1
    acc = str(match / len(test_y) * 100)
    return acc


def process(path):
	dataset = pd.read_csv(path)
	X = dataset.iloc[:, 0:13].values
	y = dataset.iloc[:, 13].values

	train_x, test_x, train_y, test_y = train_test_split(X, y)
	
	hypred = hy_alg(train_x, test_x, train_y, test_y)
	hy_acc = compute_acc(test_y, hypred)
	
	dtpred = dt_alg(train_x, test_x, train_y, test_y)
	dt_acc = compute_acc(test_y, dtpred)
	
	rfpred = rf_alg(train_x, test_x, train_y, test_y)
	rf_acc = compute_acc(test_y, rfpred)
	
	
process("dataset.csv")