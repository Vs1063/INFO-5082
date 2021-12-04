import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def process(path):
	df_main = pd.read_table(path, sep=',')
	df_main.astype(float)
	# Normalize values to range [0:1]
	df_main /= df_main.max()

	data = pd.read_table(path, sep=',')
	

	fig, ax = plt.subplots(figsize=(15,7))
	data[["num", "age"]].groupby(["age"]).count().plot.bar(stacked=True,ax=ax)
	ax.title.set_text('Agewise Count')
	ax.set_ylabel('Total Cases')
	plt.savefig('Output/Agewise.png')
	plt.show(block=True)
	plt.close()


	d=data[data.num==1].groupby(["age"]).count()
	df = pd.DataFrame(d)
	df1=df["num"]
	d=data[data.num==0].groupby(["age"]).count()
	df = pd.DataFrame(d)
	df2=df["num"]
	df=pd.concat([df1, df2], axis=1, sort=False)
	df=df.fillna(0.0)
	df.columns = [ 'y', 'n']
	fig, ax = plt.subplots(figsize=(15,7))
	df.plot.bar(stacked=True,ax=ax)
	ax.title.set_text('Age & Gender Wise')
	ax.set_ylabel('Total Cases')
	plt.savefig('Output/AgeGender.png')
	plt.show(block=True)
	plt.close()

	fig, ax = plt.subplots(figsize=(15,7))
	data[["num", "sex"]].groupby(["sex"]).count().plot.bar(stacked=True,ax=ax)
	ax.title.set_text('Gender')
	ax.set_ylabel('Total cases')
	plt.savefig('Output/Gender.png')
	plt.show(block=True)
	plt.close()

	d=data[data.num==1].groupby(["sex"]).count()
	df = pd.DataFrame(d)
	df1=df["num"]
	d=data[data.num==0].groupby(["sex"]).count()
	df = pd.DataFrame(d)
	df2=df["num"]
	df=pd.concat([df1, df2], axis=1, sort=False)
	df=df.fillna(0.0)
	df.columns = [ 'y', 'n']
	fig, ax = plt.subplots(figsize=(15,7))
	df.plot.bar(stacked=True,ax=ax)
	ax.title.set_text('Genderwise Plot')
	ax.set_ylabel('Total Count')
	plt.savefig('Output/GenderAge.png')
	plt.show(block=True)
	plt.close()
	
process("dataset.csv")