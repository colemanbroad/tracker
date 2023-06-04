import pandas
import ipdb
import numpy as np
from numpy import array

def run():

	# info: ;tp;319263177898583;0;brute_force;2314411
	cols = [
		"info",
		"tp",
		"time",
		"start",
		"name",
		"id",
	]
	table = pandas.read_csv("../err.txt", sep=';', names=cols)
	# ipdb.set_trace()

	def timedelta(df):
		a = array(df.time.iloc[1::2]) - array(df.time.iloc[::2])
		return a
	table = table.groupby('name').apply(timedelta)
	print(table['brute_force'].mean(), table['brute_force'].std())
	print(table['kdtree'].mean(), table['kdtree'].std())




