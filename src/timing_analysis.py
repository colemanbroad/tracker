import pandas
import ipdb
import numpy as np
from numpy import array

import tabulate

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
	
	display_str = "{:12s} | {:8_d} | {:8_d}"
	print(display_str.format(
		'kdtree',
		int(table['kdtree'].mean()),
		int(table['kdtree'].std())),
	)

	print(display_str.format(
		'brute_force',
		int(table['brute_force'].mean()),
		int(table['brute_force'].std())),
	)

	print(display_str.format(
		'sorted',
		int(table['sorted'].mean()),
		int(table['sorted'].std())),
	)




