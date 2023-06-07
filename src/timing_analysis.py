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
	for name in table.index:
		print(display_str.format(
			name,
			int(table[name].mean()),
			int(table[name].std())),
		)


