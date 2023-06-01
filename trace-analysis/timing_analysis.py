import pandas
# import ipdb

df = pandas.read_csv("logfile.csv", sep=';', header=None)
columns = ['Info','Tp','Time','InOut', 'Name']
df.columns = columns
df.Time = df.Time.astype('float')
delta = (df.Time - df.Time.shift(1)) / 1000.0 / 1000.0 ## ns -> us -> ms
df['delta'] = delta
print(df)

# We really only need the table and some groupings. Maybe we can make a flame
# graph later...

# --------------------------------------------

# df.plot()
# import matplotlib.pyplot as plt
# plt.show()
# input("wait...")

# import sqlite3
# db = sqlite3.connect("testsqlite.db")
# dft = pandas.read_sql_query("select * from dft ;", db)
