import pandas as pd
import numpy as np

# firts line is poly ext, second is poly int

f = open("map/map2/map_big.txt", 'r')

contents = f.readlines()

f.close()

contents[0].replace("\n", '')
contents[1].replace("\n", '')

p_ext = contents[0].split("), (")
p_int = contents[1].split("), (")



p_ext_tuple = np.asarray([eval("(" + a + ")") for a in p_ext])
p_int_tuple = np.asarray([eval("(" + a + ")") for a in p_int])

dataframe = pd.DataFrame(p_ext_tuple)

dataframe.rename(columns={0: "x_ext", 1: "y_ext"})

p_int_tuple_tr = np.transpose(p_int_tuple)

dataframe.insert(2, "x_int", p_int_tuple_tr[0] )
dataframe.insert(3, "y_int", p_int_tuple_tr[1] )

dataframe.to_csv("map/map2/sample.csv", sep = ",", header=False, index=False)




