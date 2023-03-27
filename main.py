import pandas as pd
import openpyxl
import numpy
a = 10
b = 23
c = 55


f = pd.DataFrame([[a], [b], [c]],
                  index=['Class 1', 'Class 2', 'Class 3'], columns=['Amount'])
f.to_excel('OSPAR_Test.xlsx', sheet_name='First_try')