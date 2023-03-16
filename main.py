import pandas as pd
import openpyxl
import numpy
f = pd.DataFrame([[15, 28, 40], [12, 22, 32], [31, 32, 33]],
                  index=['one', 'two', 'three'], columns=['a', 'b', 'c'])
f.to_excel('OSPAR_Test.xlsx', sheet_name='First_try')