from collections import Counter
import os
from numpy.core.fromnumeric import sort 
import pandas as pd
from pprint import pprint as print

isitokay = pd.read_csv('/opt/ml/teamrepo/team/output/output.csv')
isitokay = isitokay.sort_values(by=["ans"], ascending=[True]) 
isitokay = Counter(isitokay['ans'])
print(isitokay)


