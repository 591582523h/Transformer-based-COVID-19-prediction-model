
import numpy as np
import pandas as pd


file = 'data/us_covid.xlsx'
data = pd.read_excel(file)
confirm = list(data['Confirmed'])
confirm_change = []
for i in range(len(confirm)-1):
    confirm_change.append(confirm[i+1] - confirm[i])

confirm_change = np.array(confirm_change).reshape((-1, 1))
df = pd.DataFrame(confirm_change)
print(df)
df.to_excel("usa.xlsx", index=0)

















