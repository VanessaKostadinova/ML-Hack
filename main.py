import pandas as pd

# reading tornado dataset
dataset = pd.read_csv('dataset/2011Tornado_Summary.csv')
print(dataset)

keywords = ['storm', 'wildfire', 'blizzard', 'hurricane', 'flood']