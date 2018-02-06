
import pandas as pd
import numpy as np


# path to a file with training data
DATAFILE_PATH = './data/broad_leaf_GBIF.csv'

# a tuple of all predictors to be used in the model
MODELING_VARIABLES = ('BIO')


# ------------- Load the data --------------------------

original_data = pd.read_csv(DATAFILE_PATH, sep=';')

print(" ============== Original dataset overview ================")
original_data.info()
print("="*60)
# ------------------------------------------------------



# -------------- Data-preprocessing -----------------
data = original_data[['species', 'decimallatitude', 'decimallongitude']]

# remove nan-rows
data = data.dropna()

# remove duplicated rows
data = data.drop_duplicates()


# column names are too long, lets make them shorter...
data = data.rename(index=str, columns={"decimallatitude": "lat",
                                "decimallongitude": "lon"})

# reset index
data = data.reset_index()


print("================== Species counts =======================")
print(data['species'].value_counts())
print("="*60)
# ------------------------------------------------------










