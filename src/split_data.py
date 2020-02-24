import numpy as np
import pandas as pd
import math
from pprint import pprint
from filepaths import Root
from sklearn.model_selection import train_test_split


paths = Root(__file__, 1).paths()
clean =  paths.data.clean.path
holdout = paths.data.holdout.path
train = paths.data.train.path

train_df = pd.read_pickle(clean + 'clean.pkl')
holdout_df = pd.DataFrame(columns=train_df.columns)

holdout_size = 0.25
holdout_size = math.floor(holdout_size*len(train_df))
all_rows = range(len(train_df))
choices = np.random.choice(all_rows, size=holdout_size, replace=False)

holdout_df = train_df.iloc[choices]
train_df.drop(choices, inplace=True)


if __name__ == "__main__":
    pd.to_pickle(train_df, train + 'train.pkl')
    pd.to_pickle(holdout_df, holdout + 'holdout.pkl')
    print(train_df.head())
    print(train_df.info())
    print(holdout_df.head())
    print(holdout_df.info())