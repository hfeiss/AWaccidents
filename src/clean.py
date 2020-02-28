import pandas as pd
import numpy as np
from filepaths import Root
from spacy.lang.en import English


paths = Root(__file__, 1).paths()
raw = paths.data.raw.path
clean = paths.data.clean.path

pd.set_option('display.max_columns', 100)

difficulty_dict = {
                   'I': 1.0,
                   'I+': 1.5,
                   'II-': 1.5,
                   'II': 2.0,
                   'II+': 2.5,
                   'III-': 2.5,
                   'III': 3.0,
                   'III+': 3.5,
                   'IV-': 3.5,
                   'IV': 4.0,
                   'IV+': 4.5,
                   'V': 5.0
                   }

type_dict = {
             'F': 'Fatal',
             'I': 'Injury',
             'M': 'Medical'
             }

level_dict = {
              'L': 0,
              'M': 1,
              'H': 2,
              'F': 3
              }

exper_dict = {
              'X': 3,
              'E': 2,
              'S': 1,
              'I': 0
              }

causes = pd.read_csv(raw + '/causes.csv')
cause_dict = pd.Series(causes['cause'].values, index=causes['id']).to_dict()


def read_clean_write(source, dest, verbose=True):
    df = pd.read_csv(source)

    # drop duplicate columns and sensitive information
    cols_to_drop = [col for col in df.columns][1::2]
    cols_to_drop.extend(['victimname', 'othervictimnames',
                         'contactname', 'contactphone',
                         'contactemail', 'reachid', 'status',
                         'groupinfo', 'numvictims'])
    for col in cols_to_drop:
        del df[col]

    # make datetimes
    df['accidentdate'] = pd.to_datetime(df['accidentdate'])

    # kayak or not: other categories are unclear
    df['kayak'] = (df['boattype'] == 'K').astype(int)
    del df['boattype']

    # private or comm trip
    df['commercial'] = (df['privcomm'] == 'C').astype(int)
    del df['privcomm']

    # better to just map to linear scale
    df['experience'] = df['experience'].map(exper_dict)

    # map water level to linear scale
    df['rellevel'] = df['rellevel'].map(level_dict)

    # map roman numerals / classes to numbers
    df['difficulty'] = df['difficulty'].map(difficulty_dict)

    # map causes id# to text
    df['cause'] = df['cause'].map(cause_dict)

    # combine all text into one column
    text_columns = ['river', 'section', 'location', 'waterlevel', 'cause']
    for col in text_columns:
        df['description'] = df['description'].str.cat(df[col],
                                                      sep=' ',
                                                      na_rep='')
        del df[col]

    # dummies for target feature
    dummies = pd.get_dummies(df['type'])
    dummies = dummies.iloc[:, [1, 2, 3]]
    df = df.join(dummies)
    df['target'] = df['I'] + 2*df['F']

    df['type'] = df['type'].map(type_dict)

    if verbose:
        print('Featurized')

    df.reset_index(drop=True)
    df.to_pickle(dest)
    if verbose:
        print('Wrote pkl')


if __name__ == "__main__":
    source = raw + '/accidents.csv'
    dest = clean + '/clean.pkl'
    read_clean_write(source, dest)
    df = pd.read_pickle(dest)
    dropped = df.dropna()
    print(df.head())
    print(f'Number of NaN: {len(df) - len(dropped)}')
