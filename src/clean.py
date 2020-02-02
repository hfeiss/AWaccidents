import pandas as pd
import numpy as np
from spacy.lang.en import English


pd.set_option('display.max_columns', 100)

my_stops =  ['>', '<', 'p', '/p', 's', 'o', 't', ', ', 'd', '444444',
             '0pt', '1pt', '2pt', '4pt', '10pt', '12pt', '14pt', '15pt', 
             '0px', '1px', '2px', '4px', '10px', '12px', '14px', '15px',
             'rgb', '255', '0', 'li', 'div', 'u', 'b', '0001pt', '39', '51'
             'meta', 'font', 'size', 'arial', 'nbsp', 'align', 'justify',
             'href', 'style', 'quot', 'msonormal', 'serif', 'text', 'ldquo',
             'rdquo', 'height', 'text', 'mso', 'san', 'margin', 'class', 'tab',
             'roman', 'times', 'http', 'www', 'html', 'background', 'pad',
             'bidi', 'color', 'bidi', 'san', 'rsquo', 'br', 'spin', 'letter',
             'spacing', 'space', 'hyphenate', 'place', 'line', 'placename',
             'placetype', 'border', 'box', 'normal', 'com', 'url', 'link',
             'publish', 'lsdexception', '00', '000', '000000', 'river',
             'family', 'water', 'boat', 'stay']

seperators = ['.', ';', ':', '/', '&', '=', '(', ')', '-', ',', '>', '<']

htmls = ['\\', '\r', '\n', '\t']

spaces = [' '*i for i in range(1, 6)]

difficulty_dict = {'I': 1.0,
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
                   'V': 5.0}


causes = pd.read_csv('/Users/hfeiss/dsi/capstone-2/data/raw/causes.csv')
cause_dict = pd.Series(causes['cause'].values, index=causes['id']).to_dict()


nlp = English()
nlp.Defaults.stop_words |= set(my_stops)

def tokenize_and_lemmatize(text):
    text = str(text).lower()
    for sep in seperators:
        text = text.replace(sep, ' ')
    for html in htmls:
        text = text.replace(html, '')
    for space in spaces:
        text = text.replace(space, ' ')
    doc = nlp(text)
    # word is a spacy doc object here
    not_stops = [word for word in doc if not word.is_stop]
    not_punct = [word for word in not_stops if not word.is_punct]
    words = [token.lemma_ for token in not_punct if token.lemma_ != ' ']
    # remove custom stopwords after lemmatization
    return [word for word in words if word not in my_stops]


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

    # experience expert or experienced
    df['experienced'] = ((df['experience'] == 'E') |
                         (df['experience'] == 'X')).astype(int)
    del df['experience']

    # highwater or not
    df['highwater'] = ((df['rellevel'] == 'H') |
                       (df['rellevel'] == 'F')).astype(int)
    del df['rellevel']

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
    del df['type']

    df['target'] = 2*df['I'] + 3*df['F']

    if verbose:
        print('Featurized')

    df['description'] = df['description'].apply(tokenize_and_lemmatize)
    if verbose:
        print('Tokenized')

    df.reset_index(drop=True)
    df.to_pickle(dest)
    if verbose:
        print('Wrote pkl')


if __name__ == "__main__":
    source = '/Users/hfeiss/dsi/capstone-2/data/raw/accidents.csv'
    dest = '/Users/hfeiss/dsi/capstone-2/data/clean/clean.pkl'
    read_clean_write(source, dest)
    df = pd.read_pickle(dest)
    dropped = df.dropna()
    # print(df.head())
    print(f'Number of NaN: {len(df) - len(dropped)}')
