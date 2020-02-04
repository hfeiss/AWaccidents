import pandas as pd
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
             'family', 'water', 'boat', 'stay', 'helvetica', 'st', 'inherit',
             'width', 'false', 'face', 'non', '51', 'say', 'raft', 'rapid',
             'year', '1', '2', '3', 'rescue', 'true', 'paddle', 'w',
             'lock', 'priority', 'accent' 'semihidden', 'unhidewhenused',
             'table', 'list', 'lock', 'semihidden', 'amp', 'bt', 'grid',
             'layout', 'mode', 'narrative']

seperators = ['.', ';', ':', '/', '&', '=', '(', ')', '-', ',', '>', '<']

htmls = ['\\', '\r', '\n', '\t']

spaces = [' '*i for i in range(1, 6)]

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


if __name__ == "__main__":
    df = pd.read_pickle('/Users/hfeiss/dsi/capstone-2/data/clean/clean.pkl')
    entry = df['description'][20]
    print(entry)
    print(tokenize_and_lemmatize(entry))
    
    df = df[df['F'] == 1]
    df['description'] = df['description'].apply(tokenize_and_lemmatize)
    pd.to_pickle(df['description'], '/Users/hfeiss/dsi/capstone-2/data/clean/death_lemmas.pkl')