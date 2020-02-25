import pandas as pd
from filepaths import Root
from spacy.lang.en import English


paths = Root(__file__, 1).paths()
clean = paths.data.clean.path

pd.set_option('display.max_columns', 100)

my_stops = ['>', '<', 'p', '/p', 's', 'o', 't', ', ', 'd', '444444',
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
            'lock', 'priority', 'accent', 'semihidden', 'unhidewhenused',
            'table', 'list', 'lock', 'semihidden', 'amp', 'bt', 'grid',
            'layout', 'mode', 'narrative', 'initial', 'variant', 'weight',
            'outline', 'baseline', 'datum', 'vertical', 'leave', 'image',
            'max', 'position', 'display', '68', 'https', 'right', 'ligature',
            'stockticker', '08', '11', '06', '12', 'pa', 'source', '11pt',
            'large', 'march', 'tira', 'niyhwk', 'tcenter', 'posr', 'jim',
            'georgia', 'lucas', 'posr', 'mark', 'get', 'rock', 'be', 'kayker',
            'time', 'ndn', 'thumbtitle', 'thumbnail', 'sliderthumbnailoverlay',
            'neacato', '07', 'witness', 'stockticker', '4', '5', '6', '7',
            'jpg', '300w', 'neue', 'lucida', 'header', 'segoe', 'byline',
            'at4', '75em', '400', '1rem', 'and', 'let', 'near', 'new',
            'colorful', 'medium', 'shade', 'story', 'news', '0806', '350598',
            'wset', 'james', 'article', 'qformat', 'shade', 'provide', 'month', 
            'date', 'spacerun', 'fareast', 'attachment', 'origin', 'clip',
            'black', 'cap', '5pt', 'language', 'aolmail', 'decoration', 'webkit',
            'block', 'inline', '100', 'h1', '20px', '16px', '1em', 'title', 'auto',
            '5px', 'transform', '102', 'transparent', 'light', 'lsdexce', '10', '14',
            '20', '15', '234', 'ion', '16', '17', '35']

seperators = ['.', ';', ':', '/', '&', '=', '(', ')', '-', ',', '>', '<', '_',
              '{', '}', 'px', 'pt', 'mso']

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
    df = pd.read_pickle(clean + '/clean.pkl')
    entry = df['description'][20]
    print(entry)
    print(tokenize_and_lemmatize(entry))

    df = df[df['F'] == 1]
    df['description'] = df['description'].apply(tokenize_and_lemmatize)
    pd.to_pickle(df['description'], clean + '/death_lemmas.pkl')
