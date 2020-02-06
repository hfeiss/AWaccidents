import numpy as np
import pandas as pd
import joblib
from filepaths import paths
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from tokenator import tokenize_and_lemmatize


paths = paths(1)
clean = paths.data.clean.path
models = paths.models.path

df = pd.read_pickle(clean + '/clean.pkl')
X = df['description']
y = np.array(df['target'])


def make_vectorizer():
    vectorizer = CountVectorizer(ngram_range=(1, 2),
                                 max_df=0.55,
                                 max_features=100000,
                                 token_pattern=None,
                                 tokenizer=tokenize_and_lemmatize)

    vectorizer.fit(X)
    joblib.dump(vectorizer, models + '/bayes_vector.joblib')


vectorizer = joblib.load(models + '/bayes_vector.joblib')


def vector(data):
    return vectorizer.transform(data)


def make_bayes():

    bayes = MultinomialNB()
    bayes.fit(vector(X), y)

    joblib.dump(bayes, '/Users/hfeiss/dsi/capstone-2/models/bayes.joblib')


death = ['It could have been a good day of kayaking. The water levels '
         'were very high, but everyone was stoked. On the first rapid '
         'Jack capsized and swam into a strainer. Meanwhile, Jill got '
         'pinned in a sieve. Both spent about 10 minutes underwater '
         'before we could get to them. We performed CPR, but they were '
         'both blue. We called the sherrif, the ambuance came, and we '
         'cried a bunch.']

medical = ['There was a diabetic on our trip. He forgot his insulin. He '
           'ended up in DKA, so we pulled off of the the river. Luckily '
           'we had cell service, so we called 911. He got rushed to the '
           'ER, but the docs said hed be okay even though he had been '
           'near death earlier that day. Another person on the trip '
           'was doing a bunch of drugs like xanax, accutane, tramadol, '
           'and propecia. What a combo! They ended up falling in the river.']

injury = ['It was the end of the day, and everyone was tired. The raft guide '
          'decided to drop into the last hole sideways, and dumptrucked '
          'everyone into the river. There wasnt much rapid left at that point '
          'but most people found a rock or two to hit. Sarah bruised her leg. '
          'Sam hit his head. I got my foot trapped in the webbing of the raft.'
          'Everyone was okay, but a few of us had to get stitches.']

if __name__ == "__main__":
    # make_vectorizer()
    # make_bayes()

    vectorizer = joblib.load(models + '/bayes_vector.joblib')
    bayes = joblib.load(models + '/bayes.joblib')

    prediction = bayes.predict_proba(vector(death))
    print(prediction)

    prediction = bayes.predict_proba(vector(medical))
    print(prediction)

    prediction = bayes.predict_proba(vector(injury))
    print(prediction)
