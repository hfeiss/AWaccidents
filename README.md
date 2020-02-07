# Background
![AW](/images/screenshots/AW_logo-HorizColor_large.jpg)

The American Whitewater Accident Database catalogs over 2,400 fatalities and close calls on whitewater rivers dating back to 1972.

The project was initiated in 1975 when Charlie Walbridge observed a fatality due to foot entrapment at a slalom race. Ever since, the American Whitewater journal has collected incident reports and shared the lessons learned. In 2017, the collection of accidents was refined and made available for download on [American Whitewater's website](https://www.americanwhitewater.org/content/Accident/view/).

These reports provide a learning opportunity to the paddling community, and facilitate dialogue with river managers and decision makers. 

The goal of this repository is to identify factors that can turn a near miss into a fatality, hopefully reducing tradgeys and statistics alike.

# Vocabulary
Foot Entrapment
![](images/screenshots/foot_entrapment.jpg)

Pin / Wrap
![](images/screenshots/pin.jpg)


# Data

## American Whitewater Accident Database

The data is a combination of user submitted forms and web-scrapped articles. As such, it is supremely messy.
![](/images/screenshots/messy.png)

## Features

After deleting personal information, all text features were combined into the `description` column. In addition to the text, this analysis focused on:
* State (location)
* River level 
* River difficulty
* Victim age
* Text description of the incident
* kayak
* commercial
* experience
* Type of accident
    * Fatality
    * Medical (near miss)
    * Injury (near miss)


# EDA
## `description`

Because the descriptions of accidents are aggregated from both external websites and user submitted forms, the documents have very inconsistent structure. 

All have some level of `html` embedded in them, and some are actually in `json`. The first step in the text analysis was to convert each document into one long string. The strings were then tokenized with a purpose-built function. [Spacy's](https://spacy.io) english stop words were used as base to start. Beacuse of inconsistent description tense,  the documents were lemmatized into their root words before being vectorized into a tf-idf matrix.

### K-means
Hard clustering of the tf-idf matrix was used to find underlying structure.

![](/images/screenshots/topics_w_html.png)

Typically, some clusters identified more `html` words to add to the custom tokenization script.

![](/images/screenshots/drugs.png)

After many itterations, the clustering still identified more stop words, but salient topics emmerged.

![](/images/screenshots/good_topics.png)

    1) some stopwords, accidents clustered on drugs
    3) man-made hazards
    4) east coast boating
    5) Idaho rivers
    7) more stopwords

### PCA

Primary component analysis proved to be unilluminating. Less than 0.1% of the varience was explained in the first 8 components.

Below, the first two components are plotted with each accident labeled as a fatality, injury, or a medical emergency.

![](images/pca_targets_idf.png)

### LDA

Latent Dirichlet Analysis also failed to reveal new information.

![](/images/screenshots/lda.png)

### Description Length

As the descriptions become longer, a higher proportion of the accidents are fatal.

![](/images/description_len_death.png)

## Non-text Features

The number of accidents is likely proportional to the amount of whitewater recreation in a given state.

![](/images/screenshots/map.png)

The raw number of accidents has increased over time, but this is unadjusted for population and sport popularity.

![](/images/dates.png)


# Pipeline
## Text
Text from the river, section, location, waterlevel, and cause features were added into the description column.

The description column was tokenized, lemmatized, and vectorized before analysis.

## Categorical
River level (Low, Medium, High, and Flood), river difficulty (I, II, III, IV, V), and victim skill (Inexperienced, Some experience, Experienced, Expert) were mapped linearly to integers.

Type of watercraft was mapped to kayak (1) or not.

Trip type was mapped to commerical (1) or not.

Given an unreasonable number of 0 year olds with contradicory description entries, ages equal to 0 were dropped.

# Models
Sklearn grid searching was used to find the best hyperparameters. Models were tested on classification into three groups (Fatality, Injury, Medical) as well as Fatal or Not Fatal. For simplicity and interpitability, only the binary classification results are shown.


## Text Classification
 Tf-idf and tf matricies were compared.  Models were trained with k-folds crossvalidation and performence was judged on a holdout data set.

|          Model 	| Precision 	| Recall 	| Accuracy 	|
|---------------:	|-----------	|--------	|----------	|
| Random Forest  	| 77%       	| 98%    	| 81%      	|
| AdaBoost       	| 86%       	| 92%    	| 86%      	|
| Bagging        	| 87%       	| 92%    	| 87%      	|
| Naive Bayes    	| 76%       	| 95%    	| 79%      	|

### Boosting
![](/images/boosting_n_score.png)


### Bagging
![](/images/bagging_features_horiz.png)

### Naive Bayes

#### Medical
    There was a diabetic on our trip. He forgot his insulin. He ended up in DKA, so we pulled off of the the riveLuckily we had cell service, so we called 911. He got rushed the ER, but the docs said hed be okay even though he had been near death earlier that day. Another person on the trip was doing a bunch of drugs like xanax, accutane, tramado and propecia. What a combo! They ended up falling in the river.

|                       	| Medical 	| Injury 	| Fatality 	|
|----------------------:	|---------	|--------	|----------	|
| Predicted Probability 	| 99.9%   	| 0.0%   	| 0.0%     	|
#### Injury

#### Fatality
    It could have been a good day of kayaking. The water levels were very high, but everyone was stoked. On the first rapid Jack capsized and swam into a strainer. Meanwhile, Jill got pinned in a sieve. Both spent about 10 minutes underwater before we could get to them. We performed CPR, but they we both blue. We called the sherrif, the ambuance came, and we cried a bunch.
|                       	| Medical 	| Injury 	| Fatality 	|
|----------------------:	|---------	|--------	|----------	|
| Predicted Probability 	| 0.25%   	| 0.15%  	| 99.6%    	|

## Non-text regression

## Logistic Regression

|          Model 	        | Precision 	| Recall 	| Accuracy 	|
|---------------:	        |-----------	|--------	|----------	|
| Logistic Classification 	| 92%       	| 100%   	| 92%      	|
| NB, AB Stacked 	        | 88%       	| 92%    	| 84%      	|

## Stacked

|          Model 	        | Precision 	| Recall 	| Accuracy 	|
|---------------:	        |-----------	|--------	|----------	|
| NB, AB Stacked 	        | 88%       	| 92%    	| 84%      	|

# Conclusions
dont do drugs
age
commercial
kayaking

competent group

#### Good words
'large kayak' 'large flow''john wetsuit' feel emotion, spot stop, feel pressure, respond 'competent group',

 'thank thank', 'thank tell', 'thank support', 'cruikshank woman', 'thank shayne', 'thankful train', 'thank share', 

#### Bad words
'new', 'rock', 'rope', 'victim', 'kayaker', 'help', 'time', 'swim',
'fall', 'pin', 'foot','strainer', 'accident', 'chute enter'
'body', 'county', 'search', 'drown', 'rock', 'time', 'kayaker', 'be', 'accident march kayaker overdose, dam

# Future


![](/images/level_diff_death_2.png)
![](/images/exper_age_death.png)


![](/images/screenshots/lochsa.jpg)
![](/images/screenshots/grid_vector.png)
![](/images/screenshots/log_mod.png)
![](/images/bagging_features_horiz.png)
![](/images/elbow_km.png)
![](/images/scree.png)
![](/images/silh_km.png)


water level positively correlated with death
difficulty: no correlation
age: very correlated with death
kayaking: negatively correlated with death
commercial: possible correlation (positive), but may be due to increased reporting
experience: no correlation

![]()