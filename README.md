# Background
![AW](/images/screenshots/AW_logo-HorizColor_large.jpg)

The American Whitewater Accident Database catalogs over 2,400 fatalities and close calls on whitewater rivers dating back to 1972.

The project was initiated in 1975 when Charlie Walbridge observed a fatality due to foot entrapment at a slalom race. Ever since, the American Whitewater journal has collected incident reports and shared the lessons learned. In 2017, the collection of accidents was refined and made available for download on [American Whitewater's website](https://www.americanwhitewater.org/content/Accident/view/).

These reports provide a learning opportunity to the paddling community, and facilitate dialogue with river managers and decision makers. 

The goal of this repository is to identify factors that can turn a near miss into a fatality, hopefully reducing tradgeys and statistics alike.

# Vocabulary
Foot Entrapment
![](images/screenshots/foot_entrapment.jpg)


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
* experienced
* Type of accident
    * Fatality
    * Medical (near miss)
    * Injury (near miss)


# EDA
## `description`

Because the descriptions of accidents are aggregated from both external websites and user submitted forms, the documents have very inconsistent structure. 

All have some level of `html` embedded in them, and some are actually `json`. The first step in the text analysis was to convert each document into one string. The strings were then tokenized with a purpose-built function. [Spacy's](https://spacy.io) english stop words were used as base to start. Beacuse of inconsistent description tense,  the documents were lemmatized into their root words before being vectorized into a tf-idf matrix.


### K-means
Hard clustering of the tf-idf matrix was used to find underlying structure. Typically, some clusters identified more `html` words to add to the custom tokenization script.

![](/images/screenshots/topics_w_html.png)

After many itterations, the clustering still identified more stop words, but salient topics emmerged.

![](/images/screenshots/drugs.png)

    0) more stopwords
    1) more stopwords
    4) some stopwords, accidents clustered on drugs
    5) descriptions of rivers
    6) emergency response
    7) descriptions of location

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

## non-text features

The number of accidents is likely proportional to the amount of whitewater recreation in a given state. 

![](/images/screenshots/map.png)



![](/images/dates.png)




# Pipeline
## Map Values


# Models
## Naive Bayes
## Boosting
## Random forest
## Stacked
## Logistic Regression

# Conclusions
dont do drugs
age
commercial
kayaking

words to avoid

competent group

# Future


![](/images/level_diff_death_2.png)
![](/images/exper_age_death.png)


![](/images/screenshots/lochsa.jpg)
![](/images/screenshots/good_topics.png)
![](/images/screenshots/grid_vector.png)
![](/images/screenshots/log_mod.png)
![](/images/bagging_features_horiz.png)
![](/images/boosting_n_score.png)
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