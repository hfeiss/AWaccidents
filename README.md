# Background
![AW](/images/screenshots/AW_logo-HorizColor_large.jpg)
 
The American Whitewater Accident Database catalogs over 2,400 fatalities and close calls on whitewater rivers dating back to 1972.
 
The project was initiated in 1975 when Charlie Walbridge observed a fatality due to foot entrapment at a slalom race. Ever since, the American Whitewater journal has collected incident reports and shared the lessons learned. In 2017, the collection of accidents was refined and made available for download on [American Whitewater's website](https://www.americanwhitewater.org/content/Accident/view/).
 
These reports provide a learning opportunity to the paddling community, and facilitate dialogue with river managers and decision makers.
 
The goal of this repository is to identify factors that can turn a near miss into a fatality, hopefully reducing tragedies and statistics alike.
 
 
# Data
 
## American Whitewater Accident Database
 
The database is created from a combination of user submitted forms and web-scrapped articles. As such, it is supremely messy.

![](/images/screenshots/messy.png)
 
## Features
 
After deleting personal information, all text features (river, section, location, waterlevel, and cause) are combined into the `description` column.

In addition to the written narrative, this analysis focuses on:

* State (location)
* River level
* River difficulty
* Victim age
* kayak
* commercial
* experience
* Type of accident
   * Fatality
   * Medical (near miss)
   * Injury (near miss)

The ordinal features: river level (Low, Medium, High, and Flood), river difficulty (I, II, III, IV, V), and victim skill (Inexperienced, Some experience, Experienced, Expert) are mapped linearly to integers.

Type of watercraft is mapped to kayak (1) or not.

Trip type is mapped to commercial (1) or not.

Accident outcome is mapped to fatal (1) or not.
 
Given an unreasonable number of 0 year olds with contradictory description entries, ages equal to 0 are dropped.

Text from the river, section, location, waterlevel, and cause features are added into the description column.

# EDA
## Pre-processing
 
Because the descriptions of accidents are aggregated from both external websites and user submitted forms, the documents have very inconsistent structure.
 
All documents have some level of `html` embedded in them, and some are actually in `json`. The first step in the text analysis is to convert each document into one long string. The strings are then tokenized with a purpose-built script. [Spacy's](https://spacy.io) english stop words are used as a base to start. Because of inconsistent description tense, the documents are lemmatized into their root words before being vectorized into either a term fequency or tf-idf matrix.
 
Once vectorized, the matrix is clustered with the k-means algorithm. The underlying structure reveals documents with high  percentages of html words. The top words for those html clusters are added to the stopwords, and the process is repeated until salient, clean clusters emerge.

## Latent Diriclet Allocation

LDA does not illuminate any underlying structure.

[![](/images/screenshots/lda.png)](http://hfeiss.info/lda)

## Principal Component Analysis

Similar to LDA, PCA fails to provide new information. Indeed, less than 0.1% of the variance is explained in the first 8 components.

Below, the first two components are plotted with each accident labeled as a fatality, injury, or a medical emergency.

![](images/pca_targets_idf.png)

## Description Length

As expected, as the descriptions of accidents become longer, a higher proportion of accidents are fatal.

![](/images/description_len_death.png)

## Geographic Distribution

The number of accidents is likely proportional to the amount of whitewater recreation in a given state.

![](/images/screenshots/map.png)

## Temporal Distribution

![](/images/dates.png)

# Nautral Language Processing

## Text Classification

Sklearn grid searching is used to find the best hyperparameters with k-folds cross validation. Final performance is judged on a holdout data set. Models are tested on classification into three groups (Fatality, Injury, Medical) as well as Fatal or Near Miss. For simplicity and interpretability, only the binary classification results are shown.

### Bagging

Below are the most important words for predicting the outcome of an accident. It is worth noting that the model does not assert a positive or negative correlation, just predictive importance.

![](/images/bagging_features_horiz.png)

### Naive Bayes

After fitting a Naive Bayes model to the training data, for each category of incident, the top 100 words that made each category more and less likely are generated. Below is a curated subset of those lists.

#### Words that made Injury more likely:
    man, pin, foot, strainer, group, kayaker, march

#### Words that made Injury less likely:
    farmer wetsuit, near drowning, new york, large kayak

#### Words that made Fatality more likely:
    rock, dam, drown, pin, get help, search, rescue, time, large flow

#### Words that made Fatality less likely:
    competent group, thank, support, train, feel emotion, professional sar, respond
 
Below, mock descriptions were fed into the naive bayes model with the resulting predictions.

#### Injury
   > It was the end of the day, and everyone was tired. The raft guide decided to drop into the last hole sideways, and dump trucked everyone into the river. There wasn't much rapid left at the point but most people found a rock or two to hit. Sarah bruised her leg. Sam hit his head. I got my foot trapped in the webbing of raft. Everyone was okay, but a few of us had to get stitches.

<center>

|                        | Injury    | Fatality  |
|----------------------: |--------   |---------- |
| Predicted Probability  | 0.1%      | 99.8%     |

</center>

#### Fatality
   > It could have been a good day of kayaking. The water levels were very high, but everyone was stoked. On the first rapid Jack capsized and swam into a strainer. Meanwhile, Jill got pinned in a sieve. Both spent about 10 minutes underwater before we could get to them. We performed CPR, but they we both blue. We called the sheriff, the ambulance came, and we cried a bunch.

<center>

|                        | Injury    | Fatality  |
|----------------------: |--------   |---------- |
| Predicted Probability  | 0.15%     | 99.6%     |

</center>

# Numerical Feature Analysis
As with the text analysis, grid searching and k-folds cross validation is used to find the best hyperparameters. Final performance is judged on a holdout data set, and only binary classification results are shown.
 
## Boosting
![](/images/boosting_n_score.png)
 
## Logistic Regression
 
A simple logistic was performed on the non-text features. This model performed better than the text analysis. After removing features without predictive strength, the coefficients and their p-values are listed below. The variance inflation factors are all 1.3 or below.

<center>

|          Predictor    | Coef      | p-value   |
|-------------------:   |-------    |---------  |
| River Level           | 0.27      | 0.050     |
| River Difficulty      | 0.45      | 0.003     |
| Paddler Experience    | -0.34     | 0.034     |

</center>


## Stacked Model

Adding the Naive Bayes prediction as a feature in the logistic model, oddly, decreases performance.

# Model Performance

|          Model    | Precision     | Recall    | Accuracy  |
|---------------:   |-----------    |--------   |---------- |
| AdaBoost          | 86%           | 92%       | 86%       |
| Bagging           | 87%           | 92%       | 87%       |
| Naive Bayes       | 76%           | 95%       | 79%       |
| Random Forest     | 77%           | 98%       | 81%       |
| Logistic Classification   | 92%           | 100%      | 92%       |
| NB, LC Stacked            | 88%           | 92%       | 84%       |

 
# Conclusions
 
Combining the information from clustering, topic modeling, natural language processing, and logistic modeling, a few conclusions can be made. However, mostly the data supports existing knowledge in the whitewater community.
 
![](/images/level_diff_death_2.png)
![](/images/exper_age_death.png)

* Competent group - more than any other, this phrase decreased the likelihood of a prediction for death.

* Wetsuits reduce the liklihood of injury

* Dams (clustered with low, head) are more deadly 

* Rivers tend to become more lethal as the water level increases

* Rivers tend to become more lethal as their difficulty increases

* As paddler experience increases, the liklihood of fatality decreases

* Age (above 10 years old), type of watercraft, and being on a commercial trip do not change the prediction of a fatality

* 84% of the reported accidents where the victim is less than 18 years old are fatal

* Be weary of the "first major rapid" on any run
 
# Further
* Further modification of the tokenization, lemmatization, and vectorization could improve the models.
* More models could be tried, such as a MLP
* Stay safe out there!
 
 
![alt text](/images/screenshots/lochsa.jpg "The author enjoying a high water lap on the Lochsa, 2018")
