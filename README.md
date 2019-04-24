Overview

Sentiment analysis involves utilizing customer reviews to predict feelings.  The process can be supervised or unsupervised.  Whether supervised or unsupervised a number of key preparatory steps must be followed.  

Configuring the Data for Machine Learning

Text data contains rich information for statistical models to train on.  Before training words must be converted into numbers.  To build a consistent dataset the modeler must lower case words, trim leading and trailing spaces and lower case all words.  Next, words must be reduced to their root meaning.  For example, running and run have the same intention and should be counted as one word (one meaning).  Lemmatizing words is the process of reducing a words in past, present or future tense to one word.  Lastly, words are vectorized.  In this case study I return a count of words per sentiment.

The Data

In this exercise, I utilize a database of movie reviews found on Kaggle.com.  The dataset consists of 156,060 rows of reviewer phrases and the review magnitude scaled from 0 for a negative review up to 4 for a positive review.  The shear volume of data, breadth and depth, made it necessary to randomly shuffle rows and return the first 25000 to a feature matrix.  Thirty-three percent of rows went to the test data set with the balance going to training.

The Models

Training, testing and evaluating several models allows the scientist the flexibility to choose from several qualities: predictive quality and time to build and train being two important factors.  Random forest, adaboost and naive bayes models were trained and tuned for optimal performance.


Banik, Rounak (2019). Sentiment Analysis on Movie Reviews. https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews
