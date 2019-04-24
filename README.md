### Overview

Sentiment analysis involves utilizing customer reviews to predict feelings.  The process can be supervised or unsupervised.  Whether supervised or unsupervised a number of key preparatory steps must be followed.  

### Configuring the Data for Machine Learning

Text data contains rich information for statistical models to train on.  Before training, words must be converted into numbers.  To build a consistent dataset the modeler must lower case words, trim leading and trailing spaces and lower case all words.  Next, words must be reduced to their root meaning.  For example, running and run have the same intention and should be counted as one word (one meaning).  Lemmatizing words is the process of reducing words in past, present or future tenses to one word.  Lastly, words are vectorized.  In this case study I return a count of words per sentiment.

### The Data

In this exercise, I utilize a database of movie reviews found on Kaggle.com.  The dataset consists of 156,060 rows of reviewer phrases and the review magnitude scaled from 0 for a negative review up to 4 for a positive review.  The shear volume of data, breadth and depth, made it necessary to randomly shuffle rows and return the first 25000 to a feature matrix.  Thirty-three percent of rows went to the test data set with the balance going to training.

### The Models

Training, testing and evaluating several models allows the scientist the flexibility to choose from several qualities: predictive quality and time to build and train being two important factors.  Random forest, adaboost and naive bayes models were trained and tuned for optimal performance.

Both the random forest and ada boost models were supplied to a comprehensive grid search.  An n_estimator list of 50, 100, 150, 200, 250, and 300 was supplied to the grid search for both the random forest and ada boost models.  In addition, the min samples per leaf was restricted to a list of 3, 4, 5 and 6 (random forest).

### Results

The random forest model achieved an accuracy of 57.8 percent, naive Bayes an accuracy of 55 percent and ada boost model an accuracy of 52 percent.  The optimal random forest model used 300 estimators and 6 samples per leaf.  The adaboost model achieved its best accuracy at 50 trees.


### Sources
Banik, Rounak (2019). Sentiment Analysis on Movie Reviews. https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews
