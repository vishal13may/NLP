# Natural Language Processing Exercises
## Naive Bayes Classifier
**Directory: ** Naive Bayes - Classify Email as Spam or Ham 
A simple Naive Bayes classifier implemented to classify emails as either spam or ham.
* A total of 9537 emails classified as ham, and 7495 emails classified as spam were used for training
* 5195 unseen emails were used as a test data

### Performance
```
1. Performance on the development data with 100% of the training data
1a. spam precision: 0.993088194636439
1b. spam recall: 0.9774149659863945
1c. spam F1 score: 0.9851892484914975
1d. ham precision: 0.9467265725288831
1e. ham recall: 0.9833333333333333
1f. ham F1 score: 0.9646827992151734

2. Description of enhancements tried:
A) Removed the speacial characters like [ . , { } ? ] ) (  etc.
B) Removed the stop words
C) Captured top 100 high frequency words from both Ham and Spam and deleted the common words between these two lists from dictionary/vocabulary. 
   In short, removed high frequency common words between HAM and SPAM.

4. Best performance results based on enhancements. Note that these could be the same or worse than the standard implementation.
4a. spam precision: 0.9945175438596491
4b. spam recall: 0.9872108843537415
4c. spam F1 score: 0.9908507442305066
4d. ham precision: 0.9692206941715783
4e. ham recall: 0.9866666666666667
4f. ham F1 score: 0.9778658738024447
```

## Perceptron Classifier
A standard and averaged perceptron classifier implemented to classify emails as either spam or ham.
* A total of 9537 emails classified as ham, and 7495 emails classified as spam were used for training
* 5195 unseen emails were used as a test data

### Performance
```
1. Performance of standard perceptron on the development data with 100% of the training data
1a. spam precision: 0.9861299972803916
1b. spam recall: 0.9866666666666667
1c. spam F1 score: 0.9863982589771491
1d. ham precision: 0.9672897196261683
1e. ham recall: 0.966
1f. ham F1 score: 0.9666444296197465

2. Performance of averaged perceptron on the development data with 100% of the training data
2a. spam precision: 0.9850665218571817
2b. spam recall: 0.9872108843537415
2c. spam F1 score: 0.9861375373742866
2d. ham precision: 0.9684986595174263
2e. ham recall: 0.9633333333333334
2f. ham F1 score: 0.9659090909090909
```

## Sequence Labeling
Assigned dialogue act tags to sequences of utterances in conversations from corpus.
* Used the CRFSuite toolkit, created a set of features, and measured the accuracy of CRFSuite model using these features.
* Assigned dialogue act tags to a set of unlabeled data.
* Created 2 versions - baseline CRF and advanced CRF (Advanced CRF uses all the features from baseline CRF, as well as uses more features).

### Feature set
#### Baseline feature set
* First Utterence of dialogue
* Speaker change from previous utterence
* All tokens from utterence
* All part of speech tags from utterence
#### Advanced feature set
* A feature combining previous word and current word as one token
* A feature combining previous word POS and current word POS
* A feature showing total length of tokens in given utterence
* A feature showing First TOKEN, First POS of utternce
* A feature showing Last TOKEN, Last POS of utterence
* A feature showing Last utterence of Dialogue
* A feature showing if utternce has no tokens. e.g when \<LAUGHTER\>
#### More features tried
* A feature combining current word and next word as one token
* A feature combining current word POS and current word POS
* A feature combining previous word, current word, and next word as token
* A feature combining previous POS, current POS, and next POS
* A feature showing last last word of previous utterence
* A feature showing ? and ! symobols/token
* Removing stop words/tokens
* Marking word/token as title and digits

### Performance
* Accuracy of baseline features was: 72.67619200513727
* Accuracy of advanced features was: 74.78527853588056
