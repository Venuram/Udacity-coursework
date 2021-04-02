<h1> <b> Quora Question Pair Similarity - Udacity's Nanodegree Capestone Project </b> </h1> 

https://exploratory-data.blogspot.com/2021/04/quora-question-pair-similarity.html

Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term. so main aim of project is that predicting whether pair of questions are similar or not. This could be useful to instantly provide answers to questions that have already been answered.

<b> Credits: </b> Kaggle

<b> <h2> Problem Statement: </h2> </b>

Identify which questions asked on Quora are duplicates of questions that have already been asked.

<b> <h2> Real world/Business Objectives and Constraints : </h2> </b>

1. The cost of a mis-classification can be very high.

2. You would want a probability of a pair of questions to be duplicates so that you can choose any threshold of choice.

3. No strict latency concerns.

4. Interpretability is partially important.

<b> <h2> Performance Metric:</h2> </b>

1. log-loss

2. Binary Confusion Matrix

<b> <h2>Data Overview:</h2></b>

Data will be in a file Train.csv

Train.csv contains 5 columns : qid1, qid2, question1, question2, is_duplicate

Size of Train.csv - 60MB

Number of rows in Train.csv = 404,290

We are given a minimal number of data fields here, consisting of:

<b> id:</b> Looks like a simple rowID

<b>qid{1, 2}:</b> The unique ID of each question in the pair

<b>question{1, 2}: </b> The actual textual contents of the questions.

<b> is_duplicate: </b> The label that we are trying to predict - whether the two questions are duplicates of each other.

<b> Findings: </b>

As we discussed in the objective part, the goal is to build a model that classifies the given input as existsing qus or new qus in the forum.

<b> <h3>Data Analysis: </h3></b>

Initially, we explored the given data from kaggle to have an outlook on the data distribution.

Overview of 1,50,000 duplicate question and 2,50,000 non-duplicates that exist.

Most Number of times a qus repeated: 157.

No.of unique qus in the forum: 537933

Number of unique questions that appear more than one time: 111780 (20%)

Since we didn't have much details/info to build a model apart from basic exploration, that's where we performed feature extraction.

<b> <h3> Feature Extraction: </h3></b>

This technique fetched us some new features that we created from the given quora question pair. Analysis on these features lead us to some of the important aspects that could possibly influence our model prediction results.

1. q1_len
  
2. q2_len

3. q1_words

4. q2_words

5. words_total

6. words_common

7. words_shared

8. cwc_min

9. cwc_max

10. csc_min

11. csc_max

12. ctc_min

13. ctc_max

14. last_word_eq

15. first_word_eq

16. num_common_adj

17. num_common_prn

18. num_common_n

19. fuzz_ratio

20. fuzz_partial_ratio

21. token_sort_ratio

22. token_set_ratio

23. mean_len

24. abs_len_diff

25. longest_substr_ratio

<b> Important Feature: </b> "token_sort_ratio" had some linearly separable characterstics that was evident in the distribution plotted.

<b> Model creation: </b>

For two out of three models, we have used the same technique of implementing SGD() classifier with hyper-parameter tuning to estimate the best alpha value resulting in low log loss error.

<b> Models Used: </b>

<b> Logistic Regression on 100K data with hyper parameter tuning: </b>

Best alpha: 0.01

Log-loss: 0.44

<b> SVM on 100K data with hyper parameter tuning: </b>

Best alpha: 0.01

Log-loss: 0.48

<b> XGBoost on 100K data with no hyper parameter tuning: </b>

Best alpha: 0.02

Log-loss: 0.35

As an end result, XGBoost algorithm resulted with significantly low log loss value while classifying the inputs given.

<b> <h3> Conclusion: </h3> </b>

Overall, I chose this challenging project because that this data preparation has so many NLP techniques involved right from tokenization, text based feature extraction to converting texts into numerics for model creation, and always this domain seems to appear as the future advancement.

Finally for this capestone project, we conclude that on low dimension data,we will use 'XGBoost' model and for high dimension data we will use either 'Linear SVM' or 'Logistic Regression'

<b> Licensing, Authors, Acknowledgements, etc. </b>

Data for coding project was provided by kaggle.







