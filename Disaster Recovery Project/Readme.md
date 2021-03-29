   # Disaster Response Pipeline Project
   
   This project is part of Udacity's nanodegree program. 


   https://view6914b2f4-3001.udacity-student-workspaces.com

## 1. Required Libraries: ##

   Besides the libraries included in the Anaconda distribution for Python 3.6 the following libraries have been included in this project:

   nltk

   sqlalchemy


## 2. Introduction ##

   Figure 8 helps companies transform they data by providing human annotators and machine learning to annotate data at all scales. Disaster response is one of events that        greatly benefits from data and machine learning modeling. In this project I propose an approach to social media messages annotation. NLP allows the extraction of great        significance in text, understanding how a model classifies and predicts needed responses in disaster cases provides good understanding of the power of words in functional    responses.

   In this project I will be using a data set containing messages that were sent during disaster events and build a classifier to identify messages or events in need of          attention or relief. The data cleaning and model building will be using pipelines, automating repetitive steps, and preventing data leakage.

   The best performing machine learning model will be deployed as a web app where the user can test their own tentative messages to see how they would be classified with the    models I selected and trained. Through the web app the user can also consult visualizations of the clean and transformed data.


## 3. Files ##

   Data was downloaded from Figure 8.


## 4. ETL Pipeline ##

   File data/process_data.py contains data cleaning pipeline that:

   Loads the messages and categories dataset
   Merges the two datasets
   Cleans the data
   Stores it in a SQLite database


## 5. ML Pipeline ##

   File models/train_classifier.py contains machine learning pipeline that:

   Loads data from the SQLite database
   Splits the data into training and testing sets
   Builds a text processing and machine learning pipeline
   Trains and tunes a model using GridSearchCV
   Outputs result on the test set
   Exports the final model as a pickle file


## 6. Flask Web App ##


   Run the following commands in the project's root directory to set up your database and model.

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
   
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
    - To run ML pipeline that trains classifier and saves
    
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`


2. Run the following command in the app's directory to run your web app.
         `python run.py`


3. Go to http://0.0.0.0:3001/


   Notebooks ETL Pipeline Prep.ipynb - jupyter notebook for data exploration and cleaning ML Pipeline Preparation - jupyter notebook for model selection and evaluation

<B>As far as the imbalanced dataset concerns, I would like to point out the drawbacks when a model is trained with imbalanced dataset. </b>

1. First of all, your model might perform efficiently as expected even when it is fed with an imbalanced dataset.

2. The model evaluates the data points in a biased manner where it might naturally select the majority class holding maximum data points.

3. In order to improve F1 score, precision and recall factors, I would suggest balancing the data in the data preparation phase.

4. Try to tune the class weights so that the model doesn't behave in a biased way. 

There are different techniques that could be implemented in order to avoid model resulting biased outputs. This completely should be analysed in the initial phases of data preparation and cleaning. 

## 7. Classification Output ##

<img width="881" alt="class1" src="https://user-images.githubusercontent.com/45115437/112853413-73787600-90ca-11eb-9631-93fe678cc7c2.png">

<img width="860" alt="class2" src="https://user-images.githubusercontent.com/45115437/112853200-3d3af680-90ca-11eb-899d-7975fe0ec2ad.png">

<img width="862" alt="class3" src="https://user-images.githubusercontent.com/45115437/112853271-4fb53000-90ca-11eb-8995-affe8a5675b4.png">

## 8. Graphs ##

<img width="857" alt="class1" src="https://user-images.githubusercontent.com/45115437/112853794-d66a0d00-90ca-11eb-8c7f-ce26903c57e6.png">

<img width="873" alt="class2" src="https://user-images.githubusercontent.com/45115437/112853820-e08c0b80-90ca-11eb-88dd-698c4d83a15d.png">

<img width="875" alt="class3" src="https://user-images.githubusercontent.com/45115437/112853862-ea157380-90ca-11eb-9fae-80ea3d6249ff.png">
