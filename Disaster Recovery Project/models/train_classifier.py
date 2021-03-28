import sys
# import libraries

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
import pickle
import nltk
nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    
    ''' This function fetches the file to be processed from the database and perform data preparation
        to feed the same to the model
        
        Input:
        database_filepath: Path of the file from database
        
        Output:
        X: Holds the message column from df dataframe
        Y: Holds the transformed categories' columns
        category_names: categories of the dataframe '''
    
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_messages1', con=engine)
    
    X = df['message'] # Placing the message column
    Y = df.drop(['message', 'genre', 'id', 'original'], axis = 1) # Holding the transformed categories' columns
    category_names = Y.columns.tolist()
    
    return X,Y, category_names


def tokenize(text):
    
    ''' This function tokenize the text that's been fed as input to the model
    
        Input:
        
        text: Text needed for processing and model feed
        
        Output:
        
        clean_tokens: cleaned tokens/words from the text served as input''' 
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    #clean_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    
    ''' Function perform creation of Pipeline for the purpose of combining different tokenizers and classifier together for efficient model usage
    
        Output:
        cv: returns cross validation parameters for model performance '''
    
    pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(AdaBoostClassifier()))    
                ])
    # Involving paramters to make the search/processing better using GridSearch feature and storing it in CV.

    parameters =  {'clf__estimator__learning_rate': [0.05],
                   'clf__estimator__n_estimators': [100, 200, 300]} 
    
    cv = GridSearchCV(estimator=pipeline, cv=2, scoring='f1_weighted', verbose=3, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
    ''' This method functions for the purpose of fetching the successfully ran model parameters and evaluate the model's performance
    
        Input:
        model: model built in the previos method
        X_test: test dataframe with respect to X
        Y_test: test dataframe with respect to Y 
        categories_names: names of categories listed above
        
        Output:
        Classification report of the model's performance '''
    
    y_pred = model.predict(X_test)
    y_pred_pd = pd.DataFrame(y_pred, columns = category_names)
    
    
    print(classification_report(Y_test, y_pred, target_names = category_names))


def save_model(model, model_filepath):
    
    ''' This function saves the model to the pickle file to use the same with the web app created via flask '''
    
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
