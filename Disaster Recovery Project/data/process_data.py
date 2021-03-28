import sys
# import libraries

import sys
import os
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(messages_filepath, categories_filepath):
    
    ''' This method functions for the purpose of fetching input files and merge for data processing
    
        Input:
        message_filepath: disaster_messages.csv file
        categories_filepath: disaster_categories.csv file
        
        Output:
        df: merged file for data processing '''
        
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    print('Messages dataframe')
    print(messages.head(1))
    print('Categories dataframe')
    categories.head()
    
    df = pd.merge(messages, categories, on='id', how='outer')
    
    return df


def clean_data(df):
    
    ''' This method functions for the purpose of cleaning the merged file.
    
        Input:
        df: merged dataframe
        
        Output:
        df: cleaned dataframe ( categorize columns, remove duplicates etc.) '''
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    row = categories.head(1)
    
    category_colnames = row.applymap(lambda x: x[:-2]).iloc[0, :].tolist()
    categories.columns = category_colnames
    
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    df.drop('categories', axis=1, inplace=True)
    
    df = pd.concat([df,categories], axis=1)
    
    df.drop('related', axis=1, inplace=True)
    
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    
    ''' save_data serves by storing the dataframe in the database 
    
        Input:
        df: cleaned dataframe
        database_filename: Database to which the dataframe is stored
        
        Output:
        Storing dataframe in the database '''
    
    
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('messages', engine, index = False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
