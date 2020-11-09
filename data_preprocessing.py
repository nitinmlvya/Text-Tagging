import pandas as pd
import numpy as np
import spacy

class DataPreprocess:
    def __init__(self):
        self.CORPUS_CSV = r'conversation_data.csv'
        self.keep_single_word_pos = ['ADJ', 'NOUN', 'NUM', 'PROPN', 'VERB']
        self.nlp = spacy.load("en_core_web_sm")
        self.df = {}

    def load_csv(self):
        self.df = pd.read_csv(self.CORPUS_CSV)
        # print(self.df.head())
        print('Raw data is loaded in CSV format.')

    def test_fill_df(self, df):
        self.df = df

    def clean_conversation(self):
        print('Started Data cleaning.')
        # Step 1: Split user, numbers and actual conversation and creates seperate columns for them
        self.df[['user', 'number1', 'number2', 'conversation']] = \
            self.df['conversation'].str.split(' ', 3, expand=True)

        # Remove [silence], [laughter] etc
        self.df['conversation'] = self.df['conversation'].str.replace(r'\[(.*?)\]', '')

        # drop NULL converations
        self.df['conversation'].replace('', np.nan, inplace=True)
        self.df.dropna(inplace=True)

        # Drop duplicate rows
        self.df.drop_duplicates(subset=['conversation'], inplace=True)
        # print(self.df.head())
        print('Data cleaning is done.')

    def extract_meaning_phrases(self):
        # Step 2
        print('Started extracting meaning phrases from conversations.')
        def __filter_conversation(text):
            filtered_text = []
            text = self.nlp(text)
            for phrase in text.noun_chunks:
                if len(phrase.text.split()) == 1:
                    if phrase[0].pos_ in self.keep_single_word_pos:
                        filtered_text.append(phrase[0].text)
                else:
                    filtered_text.append(phrase.text)
            return filtered_text if filtered_text else np.nan
        # filter_conversation("uh i vaguely recall but i've never gone and and done any study on it since the S and L thing that teapot dome scandal is when i think during the Hoover years")

        self.df['conversation'] = self.df['conversation'].astype(object)
        self.df['conversation'] = self.df['conversation'].apply(__filter_conversation)
        self.df.dropna(subset=['conversation'], inplace=True)
        # print(self.df.head())
        print('Extracting meaning phrases from conversations is done.')

    def group_convs_by_file_id(self):
        # Step 3
        self.df = self.df.groupby('file_id', as_index=False).agg({'conversation': 'sum', 'topic': 'first'})
        # print(self.df.head())

    def test_group_convs(self):
        a = self.df['conversation'].sum()
        self.df = pd.DataFrame({'conversation': []})
        self.df['conversation'] = [a]

    def rm_dups_phrases_in_same_conv(self):
        # Step 4
        self.df['conversation'] = self.df['conversation'].apply(lambda x: list(set(x)))
        print('conversation')
        # self.df['conv_length'] = self.df['conversation'].apply(lambda x: len(x))
        # max_length = max(self.df.sort_values('conv_length', ascending=False)['conv_length'])
        # print(self.df.head())

    def get_X_y(self):
        # Step 5
        X = self.df['conversation'].tolist()
        X = [' '.join(a).strip() for a in X]
        y = self.df['topic'].tolist()
        print('X and y are created')
        return X, y

    def test_get_X(self):
        X = self.df['conversation'].tolist()
        X = [' '.join(a).strip() for a in X]
        print('X is created')
        return X

    def get_topic_list(self):
        return self.df['topic'].nunique()