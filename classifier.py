from build_features import BuildFeatures
from data_preprocessing import DataPreprocess
import argparse
import pandas as pd
import pickle


class Classifier:
    def __init__(self):
        parser = argparse.ArgumentParser(description='List the content of a folder')
        parser.add_argument('--text_file', type=str, help='File path to classify')
        args = parser.parse_args()
        self.text_file_path = args.text_file
        self.df = None
        self.X = None

        self.word_to_vector_model_path = r'models/w2v.pkl'
        self.dim_reduction_path = r'models/dim_reduction.pkl'
        self.model_path = r'models/svc_model.pkl'
        self.data_preprocessing = DataPreprocess()
        self.build_features = BuildFeatures()

    def read_text_file(self):
        with open(self.text_file_path) as fp:
            text = [x.strip('\r\n') for x in fp.readlines()]
            return text

    def create_dataframe(self, text):
        self.df = pd.DataFrame(text, columns=['conversation'])

    def load_model(self):
        return pickle.load(open(self.model_path, 'rb'))


    def run(self):
        text = self.read_text_file()
        self.create_dataframe(text)
        # data preprocessing pipeline
        self.data_preprocessing.test_fill_df(self.df)
        self.data_preprocessing.clean_conversation()
        self.data_preprocessing.extract_meaning_phrases()
        self.data_preprocessing.test_group_convs()
        self.data_preprocessing.rm_dups_phrases_in_same_conv()
        X_test = self.data_preprocessing.test_get_X()
        print(len(X_test))

        # Word to vectors
        X_test = self.build_features.word_to_vectors_transformed(X_test)
        # Dimenstion reduction technique.
        X_test = self.build_features.dimension_reduction_transformed(X_test)

        model = self.load_model()
        print('-*-' * 20)
        predicted_class = model.predict(X_test)
        print('Result: ', predicted_class)


if __name__=='__main__':
    Classifier().run()