from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from build_features import BuildFeatures
from data_preprocessing import DataPreprocess
import pickle
import numpy as np
np.random.seed(25)

class TrainModel:
    def __init__(self):
        self.model_path = r'models/svc_model.pkl'
        self.data_preprocessing = DataPreprocess()
        self.build_features = BuildFeatures()
        self.X = None
        self.y = None

    def run(self):
        # data preprocessing pipeline
        self.data_preprocessing.load_csv()
        self.data_preprocessing.clean_conversation()
        self.data_preprocessing.extract_meaning_phrases()
        self.data_preprocessing.group_convs_by_file_id()
        self.data_preprocessing.rm_dups_phrases_in_same_conv()
        self.X, self.y = self.data_preprocessing.get_X_y()

        # with open('X.pkl', 'rb') as fp:
        #     self.X = pickle.load(fp)
        # self.X = [list(a) for a in self.X]
        #
        # with open('y.pkl', 'rb') as fp:
        #     self.y = pickle.load(fp)

        # Train and test set
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.1,
                                                            stratify=self.y)

        # build features
        # oversampling on training data only
        X_train, y_train = self.build_features.oversampling_on_training_data(X_train, y_train)

        # X_train = [' '.join(a).replace('[PAD]', '').strip() for a in X_train]
        # X_test = [' '.join(a).replace('[PAD]', '').strip() for a in X_test]

        # Word to vectors
        self.build_features.word_to_vectors_model(X_train)
        X_train = self.build_features.word_to_vectors_transformed(X_train)
        X_test = self.build_features.word_to_vectors_transformed(X_test)

        # Dimenstion reduction technique.
        self.build_features.dimension_reduction_model(X_train)
        X_train = self.build_features.dimension_reduction_transformed(X_train)
        X_test = self.build_features.dimension_reduction_transformed(X_test)

        # train model
        model = LinearSVC(random_state=25)
        model.fit(X_train, y_train)
        print('\n\n')
        print('-*-' * 20)
        print('Training accuracy: ', model.score(X_train, y_train) * 100)
        print('Accuracy on unseen documents: ', model.score(X_test, y_test) * 100)
        print('-*-' * 20)
        pickle.dump(model, open(self.model_path, 'wb'))  # save


if __name__ == '__main__':
    TrainModel().run()




