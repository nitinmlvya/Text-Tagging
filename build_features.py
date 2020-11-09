import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


class BuildFeatures:
    def __init__(self):
        self.word_to_vector_model_path = r'models/w2v.pkl'
        self.dim_reduction_path = r'models/dim_reduction.pkl'

    def oversampling_on_training_data(self, X, y):
        df = pd.DataFrame({'X': X, 'y': y})
        print('Before oversampling: ', df.shape)

        max_size = df['y'].value_counts().max()
        lst = [df]
        for class_index, group in df.groupby('y'):
            lst.append(group.sample(max_size - len(group), replace=True))
        df = pd.concat(lst)
        print('After oversampling: ', df.shape)

        return df['X'].tolist(), df['y'].tolist()

    def word_to_vectors_model(self, X):
        # print(X)
        tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(1, 2), stop_words='english')
        model = tfidf.fit(X)
        pickle.dump(model, open(self.word_to_vector_model_path, 'wb')) # save

    def word_to_vectors_transformed(self, X):
        model = pickle.load(open(self.word_to_vector_model_path, 'rb'))
        return model.transform(X).toarray()

    def dimension_reduction_model(self, X):
        pca = PCA(n_components=240)
        # print(pca.explained_variance_ratio_)
        model = pca.fit(X)
        pickle.dump(model, open(self.dim_reduction_path, 'wb')) # save

    def dimension_reduction_transformed(self, X):
        model = pickle.load(open(self.dim_reduction_path, 'rb'))
        return model.transform(X)
