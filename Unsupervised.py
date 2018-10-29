
from __future__ import print_function

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from nltk.stem.snowball import SnowballStemmer
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np

categories = ['alt.atheism',
              'talk.religion.misc',
              'comp.graphics',
              'sci.space',
              'comp.os.ms-windows.misc',
              'comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware',
              'comp.windows.x',
              'misc.forsale',
              'rec.autos',
              'rec.motorcycles',
              'rec.sport.baseball',
              'rec.sport.hockey',
              'sci.crypt',
              'sci.electronics',
              'sci.med',
              'soc.religion.christian',
              'talk.politics.guns',
              'talk.politics.mideast',
              'talk.politics.misc',]
def remove_noise(sentence):
    result = ''
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    lemmatizer = WordNetLemmatizer()
    stopword_set = set(stopwords.words('english'))
    wordlist = re.sub(r"\n|(\\(.*?){)|}|[!$%^&*#()_+|~\-={}\[\]:\";'<>?,.\/\\]|[0-9]|[@]", ' ', sentence) # remove punctuation
    wordlist = re.sub('\s+', ' ', wordlist) # remove extra space
    wordlist_normal = [stemmer.stem(word.lower()) for word in wordlist.split()] # restore word to its original form (stemming)
    wordlist_normal = [lemmatizer.lemmatize(word, pos='v') for word in wordlist_normal] # restore word to its root form (lemmatization)
    wordlist_clean = [word for word in wordlist_normal if word not in stopword_set] # remove stopwords
    result = ' '.join(wordlist_clean)
    return result
# Uncomment the following to do the analysis on all the categories
# categories = None


A = ['K-MinIBatch', 'KMeans']
B = [20, 10, 4]
kmeans_homogeneity = list()
kmini_homogeneity = list()


for i in range(1):

    for index in range(len(B)):

        ctg = categories[:B[index]]
        dataset = fetch_20newsgroups(subset='all', categories=ctg)


        labels = dataset.target
        true_k = np.unique(labels).shape[0]
        print("k :", true_k)

        print("Extracting features from the training dataset using a sparse vectorizer")

        X_clean = map(remove_noise, dataset.data)

        vectorizer = TfidfVectorizer(max_df=0.5,
                                     min_df=2, stop_words='english',
                                     use_idf=True)

        X = vectorizer.fit_transform(X_clean)

        print("Performing dimensionality reduction using LSA")
        svd = TruncatedSVD(100)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        X = lsa.fit_transform(X)

        explained_variance = svd.explained_variance_ratio_.sum()
        print("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))

        kmini = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1000,
                                init_size=1000, batch_size=1000)

        kmeans = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=100)

        print("Clustering sparse data with %s" % kmini)

        kmini.fit(X)
        print("Kmini : ")
        print()

        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, kmini.labels_))
        kmini_homogeneity.append(metrics.homogeneity_score(labels, kmini.labels_))
        print("Completeness: %0.3f" % metrics.completeness_score(labels, kmini.labels_))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels, kmini.labels_))
        print("Adjusted Rand-Index: %.3f"
              % metrics.adjusted_rand_score(labels, kmini.labels_))
        print("Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(X, kmini.labels_, sample_size=1000))

        print()

        print("Clustering sparse data with %s" % kmeans)

        kmeans.fit(X)
        print("Kmeans : ")
        print()

        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, kmeans.labels_))
        kmeans_homogeneity.append(metrics.homogeneity_score(labels, kmeans.labels_))
        print("Completeness: %0.3f" % metrics.completeness_score(labels, kmeans.labels_))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels, kmeans.labels_))
        print("Adjusted Rand-Index: %.3f"
              % metrics.adjusted_rand_score(labels, kmeans.labels_))
        print("Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(X, kmeans.labels_, sample_size=1000))


def plotting(scores,label):
    n_groups = 3

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.25

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    result = ax.bar(index, scores, bar_width,
                    alpha=opacity, color='b',
                    error_kw=error_config,
                    label=label)

    ax.set_xlabel('Number of categories', fontweight='bold')
    ax.set_ylabel('Homogeneity ', fontweight='bold')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(('4', '10', '20'), fontweight='bold')
    ax.legend()

    labels = [i for i in scores]

    for rect, label in zip(result, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.00 * height, label,
                ha='center', va='bottom')

    fig.tight_layout()
    plt.show()


kmini_homogeneity=[46.99, 42.54, 62.22]

# plotting(kmeans_homogeneity, 'K-Means')
plotting(kmini_homogeneity, 'Mini Batch K-Means')


for20 = [44.2, 46.99]
for10 = []
for4 = [62.79, 62.22]

def plotting2(scores, label):
    n_groups = 2

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.25

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    result = ax.bar(index, scores, bar_width,
                    alpha=opacity, color='b',
                    error_kw=error_config,
                    label=label)

    ax.set_xlabel('classifiers', fontweight='bold')
    ax.set_ylabel('Homogeneity', fontweight='bold')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(('K-means', 'Mini Batch K-means'), fontweight='bold')
    ax.legend()

    labels = [i for i in scores]

    for rect, label in zip(result, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.00 * height, label,
                ha='center', va='bottom')

    fig.tight_layout()
    plt.show()


plotting2(for20, "20 Categories")
# plotting2(for10, "16 Newsgroup")
plotting2(for4, "4 Newsgroup")