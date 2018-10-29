import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from nltk.stem.snowball import SnowballStemmer
from sklearn.svm import LinearSVC
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import math


###############################################################################
data_train = 0
data_test = 0
X_train = 0
y_train = 0
X_test = 0
y_test = 0
train_vec = 0
train_y = 0
test_vec = 0
test_y = 0

nb_scores = list()
svm_scores = list()
knn_scores = list()
tree_scores = list()

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
              'talk.politics.misc',
              ]


A = ['Naive Bayes', 'SVM', 'KNN', 'Decision Tree']
B = [20, 16, 12, 8, 4]


def remove_noise(sentence):
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    lemmatizer = WordNetLemmatizer()
    stopword_set = set(stopwords.words('english'))
    wordlist = re.sub(r"\n|(\\(.*?){)|}|[!$%^&*#()_+|~\-={}\[\]:\";'<>?,.\/\\]|[0-9]|[@]", ' ', sentence)
    wordlist = re.sub('\s+', ' ', wordlist)  # remove extra space
    wordlist_normal = [stemmer.stem(word.lower()) for word in wordlist.split()]  # stemming
    wordlist_normal = [lemmatizer.lemmatize(word, pos='v') for word in wordlist_normal]  # lemmatization
    wordlist_clean = [word for word in wordlist_normal if word not in stopword_set]  # remove stopwords
    result = ' '.join(wordlist_clean)
    return result


def benchmark(s, clf):

    print('_' * 80)
    print("Training: ")
    print(s)
    if s == 'Naive Bayes':

        text_mnb_stemmed = clf.fit(data_train.data, data_train.target)
        # print(type(text_mnb_stemmed))
        predicted_mnb_stemmed = text_mnb_stemmed.predict(data_test.data)

        score = np.mean(predicted_mnb_stemmed == data_test.target)
        print("f1-score:   %0.3f" % score)
        nb_scores.append(score)

    elif s == 'knn':
        try:
            clf.fit(X_train, y_train)
        except:
            clf.fit(X_train.toarray(), y_train)

        try:
            pred = clf.predict(X_test)
        except:
            pred = clf.predict(X_test.toarray())

        score = metrics.f1_score(y_test, pred, average='micro')
        print("f1-score:   %0.3f" % score)
        knn_scores.append(score)

    else:
        try:
            clf.fit(train_vec, train_y)
        except:
            clf.fit(train_vec.toarray(), train_y)

        try:
            pred = clf.predict(test_vec)
        except:
            pred = clf.predict(test_vec.toarray())

        score = metrics.f1_score(test_y, pred, average='micro')
        print("f1-score:   %0.3f" % score)
        if s == 'svm':
            svm_scores.append(score)
        elif s == 'Tree':
            tree_scores.append(score)


for index in range(len(A)):
    for idx in range(len(B)):
        ctg = categories[:B[idx]]
        test_size_ratio = 0.2
        data_Xy = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'),
                                     categories=ctg)
        category_names = data_Xy.target_names  # text names of all categories
        train_X, test_X, train_y, test_y = train_test_split(data_Xy.data, data_Xy.target, test_size=test_size_ratio,
                                                            stratify=data_Xy.target)
        print("Training set size: %8d\tTest set size: %8d" % (len(train_X), len(test_X)))

        train_X_clean = map(remove_noise, train_X)
        test_X_clean = map(remove_noise, test_X)

        vectorizer = TfidfVectorizer(stop_words='english', min_df=5, max_df=0.95)
        train_vec = vectorizer.fit_transform(train_X_clean)
        test_vec = vectorizer.transform(test_X_clean)
        print(train_vec.shape, test_vec.shape)

        ###############################################################################
        data_train = fetch_20newsgroups(subset='train', categories=ctg, remove=('headers', 'footers', 'quotes'),
                                        shuffle=True)

        data_test = fetch_20newsgroups(subset='test', categories=ctg, remove=('headers', 'footers', 'quotes'),
                                       shuffle=True)

        y_train, y_test = data_train.target, data_test.target

        stemmer = SnowballStemmer("english", ignore_stopwords=True)


        class StemmedCountVectorizer(CountVectorizer):

            def build_analyzer(self):
                analyzer = super(StemmedCountVectorizer, self).build_analyzer()
                return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


        stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

        X_train_clean = map(remove_noise, data_train.data)
        X_test_clean = map(remove_noise, data_test.data)

        vectorizer = TfidfVectorizer(stop_words='english', min_df=5, max_df=0.95)  # , min_df=5, max_df=0.95
        X_train = vectorizer.fit_transform(X_train_clean)
        X_test = vectorizer.transform(X_test_clean)

        if index == 2:

            print('=' * 80)
            print("KNN For categories : ", B[idx])

            s = 'knn'

            benchmark(s, KNeighborsClassifier(n_neighbors=767))  # 768

        if index == 0:

            print('=' * 80)
            print("Naive Bayes For categories : ", B[idx])

            s = 'Naive Bayes'

            benchmark(s, MultinomialNB(alpha=1, fit_prior=False))

        if index == 3:

            print('=' * 80)
            print("Decision Tree For categories : ", B[idx])

            s = 'Tree'

            benchmark(s, DecisionTreeClassifier(criterion="gini", splitter="best"))

        if index == 1:

            print('=' * 80)
            print("SVM for categories : ", B[idx])

            s = 'svm'

            benchmark(s, LinearSVC(loss='hinge', penalty='l2', tol=1e-2, max_iter=100, random_state=50))


print("Naive Bayes : ", nb_scores)
print('SVM : ', svm_scores)
print('KNN : ', knn_scores)
print('TREE : ', tree_scores)
# categories = None


scores = (nb_scores, svm_scores, knn_scores, tree_scores)

for20 = list()
for12 = list()
for8 = list()
for6 = list()
for4 = list()

for i in range(5):
    if i == 0:
        for20.append(nb_scores[i])
        for20.append(svm_scores[i])
        for20.append(knn_scores[i])
        for20.append(tree_scores[i])
    elif i == 1:
        for12.append(nb_scores[i])
        for12.append(svm_scores[i])
        for12.append(knn_scores[i])
        for12.append(tree_scores[i])
    elif i == 2:
        for8.append(nb_scores[i])
        for8.append(svm_scores[i])
        for8.append(knn_scores[i])
        for8.append(tree_scores[i])
    elif i == 3:
        for6.append(nb_scores[i])
        for6.append(svm_scores[i])
        for6.append(knn_scores[i])
        for6.append(tree_scores[i])
    elif i == 4:
        for4.append(nb_scores[i])
        for4.append(svm_scores[i])
        for4.append(knn_scores[i])
        for4.append(tree_scores[i])


def plotting(scores, label):
    n_groups = 5

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
    ax.set_ylabel('F-Scores', fontweight='bold')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(('20', '16', '12', '8', '4'), fontweight='bold')
    ax.legend()

    labels = [i for i in scores]

    for rect, label in zip(result, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.00 * height, round(label * 100, 2),
                ha='center', va='bottom')

    fig.tight_layout()
    plt.show()

def plotting2(scores, label):
    n_groups = 4

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
    ax.set_ylabel('F-Scores', fontweight='bold')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(('Naive Bayes', 'SVM', 'KNN', 'Tree'), fontweight='bold')
    ax.legend()

    labels = [i for i in scores]

    for rect, label in zip(result, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.00 * height, round(label * 100, 2),
                ha='center', va='bottom')

    fig.tight_layout()
    plt.show()


plotting(nb_scores, "Naive Bayes")
plotting(svm_scores, "SVM")
plotting(knn_scores, 'KNN')
plotting(tree_scores, 'Decision Tree')

plotting2(for20, "20 Newsgroup")
plotting2(for12, "16 Newsgroup")
plotting2(for8, "12 Newsgroup")
plotting2(for6, "8 Newsgroup")
plotting2(for4, "4 Newsgroup")