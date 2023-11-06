import matplotlib.pyplot as plt
import nltk
nltk.download('wordnet')

reviews_train = []
for line in open('full_train.txt', 'r', encoding='utf-8'):
    reviews_train.append(line.strip())

reviews_test = []
for line in open('full_test.txt', 'r', encoding='utf-8'):
    reviews_test.append(line.strip())

import re
REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\\[\\]]")
REPLACE_WITH_SPACE = re.compile("(<br\\s*/><br\\s*/>)|(\\-)|(\\/)")

def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    return reviews

reviews_train_clean = preprocess_reviews(reviews_train)
reviews_test_clean = preprocess_reviews(reviews_test)

def get_lemmatized_text(corpus):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]

reviews_train_clean = get_lemmatized_text(reviews_train_clean)
reviews_test_clean = get_lemmatized_text(reviews_test_clean)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
stop_words = ['in', 'of', 'at', 'a', 'the']
tfidf_vectorizer = CountVectorizer(binary = True, ngram_range = (1,3), stop_words = stop_words)
tfidf_vectorizer.fit(reviews_train_clean)
X = tfidf_vectorizer.transform(reviews_train_clean)
X_test = tfidf_vectorizer.transform(reviews_test_clean)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

target = [1 if i < 12500 else 0 for i in range(25000)]
X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = 0.75)

# accuracy = []
# regularization = [0.01, 0.05, 0.25, 0.5, 1]
# for c in regularization:
#     lr = LinearSVC(C = c)
#     lr.fit(X_train, y_train)
#     accuracy.append(accuracy_score(y_val, lr.predict(X_val)))
#     print("Độ chính xác với C = %s: %s" % (c, accuracy_score(y_val, lr.predict(X_val))))


# highestC = [regularization[i] for i in range(len(regularization)) if accuracy[i] == max(accuracy)][0]
final_model = LinearSVC(C = 2, dual = True, max_iter = 1000)
final_model.fit(X, target)
y_pred = final_model.predict(X_test)
print("Độ chính xác cuối: %s" % accuracy_score(target, y_pred))
# print(str(highestC))
# accuracy.append(accuracy_score(target, final_model.predict(X_test)))
# regularization.append('testError C:' + 1)

from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(y_pred, target).ravel()
print(str((tn, fp, fn, tp)))