# IRVIN BARHAM
# Sentiment Analysis with Naive Bayes Classifier
# Dataset Amazon Musical Reviews
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import pickle

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
seed = 4353

data = pd.read_csv('MUSICALREVIEWS.csv')
print(data.head())

data.columns = data.columns.str.lower()
data.columns

sns.countplot(data.overall)
plt.xlabel('Overall ratings')

# Replacing Overall Ratings with labels
data['sentiment'] = data.overall.replace({
    1: 'negative',
    2: 'negative',
    3: 'neutral',
    4: 'positive',
    5: 'positive'
})

X_data = data['reviewtext'] + ' ' + data['summary']
y_data = data['sentiment']

X_data = X_data.astype(str)

X_data_df = pd.DataFrame(data=X_data)
X_data_df.columns = ['review']
X_data_df.head()

string.punctuation
def final(X_data_full):

    # function for removing punctuations
    def remove_punct(X_data_func):
        string1 = X_data_func.lower()
        translation_table = dict.fromkeys(map(ord, string.punctuation), ' ')
        string2 = string1.translate(translation_table)
        return string2

    X_data_full_clear_punct = []
    for i in range(len(X_data_full)):
        test_data = remove_punct(X_data_full[i])
        X_data_full_clear_punct.append(test_data)

    # Removing Stopwords
    def remove_stopwords(X_data_func):
        pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
        string2 = pattern.sub(' ', X_data_func)
        return string2

    X_data_full_clear_stopwords = []
    for i in range(len(X_data_full)):
        test_data = remove_stopwords(X_data_full[i])
        X_data_full_clear_stopwords.append(test_data)

    # Will Tokenize CSV data with NLTK
    def tokenize_words(X_data_func):
        words = nltk.word_tokenize(X_data_func)
        return words

    X_data_full_tokenized_words = []
    for i in range(len(X_data_full)):
        test_data = tokenize_words(X_data_full[i])
        X_data_full_tokenized_words.append(test_data)

    # Lemmatize words for reviewtext
    lemmatizer = WordNetLemmatizer()

    def lemmatize_words(X_data_func):
        words = lemmatizer.lemmatize(X_data_func)
        return words

    X_data_full_lemmatized_words = []
    for i in range(len(X_data_full)):
        test_data = lemmatize_words(X_data_full[i])
        X_data_full_lemmatized_words.append(test_data)

    # creating the bag of words model
    cv = CountVectorizer(max_features=1000)
    X_data_full_vector = cv.fit_transform(X_data_full_lemmatized_words).toarray()

    tfidf = TfidfTransformer()
    X_data_full_tfidf = tfidf.fit_transform(X_data_full_vector).toarray()

    #Saving Vectorizer for user inputs
    vec_file = 'vectorizer.pickle'
    pickle.dump(cv, open(vec_file, 'wb'))

    return X_data_full_tfidf


data_X = final(X_data)
X_train, X_test, y_train, y_test = train_test_split(data_X, y_data, test_size=0.25, random_state=seed)

# NAIVE BAYES Classifier
MNB = MultinomialNB()
MNB.fit(X_train, y_train)
predictions = MNB.predict(X_test)
# Model evaluation
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
MNB_f1 = round(f1_score(y_test, predictions, average='weighted'), 3)
MNB_accuracy = round((accuracy_score(y_test, predictions) * 100), 2)
print("Accuracy : ", MNB_accuracy, " %")
print("F-Score : ", MNB_f1)

CM = confusion_matrix(y_test, predictions)
FP = CM.sum(axis=0) - np.diag(CM)
TP = np.diag(CM)
fp = (FP/len(y_test))[2]
tp = (TP/len(y_test))[2]

plt.figure()
lw = 2
false_positive_rate = fp # get actual number from your results
true_positive_rate = tp # get actual number from your results
plt.plot([0, false_positive_rate, 1], [0,true_positive_rate, 1], color='darkorange', lw=lw, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC (Receiver operating characteristic) curve')
plt.legend(loc="lower right")
plt.show()

x = input("Please enter a review: ").lower()
print("Conducting Sentiment Analysis on: " + x + " ... ")

def input_process(text):
    translator = str.maketrans('', '', string.punctuation)
    nopunc = text.translate(translator)
    words = [word for word in nopunc.split()]
    return ' '.join(words)

#Vectorizer re-used

loaded_vec = pickle.load(open('vectorizer.pickle', 'rb'))
print(MNB.predict(loaded_vec.transform([input_process(x)])))

