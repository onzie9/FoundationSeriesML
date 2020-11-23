import re
import textstat
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')
from nltk.corpus import stopwords
import pickle


text = open('fulltext.txt')
lines = text.readlines()

nonemptylines = []
for i in range(len(lines)):
    if lines[i] != '\n':
        lines[i] = re.sub(r'[^\w\s]',' ', lines[i])
        lines[i] = re.sub(r'\s+[a-zA-Z]\s+', ' ', lines[i])
        lines[i] = lines[i].lower()
        nonemptylines.append(lines[i][:-2])

nonemptylines = nonemptylines[15:]

# Next step: key in the chapter numbers and split out first x% of chapters into a separate list.
chapterstarts = []
chapterindices = []
for i in range(len(nonemptylines)):
    if len(nonemptylines[i]) <= 3:
        if nonemptylines[i] == 'i.':
            chapterstarts.append('1')
            chapterindices.append(i)
        elif re.search(r'\d', nonemptylines[i]):
            chapterstarts.append(nonemptylines[i])
            chapterindices.append(i)

y = []
linebreaks = 25
chapchunks = []
chapstartchunks = []
for x in range(len(chapterindices)-1):
    chap = nonemptylines[chapterindices[x]: chapterindices[x+1]]
    for i in range(round(len(chap)/linebreaks)):
        chapchunks.append(' '.join(chap[linebreaks*i: linebreaks*(i+1)]))
        if i == 0:
            chapstartchunks.append(' '.join(chap[linebreaks*i: linebreaks*(i+1)]))
            y.append('chapter start')

nonstartingchunks = []

for c in chapchunks:
    if c not in chapstartchunks:
        nonstartingchunks.append(c)
        y.append('chapter body')

chunks = chapstartchunks + nonstartingchunks

vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(chunks).toarray()

# Skipping TF-IDF because all samples are roughly the same size.

# Training split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Random forest.
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train)

# Prediction
y_pred = classifier.predict(X_test)

# Accuracy
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

with open('text_classifier', 'wb') as picklefile:
    pickle.dump(classifier,picklefile)

