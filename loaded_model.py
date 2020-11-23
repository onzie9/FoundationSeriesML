import re
import pickle
import random
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
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

text = open('foundationsedge.txt')
lines = text.readlines()

nonemptylines = []
for i in range(len(lines)):
    if lines[i] != '\n':
        lines[i] = re.sub(r'[^\w\s]', ' ', lines[i])
        lines[i] = re.sub(r'\s+[a-zA-Z]\s+', ' ', lines[i])
        lines[i] = lines[i].lower()
        nonemptylines.append(lines[i][:-2])

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

linebreaks = 15
chapchunks = []
chapstartchunks = []
y = []
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

s = round(len(chunks)*.8)
samp = random.sample(range(len(chunks)), s)

X_test = []
y_test = []
for x in samp:
    X_test.append(X[x])
    y_test.append(y[x])


with open('text_classifier', 'rb') as training_model:
    model = pickle.load(training_model)

y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))