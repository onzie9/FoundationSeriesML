# The idea of this ML model is to create numerical vectors for each chunk based on textstat scores.  Then those vectors
# will be clustered into two clusters and we'll see how well the clustering does.

import re
import textstat

text = open('fulltext.txt')
lines = text.readlines()


nonemptylines = []
for i in range(len(lines)):
    if lines[i] != '\n':
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

chaplengths = []
for i in range(len(chapterindices)-1):
    chaplengths.append(int(chapterindices[i+1] - chapterindices[i]))

linebreaks = 20
chapchunks = []
chapstartchunks = []
for x in range(len(chapterindices)-1):
    chap = nonemptylines[chapterindices[x]: chapterindices[x+1]]
    for i in range(round(len(chap)/linebreaks)):
        chapchunks.append(' '.join(chap[linebreaks*i: linebreaks*(i+1)]))
        if i == 0:
            chapstartchunks.append(' '.join(chap[linebreaks*i: linebreaks*(i+1)]))

nonstartingchunks = []

for c in chapchunks:
    if c not in chapstartchunks:
        nonstartingchunks.append(c)

startvectors = []
nonstartvectors = []

for c in chapchunks:
    if c in chapstartchunks:
        startvectors.append([textstat.flesch_kincaid_grade(c), textstat.flesch_reading_ease(c), textstat.gunning_fog(c),
                             textstat.smog_index(c), textstat.automated_readability_index(c),
                             textstat.coleman_liau_index(c), textstat.linsear_write_formula(c),
                             textstat.dale_chall_readability_score(c), textstat.text_standard(c, float_output=True)])
    else:
        nonstartvectors.append([textstat.flesch_kincaid_grade(c), textstat.flesch_reading_ease(c),
                                textstat.gunning_fog(c), textstat.smog_index(c),
                                textstat.automated_readability_index(c), textstat.coleman_liau_index(c),
                                textstat.linsear_write_formula(c), textstat.dale_chall_readability_score(c),
                                textstat.text_standard(c, float_output=True)])

vectors = startvectors + nonstartvectors

# Start clustering on vectors.
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_vectors = scaler.fit_transform(vectors)

kmeans = KMeans(init='random', n_clusters=2, n_init=10, max_iter=300, random_state=42)

kmeans.fit(scaled_vectors)

print(kmeans.inertia_)
print(kmeans.cluster_centers_)
print(kmeans.n_iter_)
print(sum(kmeans.labels_[:60]))
print(sum(kmeans.labels_[60:]))
print(silhouette_score(scaled_vectors, kmeans.labels_))

print(len(vectors))
print(len(startvectors))
print(len(nonstartvectors))
print(len(kmeans.labels_))

