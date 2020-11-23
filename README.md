# FoundationSeriesML
Using ML to explore a hunch about Asimov's Foundation Series

# Background
Being a lifelong fan of Asimov's writings (independent of his personal behaviors that were questionable), and specifically of the Foundation trilogy, I had a hunch that the writing style at the onset of each chapter of his Foundation books was different from the writing in the body of the chapters.  Since I don't know much about text processing, I decided to take a quick plunge into this project to see if I could use ML to suss out the differences.

# Method 1
I imported the text of the Foundation Trilogy and Foundation's Edge (written 30 years apart), but only trained with the trilogy.  During EDA, I cleaned the data a little, and chunked the text into groups of 25 lines starting with each chapter.  I flagged the first chunk of each chapter as being a starting chunk.  I then ran a series of textstat statistics on the chunks such as the Klesch reading level and several others.  I took averages of the various stats on the starting chunks and nonstarting chunks, and noticed there were some measurable gaps.  

I continued by creating vectors for each chunk of the various textstat statistics.  Then I normalized all the vectors and ran k-means clustering with two clusters.  Sadly, clustering was ineffective, scoring no better than a coin flip at identifying starting chunks versus nonstarting chunks. This code can be found in kmeans.py.  Note that I didn't go any further than checking the cluster distribution for the 60 known starting chunks.  There were 35 in one cluster, and 25 in the other, so that is statistically insignificant.

# Method 2
After the kmeans failure, I skipped over SVM and went to a random forest with vectorization of the chunks.  Here, I had to remove all punctuation, single characters, and made everything lowercase.  I skipped TF-IDF because all the chunks under consideration were roughly the same size. After all the preprocessing, I ran a random forest classifier on the chunks, and achieved 89% accuracy in classifying the starting chunks as such.  I then output the model via pickle, and ran the same model against Foundation's Edge, using the same preprocessing and chunking paradigm, and achieved 88% accuracy.  So depsite being written 30 years apart, apparently Asimov's writing style as it related to the Foundation universe was stable enough that the same model works. This code is found in rfc.py and loaded_model.py.
