import numpy as np
import lda
import os
import lda.datasets
import json
import nltk
import re
import pickle
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.cistem import Cistem
from nltk.stem.snowball import GermanStemmer

with open("Golem Bereinigt 2010-2019.json", "r") as f:
    d = json.load(f)

sorted_d = np.sort([int(x["Veröffentlichungsdatum"].split("-")[0]) for x in d])
year_indices = {}
for ind, ind_year in enumerate(sorted([np.where(sorted_d == x)[0][0] for x in set(sorted_d)])):
    year_indices.update({list(range(2010, 2019 + 1))[ind]: ind_year})
year_indices[2020] = None

# IMPORTANT!
# 70 k times 588 k is big, sizing down therefore
year = 2010
d = d[year_indices[year]:year_indices[year + 1]]  # d[:500]

nltk.download("stopwords")
stop_words_en = stopwords.words('english')

stemmer = GermanStemmer()  # Cistem()

with open("stop_full.pkl", "rb") as f:
    stop_words = pickle.load(f)
    stop_words = [x.strip() for x in stop_words] + stop_words_en


def preprocess(text):
    text = text.lower().split()
    # text = [w.split(".")[0].split(",")[0].split(":")[0].split(";")[0] for w in text]
    text = " ".join(text)
    remove_punctuation_regex = re.compile(
        r"[^A-ZÄÖÜäöüßa-z ]")  # regex for all characters that are NOT A-Z, a-z and space " "
    text = re.sub(remove_punctuation_regex, "", text)
    text = text.split()
    text = [word for word in text if word not in stop_words]
    text = [stemmer.stem(word) for word in text]
    return text


if os.path.exists("data_{}.pkl".format(year)):

    documents = [preprocess(x["Text"]) for x in d]
    documents_lens = [len(d) for d in documents]
    plt.hist(documents_lens, bins=100)
    plt.show()

    vocab = np.unique([item for sublist in documents for item in sublist])


    with open("freq_{}.pkl".format(year), "rb") as f:
        term_frequency = pickle.load(f)

    with open("data_{}.pkl".format(year), "rb") as f:
        X = pickle.load(f)

else:

    documents = [preprocess(x["Text"]) for x in d]
    documents_lens = [len(d) for d in documents]
    plt.hist(documents_lens, bins=100)
    plt.show()

    vocab = np.unique([item for sublist in documents for item in sublist])

    term_frequency = []
    for ind,v in enumerate(vocab):
        term_frequency.append(sum([len(np.where(v == np.array(d))[0]) for d in documents]))
        print("{}/{}".format(ind,len(vocab)),end="\r",flush=True)

    with open("freq_{}.pkl".format(year), "wb") as f:
        pickle.dump(term_frequency, f, protocol=4)

    X = np.zeros((len(documents), len(vocab)), dtype=int)
    for ind_row, row in enumerate(X):
        for ind_col, col in enumerate(row):
            freq_of_word_in_given_document = len(np.where(vocab[ind_col] == np.array(documents[ind_row]))[0])
            X[ind_row, ind_col] = freq_of_word_in_given_document
            print('Document {}/{} Word {}/{}                  '.format(ind_row, len(documents), ind_col, len(vocab)),
                  end='\r', flush=True)

    with open("data_{}.pkl".format(year), "wb") as f:
        pickle.dump(X, f, protocol=4)

# toy dataset example # X = lda.datasets.load_reuters()
# vocab = lda.datasets.load_reuters_vocab()
# titles = lda.datasets.load_reuters_titles()

model = lda.LDA(n_topics=30, n_iter=150, alpha=0.3, eta=0.05, random_state=1)
model.fit(X)  # model.fit_transform(X) is also available

topic_word = model.topic_word_  # model.components_ also works
n_top_words = 8
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))


import pyLDAvis

#term_frequency = np.random.randint(0,500,len(vocab))
data = {'topic_term_dists': model.topic_word_,
        'doc_topic_dists': model.doc_topic_,
        'doc_lengths': documents_lens,
        'vocab': vocab,
        'term_frequency': term_frequency}

vis_data = pyLDAvis.prepare(**data)
pyLDAvis.show(vis_data)
pyLDAvis.save_html(vis_data,'./ldavis_'+ str(year) +'.html')


# conversion of model 1 topic data structure to different which is supported by WordCloud
topics_m1 = []
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
    topic_word_probs = topic_dist[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
    topics_m1.append((i, [(x, topic_word_probs[ind]) for ind, x in enumerate(topic_words)]))






"""
Alternativer Approach mit Gensim's LDA Implementation
"""

import sys
import re, numpy as np, pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
from collections import Counter

data_ready = documents  # so wie oben im ersten Beispiel erzeugt

# Create Dictionary
id2word = corpora.Dictionary(data_ready)

# Create Corpus: Term Document Frequency
corpus = [id2word.doc2bow(text) for text in data_ready]

# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=10,
                                            random_state=100,
                                            update_every=1,
                                            passes=10,
                                            alpha=0.3,
                                            eta=0.05,
                                            iterations=1500,
                                            per_word_topics=True,
                                            )

"""
Below is the Word Cloud plot
"""

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(stopwords=stop_words,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=8,
                  colormap=None,
                  color_func=None, #*args, **kwargs: cols[i],
                  prefer_horizontal=0.90)

topics = topics_m1  # lda_model.show_topics(formatted=False)

fig, axes = plt.subplots(6, 5, figsize=(10, 10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')

plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()

"""
Below is the Frequency Weight Plot
"""

topics = lda_model.show_topics(formatted=False)
data_flat = [w for w_list in data_ready for w in w_list]
counter = Counter(data_flat)

out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i, weight, counter[word]])

df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])

# Plot Word Count and Weights of Topic Keywords
fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True, dpi=160)
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=df.loc[df.topic_id == i, :], color=cols[i], width=0.5, alpha=0.3,
           label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id == i, :], color=cols[i], width=0.2,
                label='Weights')
    ax.set_ylabel('Word Count', color=cols[i])
    ax_twin.set_ylim(0, 0.030);
    ax.set_ylim(0, 3500)
    ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(df.loc[df.topic_id == i, 'word'], rotation=30, horizontalalignment='right')
    ax.legend(loc='upper left');
    ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)
fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)
plt.show()
