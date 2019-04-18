import json
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
from nltk.corpus import stopwords
import numpy as np

meta_file = 'meta_Musical_Instruments.json'
rating_file = 'ratings_Musical_Instruments.csv'
review_file = 'reviews_Musical_Instruments.json'


def fetch_data(review_file_path, rating_file_path, meta_file_path):
    # Metadata
    meta_item_dict = {}
    with open(meta_file_path, 'r') as f:
        raw_data = f.read().split('\n')
        for entry in raw_data:
            try:
                d = json.loads(entry.replace("'", '"'))
                if 'brand' in d.keys():
                    if len(d['brand'].strip()) > 1:
                        val = {'asin': d['asin'], 'price': d['price'], 'imUrl': d['imUrl'], 'brand': d['brand'],
                               'name': d['title']}
                        meta_item_dict[d['asin']] = val
            except:
                continue

    # Ratings
    ratings_list = pd.read_csv(rating_file_path, names=['cid', 'asin', 'rating', 'timestamp']).dropna(
        how='any').groupby('asin').mean().reset_index().to_dict('records')

    # Reviews
    itemwise_review_dict = {}
    with open(review_file_path, 'r') as f:
        raw_data = f.read().split('\n')
        for entry in raw_data:
            try:
                d = json.loads(entry.replace("'", '"'))
                if d['asin'] in itemwise_review_dict:
                    itemwise_review_dict[d['asin']].append(d['reviewText'])
                else:
                    itemwise_review_dict[d['asin']] = [d['reviewText']]
            except:
                continue

    # Merge data from all 3 based on asin
    final_list = []
    for item in ratings_list:
        asin = item['asin']
        if (asin in itemwise_review_dict) and (asin in meta_item_dict):
            reviews = itemwise_review_dict[asin]
            metadata = meta_item_dict[asin]
            final_list.append(
                {'asin': asin, 'reviews': " ".join(reviews), 'price': metadata['price'], 'imUrl': metadata['imUrl'],
                 'brand': metadata['brand'], 'rating': item['rating'], 'name': metadata['name']})

    return pd.DataFrame.from_dict(final_list).drop_duplicates(subset=['asin'])


def get_topicwise_words(documents):
    vectorizer = CountVectorizer(stop_words='english', lowercase=True, token_pattern='\s\w+\s')
    vectorized_data = vectorizer.fit_transform(documents)
    lda = LatentDirichletAllocation(n_components=4, max_iter=15)
    lda.fit(vectorized_data)
    components = lda.components_.T
    features = vectorizer.get_feature_names()
    labels = {0: [], 1: [], 2: [], 3: []}
    stop_words = set(stopwords.words('english'))
    for i in range(len(features)):
        label = np.argmax(components[i])
        word = features[i].lower().strip()
        if word not in stop_words:
            labels[label].append(word)

    return labels


def get_frequency_table(documents):
    giant_document = " ".join(documents)
    all_words = re.findall('\s\w+\s', giant_document)
    frequency_table = {}
    for word in all_words:
        l_word = word.lower().strip()
        if l_word in frequency_table:
            frequency_table[l_word] += 1
        else:
            frequency_table[l_word] = 1

    return frequency_table


def get_word_cloud_words(document_list):
    ft = get_frequency_table(document_list)
    labels = get_topicwise_words(document_list)

    temp = {}
    for label, words in labels.items():
        sorted_words = sorted(words, key=(lambda x: 0 if x not in ft else ft[x]), reverse=True)
        sorted_words = list(filter((lambda x: x in ft), sorted_words))
        temp[label] = sorted_words[:10]

    word_cloud_words = []
    for label, words in temp.items():
        for word in words:
            word_cloud_words.append({'word': word, 'count': ft[word], 'label': label})

    return word_cloud_words

def get_top_k_brands(data, K=10):
    brands_list = data['brand'].tolist()
    counts = {}
    for brand in brands_list:
        if brand in counts:
            counts[brand] += 1
        else:
            counts[brand] = 1
    count_tuples = list([k, v] for k, v in counts.items())
    count_tuples = sorted(count_tuples, key=(lambda x: x[1]), reverse=True)
    return list(x[0] for x in count_tuples)[:K]

def get_brand_word_clouds(top_brands, data):
    brand_wise_words = {}
    for top_brand in top_brands:
        subset_data = data[data['brand'] == top_brand]['reviews'].tolist()
        brand_wise_words[top_brand] = get_word_cloud_words(subset_data)

    return brand_wise_words

d = fetch_data(review_file, rating_file, meta_file)

top_brands = get_top_k_brands(d, 10)
dictionary = get_brand_word_clouds(top_brands, d)
with open('result.json', 'w') as fp:
    json.dump(dictionary, fp)

    # Brand frequency
    # fig, ax = plt.subplots()
    # op['brand'].value_counts().head(10).plot(ax=ax, kind='bar')
    # plt.show()
