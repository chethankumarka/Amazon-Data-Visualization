import csv
<<<<<<< HEAD
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
=======

from wordcloud import WordCloud
import matplotlib.pyplot as plt
>>>>>>> 18bfeabedab1b93401c0b52a3520623cbaab804c
import pandas as pd
import time
import json
from collections import Counter
# Load library
<<<<<<< HEAD
import numpy as np
=======
>>>>>>> 18bfeabedab1b93401c0b52a3520623cbaab804c
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
# You will have to download the set of stop words the first time
import nltk

# Load stop words
stop_words = set(stopwords.words('english'))
start = time.clock()
meta_file = 'meta_Musical_Instruments.json'
# rating_file = 'ratings_Musical_Instruments.csv'
review_file = 'reviews_Musical_Instruments.json'


def fetch_data(review_file_path, meta_file_path):
    # Metadata
    meta_item_dict = {}
    brand_count_dict = {}
    with open(meta_file_path, 'r') as f:
        raw_data = f.read().split('\n')
        for entry in raw_data:
            try:
                d = json.loads(entry.replace("'", '"'))
                if 'brand' in d.keys():
                    if len(d['brand'].strip()) > 1:
                        if d['brand'] in brand_count_dict:
                            brand_count_dict[d['brand']] += 1
                        else:
                            brand_count_dict[d['brand']] = 1
<<<<<<< HEAD
                        val = {'asin': d['asin'], 'title':d['title'],'imUrl': d['imUrl'], 'brand': d['brand']}
=======
                        val = {'asin': d['asin'], 'imUrl': d['imUrl'], 'brand': d['brand']}
>>>>>>> 18bfeabedab1b93401c0b52a3520623cbaab804c
                        meta_item_dict[d['asin']] = val
            except:
                continue

    # Ratings
    # ratings_list = pd.read_csv(rating_file_path, names=['cid', 'asin', 'rating', 'timestamp']).dropna(
    #     how='any').drop_duplicates().to_dict('records')

    # Reviews
    itemwise_review_dict = {}
    itemwise_rating_dict = {}
    with open(review_file_path, 'r') as f:
        raw_data = f.read().split('\n')
        for entry in raw_data:
            try:
                d = json.loads(entry)
                if d['asin'] in meta_item_dict and d['asin'] in itemwise_review_dict:
                    s = d['reviewText']
                    # s = s.replace(";"," ")
                    # s = s+";"
                    rating = itemwise_rating_dict[d['asin']]
<<<<<<< HEAD
                    rating[int(d['overall']-1)] += 1
                    itemwise_rating_dict[d['asin']] = rating
=======
                    rating[int(d['overall'])] += 1
                    itemwise_rating_dict[d['asin']] = rating

>>>>>>> 18bfeabedab1b93401c0b52a3520623cbaab804c
                    itemwise_review_dict[d['asin']].append([s])
                elif d['asin'] in meta_item_dict:
                    s = d['reviewText']
                    # s = s.replace(";", " ")
                    # s = s + ";"
                    rating = [0, 0, 0, 0, 0]
<<<<<<< HEAD
                    rating[int(d['overall']-1)] = 1
=======
                    rating[int(d['overall'])] = 1
>>>>>>> 18bfeabedab1b93401c0b52a3520623cbaab804c
                    itemwise_rating_dict[d['asin']] = rating
                    itemwise_review_dict[d['asin']] = [[s]]
                else:
                    continue
<<<<<<< HEAD
            except Exception as e:
                print("the error is::",e)
=======
            except:
>>>>>>> 18bfeabedab1b93401c0b52a3520623cbaab804c
                continue

    # Merge data from all 3 based on asin
    common_brandsdict = dict(Counter(brand_count_dict).most_common(8))
    common_brands = common_brandsdict.keys()
    final_list = []
    for item in meta_item_dict.values():
        asin1 = item['asin']

        if (asin1 in itemwise_review_dict) and (asin1 in meta_item_dict) and (item['brand'] in common_brands):
            reviews = itemwise_review_dict[asin1]
            metadata = meta_item_dict[asin1]
            final_list.append(
<<<<<<< HEAD
                {'asin': asin1, 'reviews': reviews, 'imUrl': metadata['imUrl'],'title': metadata['title'],
=======
                {'asin': asin1, 'reviews': reviews, 'imUrl': metadata['imUrl'],
>>>>>>> 18bfeabedab1b93401c0b52a3520623cbaab804c
                 'brand': metadata['brand'], 'rating': itemwise_rating_dict[asin1]})

    return pd.DataFrame(final_list), common_brandsdict


# write the data frame with top k brands to a csv file
op, common_brands = fetch_data(review_file, meta_file)
grp1 = op.sort_values(by=['brand'])
# grp1.to_csv('./output.csv')

# Run sentiment Ananlysis for the Products using nltk
# grp1 = pd.read_csv('./output.csv')

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()


def sentiment_analyzer_scores(sentence):
    score = []
    # pos = 0
    # neg = 0
    # neutral = 0
    compound = 0
    max = -1
    best_sentence = ""
    for i in sentence:
        res = analyser.polarity_scores(i[0])
        score.append(res)
        if res['pos'] > max:
            max = res['pos']
            best_sentence = i
        # pos = pos + res['pos']
        # neg = neg + res['neg']
        # neutral = neutral + res['neu']
        compound = compound + res['compound']

    return score, compound / len(sentence), best_sentence


compound_product_sentiment = {}
overall_brand_sentiment = {}
total_reviews = []
<<<<<<< HEAD
product_name = []
for index, row in grp1.iterrows():
    total_reviews.append(len(row['reviews']))
    product_name.append((row['title']))
=======
for index, row in grp1.iterrows():
    total_reviews.append(len(row['reviews']))
>>>>>>> 18bfeabedab1b93401c0b52a3520623cbaab804c
    compound_product_sentiment[row['asin']] = sentiment_analyzer_scores(row['reviews'])
grp1['total_reviews'] = total_reviews


<<<<<<< HEAD

=======
>>>>>>> 18bfeabedab1b93401c0b52a3520623cbaab804c
def findsentiment_label():
    label = []
    sentiment = []
    best_review = []
    for i in compound_product_sentiment.keys():
        # print(str(i)+":")
        # print(compound_product_sentiment[i][1])
        compound = compound_product_sentiment[i][1]
        sentiment.append(compound)
        best_review.append(compound_product_sentiment[i][2])
        if compound > 0:
            label.append("positive")
        else:
            label.append("negative")
        # else:
        #     label.append("neutral")
    grp1['sentiment'] = sentiment
    grp1['label'] = label
    grp1['best_review'] = best_review


# Call findsentiment_label to append data and label to grp1 data frame
findsentiment_label()


# Calculate overall brand sentiment
def write_piechartjson(df):
    d = {}
    for i, v in df.iteritems():
        if i[0] in d.keys():
            a = d[i[0]]
            a[i[1]] = v
            d[i[0]] = a
        else:
            a = {i[1]: v}
            d[i[0]] = a
    with open("bubbleCloud.json", 'w') as fp:
        with open("bubbleCloud.csv",'w') as fp2:
            for i in d.keys():
                freq_list = {i: d[i]}
                fp.write(json.dumps(freq_list) + "\n")
                w = csv.writer(fp2, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                vals = d[i].values()
                w.writerow([i,d[i]['positive'],d[i]['negative'],d[i]['positive']+d[i]['negative']])


grouped_df = grp1.groupby('brand')['label'].value_counts()

# Write the values required for the pie chart and the bubble chart
write_piechartjson(grouped_df)

# #products with a minimum review count of 15
filtered_output = grp1.loc[grp1['total_reviews'] > 15]
<<<<<<< HEAD
#filtered_output.to_csv('filtered_output.csv')
=======
# filtered_output.to_csv('filtered_output.csv')
>>>>>>> 18bfeabedab1b93401c0b52a3520623cbaab804c

# Select the top k products by changing the head count
df1 = filtered_output.sort_values('sentiment', ascending=False).groupby('brand').head(8)

<<<<<<< HEAD
df1 = df1.sort_values('brand', ascending=True)

# Write the final data required in a csv ( asin	brand	imUrl	rating	reviews	total_reviews	sentiment	label	best_review	)
#df1.to_csv('final.csv')

with open("Final_json.json","w+",encoding='utf-8') as json_file:
    df1.to_json(json_file,orient="records",force_ascii=False)
=======
# Write the final data required in a csv ( asin	brand	imUrl	rating	reviews	total_reviews	sentiment	label	best_review	)
df1.to_csv('final.csv')
>>>>>>> 18bfeabedab1b93401c0b52a3520623cbaab804c


# Word Cloud for each top product
# df1 = pd.read_csv('final.csv',dtype={'reviews': object})
# def build_WordCloud(reviews,brand,asin):
# Start with one review:
# text = ""
# for i in reviews:
#     text = text+" "+i[0]

# # Create and generate a word cloud image:
# wordcloud = WordCloud(width=800, height=800,
#                       background_color='white',
#                       stopwords=stop_words,
#                       min_font_size=10,max_words=50).generate(reviews)
#
# # Display the generated image:
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.savefig("word_clouds/"+brand+asin+".png", format="png")

###Function to calculate word frequency for word cloud with 50 top words
def calculate_word_frequency(st):
    # Post: return a list of words ordered from the most
    # frequent to the least frequent
    import string
<<<<<<< HEAD

=======
>>>>>>> 18bfeabedab1b93401c0b52a3520623cbaab804c
    text = ""
    for j in st:
        text = text + " " + j[0].lower()

    combined_review = text.translate(string.punctuation)

    tokenizer = RegexpTokenizer(r'\w+')
    word_tokens = tokenizer.tokenize(combined_review)
    stop_words.update(["(", ')', '[', ']'])
    filtered_sentence = [w for w in word_tokens if w not in stop_words]
    words = Counter()
    words.update(filtered_sentence)
<<<<<<< HEAD
    frequent_words = words.most_common(200)
    sentiment = 0
    count = 0
    sent_dict = {}
    for i in words:
        sentiment = TextBlob(i).sentiment
        count = words[i]

        sent_dict[i] = {'polarity':sentiment.polarity,'count':count}
    sorted_dict = sorted(sent_dict,key = lambda x: (sent_dict[x]['polarity']*sent_dict[x]['count']))
    i = 0
    j = len(sorted_dict)-1
    pos_neg_sent_dict={}
    while i<min(100,len(sorted_dict)/2) and j>len(sorted_dict)-min(100,(len(sorted_dict)/2)-1):
        pos_neg_sent_dict[sorted_dict[i]] = {'polarity':sent_dict[sorted_dict[i]]['polarity'],'count':sent_dict[sorted_dict[i]]['count']}
        pos_neg_sent_dict[sorted_dict[j]] = {'polarity': sent_dict[sorted_dict[j]]['polarity'],
                                             'count': sent_dict[sorted_dict[j]]['count']}
        i+=1
        j-=1
    #print(pos_neg_sent_dict)
    return pos_neg_sent_dict
    #return dict(frequent_words)
=======
    frequent_words = words.most_common(50)
    return dict(frequent_words)
>>>>>>> 18bfeabedab1b93401c0b52a3520623cbaab804c


# Calculate word frequency of a product and write json data word cloud
with open('wordCloud.json', 'w') as fp:
    for index, row in df1.iterrows():
        # build_WordCloud(row['reviews'],row['brand'],row['asin'])
        diction = calculate_word_frequency(row['reviews'])
        s = row['brand'] + row['asin']
        freq_list = {s: diction}
        fp.write(json.dumps(freq_list) + "\n")

<<<<<<< HEAD
print("Execution Time: ", time.clock()-start)


def do_lda(reviews_string):
    vectorizer = CountVectorizer(stop_words='english', lowercase=True, token_pattern='\s\w+\s', max_df=0.8)
    vectorized_data = vectorizer.fit_transform(reviews_string)
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
    all_words = giant_document.split()
    frequency_table = {}
    for word in all_words:
        l_word = word.lower().strip()
        if l_word in frequency_table:
            frequency_table[l_word] += 1
        else:
            frequency_table[l_word] = 1

    return frequency_table


def get_lda_words(data, top_brands):
    lda_brand_dict = {}
    for brand in top_brands:
        lda_list = []
        brand_reviews = data[data['brand'] == brand]['reviews'].tolist()
        temp_array = []
        for l1 in brand_reviews:
            for l2 in l1:
                for l3 in l2:
                    temp_array.append(l3)
        labels = do_lda(temp_array)
        ft = get_frequency_table(temp_array)

        for label, words in labels.items():
            labels[label] = sorted(words, key=(lambda x: ft[x]), reverse=True)

        for label, words in labels.items():
            ii = 0
            counter = 0
            while (counter < 10):
                if words[ii] in ft:
                    counter += 1
                    lda_list.append({'word': words[ii], 'count': ft[words[ii]], 'label': label})
                ii += 1

        lda_brand_dict[brand] = lda_list

    with open('result.json', 'w') as fp:
        json.dump(lda_brand_dict, fp)

    return


get_lda_words(op, common_brands)
=======
print("Execution Time: ", time.clock()-start)
>>>>>>> 18bfeabedab1b93401c0b52a3520623cbaab804c
