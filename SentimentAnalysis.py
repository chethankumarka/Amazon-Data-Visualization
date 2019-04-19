import csv

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import time
import json
from collections import Counter
# Load library
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
                        val = {'asin': d['asin'], 'imUrl': d['imUrl'], 'brand': d['brand']}
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
                    rating[int(d['overall'])] += 1
                    itemwise_rating_dict[d['asin']] = rating

                    itemwise_review_dict[d['asin']].append([s])
                elif d['asin'] in meta_item_dict:
                    s = d['reviewText']
                    # s = s.replace(";", " ")
                    # s = s + ";"
                    rating = [0, 0, 0, 0, 0]
                    rating[int(d['overall'])] = 1
                    itemwise_rating_dict[d['asin']] = rating
                    itemwise_review_dict[d['asin']] = [[s]]
                else:
                    continue
            except:
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
                {'asin': asin1, 'reviews': reviews, 'imUrl': metadata['imUrl'],
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
for index, row in grp1.iterrows():
    total_reviews.append(len(row['reviews']))
    compound_product_sentiment[row['asin']] = sentiment_analyzer_scores(row['reviews'])
grp1['total_reviews'] = total_reviews


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
# filtered_output.to_csv('filtered_output.csv')

# Select the top k products by changing the head count
df1 = filtered_output.sort_values('sentiment', ascending=False).groupby('brand').head(8)

# Write the final data required in a csv ( asin	brand	imUrl	rating	reviews	total_reviews	sentiment	label	best_review	)
df1.to_csv('final.csv')


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
    frequent_words = words.most_common(50)
    return dict(frequent_words)


# Calculate word frequency of a product and write json data word cloud
with open('wordCloud.json', 'w') as fp:
    for index, row in df1.iterrows():
        # build_WordCloud(row['reviews'],row['brand'],row['asin'])
        diction = calculate_word_frequency(row['reviews'])
        s = row['brand'] + row['asin']
        freq_list = {s: diction}
        fp.write(json.dumps(freq_list) + "\n")

print("Execution Time: ", time.clock()-start)