# Packages installation
!pip install xmltodict
!pip install ktrain
!pip install gensim
!pip install nltk

# import libraries
import urllib # For making API call
import xmltodict as xtd
import ktrain
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS as stopwordss


# Get English stopwords
english_stopwords = set(stopwords.words('english'))

# Function to remove stopwords from a string
def remove_stopwords(string):
    # Tokenize the string
    words = word_tokenize(string)
    # Remove stopwords
    filtered_words = [word for word in words if word.lower() not in english_stopwords]
    # Join the filtered words back into a string
    filtered_string = ' '.join(filtered_words)
    return filtered_string


stemmer = SnowballStemmer('english')

def stem(text):
  return stemmer.stem(text)


def preprocess(text):
  result = []
  for token in simple_preprocess(text, min_len = 4):
    if token not in stopwordss:
      result.append(stem(token))

  return result



keyWords = input("Please provide your search terms or key words: ")
searchTerm = '+'.join(keyWords.replace(',', ' ').split())

print('searchTerm: ', searchTerm)



# Get dataset
url = 'http://export.arxiv.org/api/query?search_query=all:'+searchTerm+'&start=0&max_results=1000'
data = urllib.request.urlopen(url)

papers_in_xml = data.read().decode('utf-8')
papers_in_xml


# Preporcess data - Convert dataset from XML to dictionary data type for easy processing
papers_in_dict = xtd.parse(papers_in_xml)
papers = papers_in_dict['feed']['entry']
print(papers)

# Preporcess data - Format data to include only ID, title, and summary
mainPapers = [];
for paper in papers:
  id = paper['id'],
  title = paper['title'],
  summary = paper['summary']
  mainPapers.append({"id": id[0], "title": title[0], "summary": summary, "text": title[0]+" \n "+summary})


# Show the number of datasets fetched based on user keywords.
print('Total dataset: %s'% len(mainPapers));


# Preporcess data - build up the list of texts to be fed to the recommender model
texts = [ paper['text'] for paper in mainPapers ]
print(texts[0])


# Fit a topic model to documents in <texts>.
# The get_topic_model function learns a topic model using Latent Dirichlet Allocation (LDA).
%%time
tm = ktrain.text.get_topic_model(texts=texts, model_type='lda', n_topics=100, n_features=10000, min_df=2,
                                 max_df=0.8, stop_words='english', max_iter=10, lda_max_iter=None,
                                 lda_mode='online', token_pattern=None, verbose=1
                                 )



# Show generated topics to get a feel for what kinds of subjects
# are discussed within this dataset.
tm.print_topics()



from wordcloud import WordCloud
import matplotlib.pyplot as plt

filtered_texts = [', '.join(preprocess(text)) for text in texts]

# Generate word cloud for the entire corpus
text = ' '.join(filtered_texts)  # Combine all filtered strings into a single text
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Plot word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud on Dataset for Search Term: ' + keyWords)
plt.axis('off')
plt.show()


# Plot histogram of document lengths
doc_lengths = [len(string.split()) for string in filtered_texts]
plt.figure(figsize=(8, 6))
plt.hist(doc_lengths, bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Document Lengths')
plt.xlabel('Document Length')
plt.ylabel('Frequency')
plt.show()




import matplotlib.pyplot as plt
from collections import Counter

# Tokenize each document and flatten the list of tokens
tokens = [word.lower() for text in filtered_texts for word in text.split()]

# Calculate word frequencies
word_freq = Counter(tokens)

# Sort the word frequencies in descending order
sorted_word_freq = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True))

# Extract the top N most frequent words and their frequencies
top_n = 10 
top_words = list(sorted_word_freq.keys())[:top_n]
top_freqs = list(sorted_word_freq.values())[:top_n]

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(top_words, top_freqs, color='skyblue')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 10 Most Frequent Words')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()




# The build fnx uses Latent Dirichlet Allocation (LDA) to analyze the input documents and determine their topic distributions.
# For each document in texts, the function computes the probability distribution over topics.
# It also filters out documents whose highest topic probability is below the set threshold.
tm.build(texts, threshold=0.2)


# This is useful to ensure all data and metadata are aligned with the same array indices in case
# we want to use them later (e.g., in visualizations, for example).
texts = tm.filter(texts)


# Show topics with words that make up the topic and the number of documents with that topic as primary.
tm.print_topics(show_counts=True)


# Visualizing the corpus
tm.visualize_documents(doc_topics=tm.get_doctopics())


# Train the model
tm.train_recommender()




# Extract search terms
search_terms = keyWords.lower().split()

# Remove stopwords from Search Terms
filtered_searchTerms = [remove_stopwords(word) for word in search_terms]

# Calculate the minimum number of key terms required for relevance
min_key_terms_required = int(0.4 * len(filtered_searchTerms))

# Determine relevant documents based on search terms in titles
relevant_documents = []

for paper in mainPapers:
    title_lower = paper['text'].lower()
    # Count the number of key terms present in the title
    key_terms_in_title = sum(1 for term in filtered_searchTerms if term in title_lower)
    # Calculate relevance score based on the percentage of key terms present in the title
    relevance_score = key_terms_in_title / len(filtered_searchTerms)
    # Check if the paper has at least 40% of the key terms
    if key_terms_in_title >= min_key_terms_required:
        paper['relevance_score'] = relevance_score  # Assign relevance score to the document
        relevant_documents.append(paper)

# Sort relevant_documents based on relevance score
relevant_documents.sort(key=lambda x: x['relevance_score'], reverse=True)

print(filtered_searchTerms)
print(len(relevant_documents))



# Make recommendation
n_neighbor = 100 if len(mainPapers) > 100 else len(mainPapers) - 10
recommended_documents = []

# Sort relevant_documents based on relevance score
relevant_documents.sort(key=lambda x: x['relevance_score'], reverse=True)

# relevant_documents[:5]
count = 0  # Initialize a count variable to keep track of the number of recommendations appended
for i, doc in enumerate(tm.recommend(text=keyWords, n=len(relevant_documents) + 5, n_neighbors=n_neighbor), start=1):
    for p in relevant_documents:  # Only consider relevant documents for recommendations
        if p['text'] == doc['text']:
            recommended_documents.append(p)

count = 0  # Initialize a count variable to keep track of the number of recommendations appended
for i, doc in enumerate(tm.recommend(text=keyWords, n=len(relevant_documents) + 5, n_neighbors=n_neighbor), start=1):
    for p in relevant_documents:  # Only consider relevant documents for recommendations
        if p['text'] == doc['text']:
            print('Recommendation #%s' % i)
            print('Access Link:\t%s' % p['id'])
            # print('Relevance Score:\t%s' % p['relevance_score'])  # Print relevance score
            print('Title:\t%s' % p['title'])
            print('Summary:\t%s' % p['summary'])
            print()
            count += 1
            if count >= 5:  # Break out of the loop after appending the top 5 recommendations
                break
    if count >= 5:  # Break out of the outer loop as well if the top 5 recommendations are found
        break
    




# Evaluate the system
relevant_set = set([doc['id'] for doc in relevant_documents])
recommended_set = set([doc['id'] for doc in recommended_documents])


true_positives = len(relevant_set.intersection(recommended_set))
false_positives = len(recommended_set - relevant_set)
false_negatives = len(relevant_set - recommended_set)
total_documents = len(relevant_set) + len(recommended_set)
true_negatives = total_documents - (true_positives + false_positives + false_negatives)

# Calculate precision, recall, and F1-score
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision > 0 or recall > 0) else 0
accuracy = (true_positives + true_negatives) / total_documents

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)



import numpy as np
import matplotlib.pyplot as plt

# Define the confusion matrix
conf_matrix = np.array([[true_positives, false_negatives],
                        [false_positives, true_negatives]])

# Plotting
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)

# Add count annotations to each cell
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(conf_matrix[i, j]), horizontalalignment='center', verticalalignment='center', color='black')

# Add labels to the x-axis and y-axis
plt.xticks([0, 1], ['Predicted Negative', 'Predicted Positive'])
plt.yticks([0, 1], ['True Negative', 'True Positive'])

plt.title('Confusion Matrix')
plt.colorbar()
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
