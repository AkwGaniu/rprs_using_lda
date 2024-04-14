# Install packages
!pip install xmltodict
!pip install ktrain


# import libraries
import urllib # For making API call
import xmltodict as xtd
import ktrain

# Initialize English stopwords.
stopwords = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
    't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've',
    'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
    'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
    "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
    'wouldn', "wouldn't"
]



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
tm = ktrain.text.get_topic_model(texts=texts, model_type='lda', n_topics=100, n_features=10000, min_df=5,
                                 max_df=0.5, stop_words='english', max_iter=5,lda_max_iter=None,
                                 lda_mode='online', token_pattern=None, verbose=1
                                 )



# Show generated topics to get a feel for what kinds of subjects
# are discussed within this dataset.
tm.print_topics()

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
filtered_searchTerms = [word for word in search_terms if word.lower() not in stopwords]

# Calculate the minimum number of key terms required for relevance
min_key_terms_required = int(0.4 * len(filtered_searchTerms))

# Determine relevant documents based on search terms in titles
relevant_documents = []
for paper in mainPapers:
    title_lower = paper['text'].lower()
    # Count the number of key terms present in the title
    key_terms_in_title = sum(1 for term in filtered_searchTerms if term in title_lower)
    # Check if the paper has at least 70% of the key terms
    if key_terms_in_title >= min_key_terms_required:
        relevant_documents.append(paper)

print(filtered_searchTerms)
print(len(relevant_documents))



# Make recommendation
n_neighbor = 100 if len(mainPapers) > 100 else len(mainPapers) - 10;
recommended_documents = []
for i, doc in enumerate(tm.recommend(text=keyWords,  n_neighbors=n_neighbor), start=1):
  for p in mainPapers:
    if p['text'] == doc['text']:
      recommended_documents.append(p)
      print('Recommendation #%s'% i)
      print('Access Link: \t%s'% p['id'])
      print('Topic probability: \t%s'% doc['topic_proba'])
      print('Title: \t%s'% p['title'])
      print('Summary: \t%s'% p['summary'])
      print()



# Calculate precision and recall
relevant_set = set([doc['id'] for doc in relevant_documents])
recommended_set = set([doc['id'] for doc in recommended_documents])

true_positives = len(relevant_set.intersection(recommended_set))
false_positives = len(recommended_set - relevant_set)
false_negatives = len(relevant_set - recommended_set)
total_documents = len(relevant_set) + len(recommended_set)
true_negatives = total_documents - (true_positives + false_positives + false_negatives)

precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision > 0 or recall > 0) else 0
accuracy = (true_positives + true_negatives) / total_documents


print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)


# Visualizing the confusion matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Define the confusion matrix
conf_matrix = np.array([[true_positives, false_negatives],
                        [false_positives, true_negatives]])

# Plotting
labels = ['True Positives', 'False Negatives', 'False Positives', 'True Negatives']
conf_matrix = np.reshape(conf_matrix, (2,2))
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
