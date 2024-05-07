# Research Paper Recommendation System (RPRS)

## Overview
The Research Paper Recommendation System (RPRS) is a tool designed to help researchers find relevant papers from the arXiv repository. By leveraging topic modeling, the system analyzes paper titles and summaries to recommend the most relevant papers based on user-provided keywords.

## Core Features
- **Fetch Papers from arXiv**: The system retrieves a set of papers from the arXiv repository using user-defined search terms.
- **Preprocess Text Data**: It processes the text data to remove stopwords, tokenize, and stem words for further analysis.
- **Topic Modeling**: Using Latent Dirichlet Allocation (LDA), the system identifies key topics within the fetched papers.
- **Generate Recommendations**: Based on topic distributions and relevance scores, the system suggests papers that align with user interests.
- **Visualizations**: The system creates visual representations like word clouds and histograms to illustrate key patterns and insights.

## How It Works
1. **Fetch Data**: The system makes an API call to arXiv to retrieve papers based on search terms provided by the user.
2. **Preprocess Data**: It converts the XML data into a structured format and applies text preprocessing techniques like stopword removal and stemming.
3. **Build Topic Model**: An LDA-based topic model is built to identify topics in the dataset. The system also provides visualizations like word clouds.
4. **Recommend Papers**: Based on a relevance score derived from user-defined keywords, the system recommends the top papers to the user.
5. **Evaluate Recommendations**: The system evaluates its performance using metrics like precision, recall, and F1-score.

## Usage
To use the system, you need to:
1. Define search terms or keywords that describe the papers you are interested in.
2. Run the code to fetch papers from arXiv and build the topic model.
3. Review the recommended papers to find the ones most relevant to your research interests.

## Dependencies
To run this system, you need the following Python packages:
- `xmltodict`
- `ktrain`
- `gensim`
- `nltk`
- `wordcloud`
- `matplotlib`
