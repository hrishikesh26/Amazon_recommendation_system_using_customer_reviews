# Amazon Product Recommendation System Using Customer Review Analysis

This repository contains code and documentation for an Amazon product recommendation system that leverages customer reviews to provide personalized product suggestions. The project emphasizes natural language processing (NLP) techniques to preprocess and analyze textual reviews and employs content-based filtering using cosine similarity as the recommendation engine.

## Table of Contents

- [Overview](#overview)
- [Motivation and Research Problem](#motivation-and-research-problem)
- [Dataset](#dataset)
- [Data Preprocessing and NLP Pipeline](#data-preprocessing-and-nlp-pipeline)
- [Sentiment Analysis and Classification](#sentiment-analysis-and-classification)
- [Recommendation System](#recommendation-system)
- [Results](#results)
- [Usage and Installation](#usage-and-installation)
- [Contributors](#contributors)
- [License](#license)

## Overview

This project tackles the challenge of filtering through large volumes of Amazon customer reviews to provide high-quality product recommendations. Our approach involves two key components:

1. **NLP for Customer Review Analysis:**  
   - Preprocessing textual reviews (including cleaning, tokenization, and vectorization).
   - Performing sentiment analysis to create a binary sentiment label for each review.
   - Evaluating multiple classification models to assess user sentiment.

2. **Content-Based Recommendation:**  
   - Building recommendation models based on cosine similarity computed over TF-IDF vectorized reviews.
   - Experimenting with n-gram models (unigram, bigram, trigram, and Genism-based representations) to improve the recommendation quality.

## Motivation and Research Problem

In an increasingly digital marketplace, buyers rely heavily on online reviews to inform purchasing decisions. However, the sheer volume of reviews makes it challenging to identify trustworthy opinions and similar product alternatives. Our project addresses this by:

- **Enhancing the Online Shopping Experience:** Tailoring recommendations based on user reviews and sentiment.
- **Maintaining Recommendation Quality:** Filtering out products that receive consistently low ratings.
- **Fostering Long-term Customer Loyalty:** Delivering personalized and reliable product suggestions that meet users’ preferences.

## Dataset

The project utilizes a large-scale dataset sourced from the [Stanford Network Analysis Project (SNAP)](https://snap.stanford.edu/data/web-Amazon.html). The key aspects of the dataset include:

- **Amazon Customer Reviews:**  
  - ~13 million records originally available.
  - A balanced subset of ~200K reviews (around 40K rows extracted from each rating class) is used after addressing imbalances.
  
- **Amazon Product Data:**  
  - Initially over 500K rows, streamlined to ~198K after handling null values.
  
- **Data Integration:**  
  - The review and product datasets are merged on the product identifier (parent_asin), resulting in a consolidated dataset of ~110K rows.

## Data Preprocessing and NLP Pipeline

The NLP work in the project encompasses:

- **Data Cleaning:**  
  - Removing null values and duplicate records.
  - Standardizing text inputs and handling imbalanced classes.

- **Text Preprocessing:**  
  - **Stopword Removal:** Eliminates common words to reduce noise.
  - **Tokenization:** Breaks review texts into individual tokens.
  - **TF-IDF Vectorization:** Converts reviews into numerical representations using unigram, bigram, and trigram models.
  - **Padding and Normalization:** Ensures consistent input dimensions and scale.
  - **Truncated SVD:** Reduces the dimensionality of the sparse matrix to facilitate efficient similarity computation.

- **Sentiment Analysis:**  
  - A sentiment score is computed for each review.
  - A binary `Sentiment_Label` column is created based on the computed sentiment (1 for positive, 0 for negative).

These steps ensure the data is well-prepared for both classification and recommendation tasks.

## Sentiment Analysis and Classification

To determine the impact of review sentiment on product recommendations, multiple classification models were evaluated. These include:

- **Decision Tree**
- **K-Nearest Neighbour (KNN)**
- **Random Forest**
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Gradient Boosting**
- **AdaBoost**
- **XGBoost**
- **Multinomial Naive Bayes**
- **Categorical Naive Bayes**

Each model was assessed using key performance metrics such as Accuracy, Recall, F1 Score, and Precision, to inform the integration of the sentiment analysis within the recommendation engine.

## Recommendation System

Our recommendation system is built on content-based filtering. The main steps include:

1. **Vector Representation of Reviews:**  
   - The cleaned reviews are vectorized using TF-IDF for unigrams, bigrams, trigrams, and an alternative representation via Genism.
   
2. **Cosine Similarity Calculation:**  
   - The similarity between products is measured by computing cosine similarity on the TF-IDF vectors.
   
3. **Evaluation via NDGC:**  
   - The system’s performance is evaluated using the Normalized Discounted Cumulative Gain (NDGC) metric.
   - Reported NDGC scores are:
     - **Unigram:** 0.98
     - **Bigram:** 0.99
     - **Trigram:** 0.97
     - **Genism:** 0.98

These high NDGC scores confirm that the recommendation system effectively identifies products with similar review profiles, enhancing the overall user experience.

## Results

The project demonstrates that by combining robust NLP techniques with an effective recommendation algorithm, it is possible to create a system that delivers personalized product suggestions based on the sentiment and content of Amazon reviews. Key highlights include:

- **Balanced Classification:** Successful evaluation of multiple models, with competitive performance among Logistic Regression, SVM, and Gradient Boosting.
- **Effective Recommendations:** High NDGC scores from the content-based filtering approach indicate precise matching of similar products.

## Usage and Installation

To run this project on your local machine:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/<your-username>/amazon-recommender-system.git
   cd amazon-recommender-system
2. Set Up Your Environment:

  - It is recommended to use a virtual environment.

  - Install the necessary dependencies:

```bash
Copy
pip install -r requirements.txt
```
3. Data Preparation:

  - Download the dataset from [SNAP](https://snap.stanford.edu/data/web-Amazon.html) if not already provided.

  - Ensure that the review and product CSV/JSON files are placed in the correct directory.

4. Running the Notebooks:

  - The repository includes several Jupyter Notebook files for:

  - Exploratory Data Analysis and Sentiment Analysis (`Amazon_Fine_Food_Reviews_Analysis.ipynb`)

  - Content-Based Recommendation using Cosine Similarity (`Content-based-filtering using cosine similarity.ipynb`)

  - Main recommendation system implementation (`Main - cosine similarity - recommendation system.ipynb`)

  - Launch Jupyter Notebook:

  ```bash
  jupyter notebook
  ```
  - Open and execute the desired notebooks sequentially.

## **Contributors**
- Simran Mhaske 

  - Hrishikesh Pawar 

  - Harshal Sawant 

  - Ishan Prabhune 

  - Yashraj Diwate 

  - Pratik Patil 

## **License**
This project is licensed under the MIT License. See the LICENSE file for details.
