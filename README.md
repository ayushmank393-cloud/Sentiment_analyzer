## Overview

This project analyzes customer reviews using **Python and VADER sentiment analysis**.
It reads reviews from a **CSV file** and generates sentiment results in another **CSV file**.

## 🔨Tools Used

* Python
* Pandas
* NLTK (VADER)
* Scikit-learn

## 📊Dataset

* Source: Kaggle – Amazon Fine Food Reviews
* Input columns used:

  * `Text` (review)
  * `Score` (rating)

## 🪛How It Works

* Reads full CSV dataset
* Converts ratings to real sentiments
* Predicts sentiment using VADER
* Saves results to output CSV

## 🔩How to Run
pip install pandas nltk scikit-learn
python sentiment.py

## ❗Output

* File: `output/sentiment_output.csv`
* Contains:

  * review
  * real_sentiment
  * predicted_sentiment

##📝 Note
No UI is used. The system works as a **backend CSV processing pipeline**, which is standard in industry projects.


