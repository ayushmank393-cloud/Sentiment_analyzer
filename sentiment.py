import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score
import os

nltk.download('vader_lexicon')

# Create output folder
os.makedirs("output", exist_ok=True)

# Load dataset
data = pd.read_csv("Reviews.csv")

# 🔥 FIX: clean column names
data.columns = data.columns.str.strip()

# Now this WILL work
data = data[['Text', 'Score']]
data.rename(columns={'Text': 'review'}, inplace=True)

# Convert rating to real sentiment
def score_to_sentiment(score):
    if score >= 4:
        return "Positive"
    elif score <= 2:
        return "Negative"
    else:
        return "Neutral"

data['real_sentiment'] = data['Score'].apply(score_to_sentiment)
data.drop(columns=['Score'], inplace=True)

# Initialize VADER
sia = SentimentIntensityAnalyzer()

def predict_sentiment(review):
    score = sia.polarity_scores(str(review))['compound']
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
data['predicted_sentiment'] = data['review'].apply(predict_sentiment)

# Accuracy
accuracy = accuracy_score(
    data['real_sentiment'],
    data['predicted_sentiment']
)

print("Total reviews:", len(data))
print("Accuracy:", accuracy)

# Save output CSV
data.to_csv("output/sentiment_output.csv", index=False)
print("✅ Output saved to output/sentiment_output.csv")