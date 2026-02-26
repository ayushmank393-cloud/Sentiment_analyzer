import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score

# Download VADER lexicon (only first time)
nltk.download('vader_lexicon')

# ==============================
# STEP 1: AMAZON → CLEAN CSV
# ==============================

input_file = "Reviews.csv"
clean_file = "amazon_clean.csv"

# Load Amazon dataset
data = pd.read_csv(input_file)

# Keep required columns
data = data[['Text', 'Score']]
data.rename(columns={'Text': 'review'}, inplace=True)

# Convert ratings to REAL sentiments
def score_to_sentiment(score):
    if score >= 4:
        return "Positive"
    elif score <= 2:
        return "Negative"
    else:
        return "Neutral"

data['real_sentiment'] = data['Score'].apply(score_to_sentiment)

# Remove rating column
data.drop(columns=['Score'], inplace=True)

# Save clean CSV automatically
data.to_csv(clean_file, index=False)
print("✅ amazon_clean.csv created")

# ==============================
# STEP 2: SENTIMENT ANALYSIS
# ==============================

# Load newly created CSV
data = pd.read_csv(clean_file)

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

# Predict sentiment
data['predicted_sentiment'] = data['review'].apply(predict_sentiment)

# Calculate accuracy using REAL sentiments
accuracy = accuracy_score(
    data['real_sentiment'],
    data['predicted_sentiment']
)

print("🎯 Accuracy:", accuracy)

# Save final output
data.to_csv("sentiment_output.csv", index=False)
print("✅ sentiment_output.csv created")