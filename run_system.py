import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score

# Download VADER lexicon (run once)
nltk.download('vader_lexicon')

# Load dataset
data = pd.read_csv("reviews.csv")
data.rename(columns={data.columns[0]: "review"}, inplace=True)

# Initialize VADER
sia = SentimentIntensityAnalyzer()

# Predict sentiment
def predict_sentiment(review):
    score = sia.polarity_scores(str(review))['compound']
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Apply prediction
data['predicted_sentiment'] = data['review'].apply(predict_sentiment)

# Calculate accuracy using REAL sentiments
accuracy = accuracy_score(
    data['real_sentiment'],
    data['predicted_sentiment']
)

print("Accuracy:", accuracy)
print(data)
