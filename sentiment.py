import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

nltk.download('vader_lexicon')

# Create output folder
os.makedirs("output", exist_ok=True)

input_file = "Reviews.csv"
output_file = "output/sentiment_output.csv"

# Initialize VADER once
sia = SentimentIntensityAnalyzer()

# Remove output file if already exists
if os.path.exists(output_file):
    os.remove(output_file)

# Process CSV in chunks
chunk_size = 50000  # you can change: 10k / 50k / 100k

for chunk in pd.read_csv(input_file, chunksize=chunk_size):
    
    # Clean column names
    chunk.columns = chunk.columns.str.strip()

    # Keep required columns
    chunk = chunk[['Text', 'Score']]
    chunk.rename(columns={'Text': 'review'}, inplace=True)

    # REAL binary sentiment from rating
    chunk['real_sentiment'] = chunk['Score'].apply(
        lambda x: "Positive" if x >= 4 else "Negative"
    )
    chunk.drop(columns=['Score'], inplace=True)

    # Predicted binary sentiment
    chunk['predicted_sentiment'] = chunk['review'].apply(
        lambda x: "Positive"
        if sia.polarity_scores(str(x))['compound'] >= 0
        else "Negative"
    )

    # Append to output CSV
    chunk.to_csv(
        output_file,
        mode='a',
        index=False,
        header=not os.path.exists(output_file)
    )

print("✅ Large dataset processed successfully")
print("Output saved to:", output_file)
