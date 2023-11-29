import matplotlib.pyplot as plt
import os


def generate_emotion_plots(sentiment, emotion, word_count, ner_tags, top_10_words):
    plot_dir = 'F:/Pycharm Central Zone/LLM/image/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    emotions = ['Positive', 'Negative', 'Neutral']
    emotion_counts = [emotion.count(e) for e in emotions]

    # Emotion Distribution plot
    plt.figure(figsize=(8, 6))
    plt.pie(emotion_counts, labels=emotions, autopct='%1.1f%%', startangle=140)
    plt.title('Emotion Distribution')
    plt.savefig(os.path.join(plot_dir, 'emotion_distribution.png'))
    plt.close()

    # Sentiment Polarity plot
    plt.figure(figsize=(8, 6))
    plt.bar(['Polarity'], [sentiment.polarity], color='skyblue')
    plt.title('Sentiment Polarity')
    plt.savefig(os.path.join(plot_dir, 'sentiment_polarity.png'))
    plt.close()

    # Word Count Distribution plot
    plt.figure(figsize=(8, 6))
    plt.hist(word_count, bins=10, color='orange')
    plt.title('Word Count Distribution')
    plt.savefig(os.path.join(plot_dir, 'word_count_distribution.png'))
    plt.close()

    if ner_tags:  # Check if ner_tags is not empty
        # Named Entity Recognition (NER) Distribution plot
        ner_labels, _ = zip(*ner_tags)
        ner_counts = {label: ner_labels.count(label) for label in set(ner_labels)}
        plt.figure(figsize=(10, 6))
        plt.bar(ner_counts.keys(), ner_counts.values(), color='green')
        plt.title('Named Entity Recognition (NER) Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'ner_distribution.png'))
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.scatter([word[0] for word in top_10_words], [word[1] for word in top_10_words], color='red')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.title('Top 10 Word Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'top_10_word_frequency.png'))
        plt.close()