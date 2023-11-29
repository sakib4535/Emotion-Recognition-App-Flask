from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField
from wtforms.validators import DataRequired
from flask import Flask, render_template, send_file
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import spacy
from spacy import displacy
import os
from collections import Counter
from plot_generator import generate_emotion_plots

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nlp = spacy.load("en_core_web_sm")


app = Flask(__name__)
app.config['WTF_CSRF_ENABLED'] = False

class InputForm(FlaskForm):
    text = TextAreaField('Text', validators=[DataRequired()])
    submit = SubmitField('Analyze')


def analyze_emotion(text):
    analysis = TextBlob(text)
    sentiment = analysis.sentiment
    sentiment_assessment = analysis.sentiment_assessments

    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(text)

    if scores['compound'] >= 0.05:
        emotion = "Positive or Joyful"
    elif scores['compound'] <= -0.05:
        emotion = "Negative or Angry"
    else:
        emotion = "Neutral or Calm"


    doc = nlp(text)
    ner_tags = [(ent.text, ent.labels_) for ent in doc.ents]
    word_count = len(analysis.words)

    words = nltk.word_tokenizer(text)
    top_10_words = Counter(words).most_common(10)



    return sentiment, sentiment_assessment, emotion, word_count, ner_tags, top_10_words


def calculate_accuracy(predicted, ground_truth):
    return (predicted / ground_truth) * 100


@app.route('/', methods=['GET', 'POST'])
def index():

    form = InputForm()

    if form.validate_on_submit():
        text = form.text.data
        sentiment, sentiment_assessments, emotion, word_count, ner_tags, top_10_words = analyze_emotion(text)
        generate_emotion_plots(sentiment, emotion, word_count, ner_tags, top_10_words)

        # Sample accuracy calculation (replace these values with your actual data)
        # Replace with ground truth polarity and predicted polarity

        ground_truth_polarity = 0.75
        predicted_polarity = sentiment.polarity

        accuracy = calculate_accuracy(predicted_polarity, ground_truth_polarity)

        return render_template('result.html', text=text, sentiment=sentiment,
                               sentiment_assessments=sentiment_assessments,
                               emotion=emotion, word_count=word_count, ner_tags=ner_tags,
                               accuracy=accuracy
                               )

    return render_template('main.html', form=form)


@app.route('/show_plot/<plot_name>', methods=['GET'])
def show_plot(plot_name):
    plot_dir = 'F:/Pycharm Central Zone/LLM/image/'  # Update with your plot directory
    plot_path = os.path.join(plot_dir, f'{plot_name}.png')
    if not os.path.isfile(plot_path):
        return "Invalid plot request"
    return send_file(plot_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)

