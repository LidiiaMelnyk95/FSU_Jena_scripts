import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List
import torch
import re
import typing

class SentimentAnalysisProcessor:
    def __init__(self, model_name: str = "oliverguhr/german-sentiment-bert"):
        self.model_name = model_name
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.clean_chars = re.compile(r'[^A-Za-züöäÖÜÄß ]', re.MULTILINE)
        self.clean_http_urls = re.compile(r'https*\S+', re.MULTILINE)
        self.clean_at_mentions = re.compile(r'@\S+', re.MULTILINE)

    def predict_sentiment(self, texts: List[str]) -> List[str]:
        texts = [self.clean_text(text) for text in texts]
        input_ids = self.tokenizer.batch_encode_plus(texts, padding=True, add_special_tokens=True, truncation=True)
        input_ids = torch.tensor(input_ids["input_ids"]).to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids)

        label_ids = torch.argmax(logits[0], axis=1)

        labels = [self.model.config.id2label[label_id] for label_id in label_ids.tolist()]
        return labels

    def replace_numbers(self, text: str) -> str:
        return text.replace("0", " null").replace("1", " eins").replace("2", " zwei") \
            .replace("3", " drei").replace("4", " vier").replace("5", " fünf") \
            .replace("6", " sechs").replace("7", " sieben").replace("8", " acht") \
            .replace("9", " neun")

    def clean_text(self, text: str) -> str:
        text = text.replace("\n", " ")
        text = self.clean_http_urls.sub('', text)
        text = self.clean_at_mentions.sub('', text)
        text = self.replace_numbers(text)
        text = self.clean_chars.sub('', text)
        text = ' '.join(text.split())
        text = text.strip().lower()
        return text


def process_comments(input_path, output_path):
    data = pd.read_csv(input_path, sep=',', error_bad_lines=False)
    data['Comment'] = data['Comment'].astype(str)

    sentiment_processor = SentimentAnalysisProcessor()

    for i, row in data.iterrows():
        sentiment_list = []
        for comment_segment in row['Comment'].split('.'):
            cleaned_segment = sentiment_processor.clean_text(comment_segment)
            sentiment = sentiment_processor.predict_sentiment([cleaned_segment])
            sentiment_list.append(sentiment[0])

        data.at[i, 'Sentiment'] = ','.join(sentiment_list)

    new_columns = ['Name', 'Comment', 'Time', 'Sentiment']
    data = data.reindex(columns=new_columns)

    with open(output_path, 'w+', encoding='utf-8-sig', newline='') as my_file:
        data.to_csv(my_file, sep=',', na_rep='', float_format=None,
                    header=True, index=False, index_label=None,
                    mode='a', compression='infer',
                    quoting=None, quotechar='"', line_terminator=None, chunksize=None,
                    date_format=None, doublequote=True, escapechar=None, decimal='.')


if __name__ == "__main__":
    input_path = './youtube_comments.csv'
    output_path = './youtube_comments_sentiment.csv'

    process_comments(input_path, output_path)
