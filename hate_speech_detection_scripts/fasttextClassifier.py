import csv
import fasttext

class FastTextClassifier:
    def __init__(self, train_file, model_file="model.bin"):
        self.train_file = train_file
        self.model_file = model_file

    @staticmethod
    def load_data(path):
        with open(path, "rU", encoding="utf-8") as file:
            data = file.readlines()
        return [line.split(";") for line in data]

    @staticmethod
    def save_data(path, data):
        with open(path, 'w', encoding="utf-8") as f:
            f.write("\n".join(data))

    def transform(self, input_file, output_file):
        data = self.load_data(input_file)
        data = [f"__label__{line[1].rstrip()}\t{line[0]}" for line in data]
        self.save_data(output_file, data)

    def train(self, training_params):
        model = fasttext.train_supervised(input=self.train_file, **training_params)
        model.save_model(self.model_file)
        return model

    @staticmethod
    def test(model, test_file='fasttext.test'):
        f1_score = lambda precision, recall: 2 * ((precision * recall) / (precision + recall))
        nexamples, recall, precision = model.test(test_file)
        print(f'recall: {recall}')
        print(f'precision: {precision}')
        print(f'f1 score: {f1_score(precision, recall)}')
        print(f'number of examples: {nexamples}')


if __name__ == "__main__":
    # For Ukrainian
    ukr_classifier = FastTextClassifier(train_file='/Users/lidiiamelnyk/Documents/ukr_comment_one_sentence.csv')
    ukr_classifier.transform('/Users/lidiiamelnyk/Documents/ukr_comment_one_sentence.csv', "fasttext_ukr.train")
    training_params_ukr = {'epoch': 5000, 'lr': 0.85, 'wordNgrams': 1, 'verbose': 2,
                            'minCount': 1, 'loss': "ns", 'lrUpdateRate': 100, 'thread': 1, 'ws': 5, 'dim': 100}
    trained_model_ukr = ukr_classifier.train(training_params_ukr)
    ukr_classifier.test(trained_model_ukr)

    # For Russian
    rus_classifier = FastTextClassifier(train_file='/Users/lidiiamelnyk/Documents/ru_comment_one_sentence.csv')
    rus_classifier.transform('/Users/lidiiamelnyk/Documents/ru_comment_one_sentence.csv', "fasttext_rus.train")
    training_params_rus = {'epoch': 50000, 'lr': 0.85, 'wordNgrams': 1, 'verbose': 2,
                            'minCount': 1, 'loss': "ns", 'lrUpdateRate': 100, 'thread': 1, 'ws': 5, 'dim': 100}
    trained_model_rus = rus_classifier.train(training_params_rus)
    rus_classifier.test(trained_model_rus)
