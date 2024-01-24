import pandas as pd
import matplotlib.pyplot as plt

class CommentsAnalyzer:
    def __init__(self, file_paths):
        self.dictionaries = [pd.read_excel(file_path, sheet_name=None) for file_path in file_paths]

    def count_comments_length(self):
        length_dictionary = {"Name": [], "Length": []}

        for element in self.dictionaries:
            for key, sub_df in element.items():
                l = sub_df.apply(lambda row: self.count_non_nan_comments(row['Comment']) + self.count_non_nan_comments(row['Reply']), axis=1).sum()
                length_dictionary['Name'].append(key)
                length_dictionary['Length'].append(l)

        return length_dictionary

    def count_non_nan_comments(self, comment):
        return sum(1 for k in str(comment).split('\n') if k != 'nan')

    def plot_comments_length(self):
        length_dictionary = self.count_comments_length()

        plt.ylabel('Number of comments')
        plt.xlabel('Number of videos')
        plt.plot(sorted(length_dictionary['Length'], reverse=True))
        plt.show()


if __name__ == "__main__":
    file_paths = [
        '/Users/lidiiamelnyk/Downloads/Копия youtube comments.xlsx',
        '/Users/lidiiamelnyk/Downloads/youtube comments.xlsx',
        '/Users/lidiiamelnyk/Downloads/geschlechtsveränderung.xlsx'
    ]

    comments_analyzer = CommentsAnalyzer(file_paths)
    comments_analyzer.plot_comments_length()
