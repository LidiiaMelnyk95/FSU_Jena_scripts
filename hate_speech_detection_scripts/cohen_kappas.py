import os
import pandas as pd

def cohen_kappa(ann1, ann2):
    """Computes Cohen kappa for pair-wise annotators.
    :param ann1: annotations provided by the first annotator
    :type ann1: list
    :param ann2: annotations provided by the second annotator
    :type ann2: list
    :rtype: float
    :return: Cohen kappa statistic
    """
    count = sum(1 for a1, a2 in zip(ann1, ann2) if a1 == a2)
    A = count / len(ann1)  # observed agreement A (Po)

    uniq = set(ann1 + ann2)
    E = sum(((ann1.count(item) / len(ann1)) * (ann2.count(item) / len(ann2))) for item in uniq)

    return round((A - E) / (1 - E), 4)

# Use environment variables for file paths
file_path_ann1 = os.getenv("ANN1_FILE_PATH", '/Users/lidiiamelnyk/Downloads/Distribution_ukrainian_comments_MARIA_KHAR .csv')
file_path_ann2 = os.getenv("ANN2_FILE_PATH", '/Users/lidiiamelnyk/Downloads/Distribution_ukrainian_comments_puhach.csv')

# Read CSV files
ann1_pd = pd.read_csv(file_path_ann1, sep=',', encoding='utf-8-sig')
ann2_pd = pd.read_csv(file_path_ann2, sep=',', encoding='utf-8-sig')

# Assuming 'HATE/NO' column exists in both dataframes
ann1 = ann1_pd['HATE/NO'].tolist()
ann2 = ann2_pd['HATE/NO'].tolist()

my_cohen_kappa = cohen_kappa(ann1, ann2)
print(my_cohen_kappa)
