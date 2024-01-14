import os

config = {
    'russian_url': os.getenv('RUSSIAN_URL'),
    'ukrainian_url': os.getenv('UKRAINIAN_URL'),
    'russian_corpus_path': os.getenv('RUSSIAN_CORPUS_PATH'),
    'comments_path': os.getenv('ALL_COMMENTS_PATH'),
    'russian_output_path': os.getenv('RUSSIAN_OUTPUT_PATH'),
    'zn_ua_folder': os.getenv('ZN_UA_FOLDER')
}
