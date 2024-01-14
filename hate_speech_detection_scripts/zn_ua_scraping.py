import concurrent.futures
import pandas as pd
import requests
from bs4 import BeautifulSoup
from config import config

url_list = ['https://zn.ua/theme/14786/', 'https://zn.ua/theme/74/', 'https://zn.ua/theme/69/']
columns_names = ['url', 'comment', 'date', 'name']


def construct_topics_urls(url):
    topic_urls = []
    i = 1

    while True:
        url_new = f"{url}p{i}"
        response = requests.get(url_new, timeout=5)
        content = BeautifulSoup(response.content, "html.parser")

        if not content.find('div', attrs='sbody'):
            break

        topic_urls.append(url_new)
        i += 1

    return topic_urls


def scrape_text(url):
    init_url = url
    comment_array = []

    def request_comment_data(page=1):
        response = requests.get(url, timeout=30)
        content = BeautifulSoup(response.content, "html.parser")

        try:
            article = content.find('div', attrs="comment_block_art") \
                .find('div', attrs="comments_list") \
                .find('ul', attrs="main_comments_list")['data-absnum']

            try:
                x = requests.post(url='https://zn.ua/actions/comments/',
                                  data={'article': article, 'page': page, 'typeload': 4, 'comtype': 1}).json()
            except:
                x = requests.post(url='https://zn.ua/actions/comments/', data={'article': article, 'page': 1, 'typeload': 1, 'comtype': 1}).json()
            json_file = x

        except AttributeError and ValueError:
            return None

        if json_file['comments']['success']:
            return json_file
        else:
            print('No comments found here')
            return None

    def get_comments_text():
        file = request_comment_data()
        if file is None:
            return [], None
        pages_count = file['comments']['result']['pages']
        comment_array = []
        page_counter = 1
        while page_counter <= pages_count:
            to_parse = file['comments']['result']['html']
            content = BeautifulSoup(to_parse, "html.parser")
            parsed_comment = content.find_all('li', attrs='comment_item')
            try:
                for comment in parsed_comment:
                    comment_text = comment.find('span', attrs='comment_text_block').find('span',
                                                                                         attrs='comment_txt').get_text()
                    parsed_name = comment.find('span', attrs='user_info_block').find('span',
                                                                                     attrs='user_nickname').get_text()
                    parsed_date = comment.find('span', attrs='user_info_block').find('span',
                                                                                     attrs='comment_time').get_text()
                    comment_array.append({'comment': comment_text, 'date': parsed_date, 'author_name': parsed_name})
            except AttributeError:
                pass
            page_counter += 1
            file = request_comment_data(page_counter)

        return comment_array

    while url is not None:
        comments = get_comments_text()
        comment_array = comment_array + comments

    rows = []
    df1 = pd.DataFrame(columns=columns_names)
    for item in comment_array:
        rows.append({'url': init_url, 'comment': item.get('comment'), 'date': item.get('date'),
                     'name': item.get('author_name')})
    df1 = df1.append(rows, ignore_index=True)
    df1 = df1.drop_duplicates()
    count = df1['comment'].count()
    if count <= 0:
        return
    filename = f"{config['zn_ua_folder']}{hash(init_url)}.csv"
    df1.to_csv(filename, sep=',', na_rep='', float_format=None)
    return f"Comments count {count}"


def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=500) as executor:
        futures = []

        for url in url_list:
            topic_urls_main = construct_topics_urls(url)
            print(f"Topic URLs amount {len(topic_urls_main)}")

            url_array = set()
            for topic_url in topic_urls_main:
                response = requests.get(topic_url, timeout=60)
                content = BeautifulSoup(response.content, "html.parser")
                main_div = content.find('div', attrs='sbody') \
                    .find('div', id="container") \
                    .find('div', id='holder') \
                    .find('div', id='left')

                try:
                    news_sections = main_div.find('div', attrs="left_news_list section").find('ul',
                                                                                              attrs='news_list').find_all(
                        'li')

                    url_array.update(['https://zn.ua' + news.find('a').get('href') for news in news_sections])

                except AttributeError:
                    print("Cannot find news_section")
                    continue

            for future_url in url_array:
                futures.append(executor.submit(scrape_text, url=future_url))

        for future in concurrent.futures.as_completed(futures):
            if future.result():
                print(future.result())


if __name__ == '__main__':
    try:
        main()
    except ConnectionResetError:
        pass
