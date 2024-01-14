
from bs4 import BeautifulSoup
import requests
from config import config
import os
import time

def get_text(url):
    try:
        response = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
        response.raise_for_status()
        content = BeautifulSoup(response.content, "html.parser")
        comment_text_array = [p.get_text() for p in content.find('div', class_='text').find_all('p')]
        return comment_text_array
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {url}, Error: {e}")
        return []

def scrape_page(page_num, language):
    url = f"{config['russian_url']}{page_num}" if language =='russian' else f"{config['ukrainian_url']}{page_num}"
    try:
        response = requests.get(url, timeout=30, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
        response.raise_for_status()
        content = BeautifulSoup(response.content, "html.parser")
        main_div = content.find('div', class_="curpane")
        news_sections = main_div.find_all('article', class_="item type1")
        url_array = list({news.find('h3').find('a').get('href') for news in news_sections})
        return url_array
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {url}, Error: {e}")
        return []

def main(language):
    path = os.path.join(os.getcwd(), f'articles_{language}.txt')
    with open(path, 'a', encoding='utf-8-sig') as myfile:
        page_num = 1
        while True:
            url_array = scrape_page(page_num, language)
            if not url_array:  # Break if no more pages are found
                break
            for url in url_array:
                comments = get_text(url)
                for row in comments:
                    myfile.write(f"{row}\n")
            time.sleep(2)  # Adding a delay to avoid overloading the server
            page_num += 1


if __name__ == '__main__':
    main()
    # while True:
    #    main()
    #    time.sleep(3600)