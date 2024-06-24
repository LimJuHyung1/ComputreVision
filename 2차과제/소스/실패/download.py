# 03조 (임주형,이세비,최하은)
from icrawler.builtin import GoogleImageCrawler

def download_images(keyword, max_num, output_directory):
    google_crawler = GoogleImageCrawler(storage={'root_dir': output_directory})
    google_crawler.crawl(keyword=keyword, max_num=max_num)

keywords = [
    ("Elsa Frozen", "frozen_characters/Elsa"),
    ("Elsa Frozen character", "frozen_characters/Elsa"),
    ("Frozen Elsa", "frozen_characters/Elsa"),
    ("Anna Frozen", "frozen_characters/Anna"),
    ("Anna Frozen character", "frozen_characters/Anna"),
    ("Frozen Anna", "frozen_characters/Anna")
]

for keyword, directory in keywords:
    download_images(keyword, max_num=500, output_directory=directory)
