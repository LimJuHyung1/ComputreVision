from icrawler.builtin import GoogleImageCrawler

def download_images(keyword, max_num, output_directory):
    google_crawler = GoogleImageCrawler(storage={'root_dir': output_directory})
    google_crawler.crawl(keyword=keyword, max_num=max_num)

# 예시로 엘사와 안나의 이미지를 다운로드
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
