import os, sys
from tqdm.autonotebook import tqdm as tqdm
import requests, json
import hashlib
from bs4 import BeautifulSoup

categories = [
  { "category": "Stiri", "initial_url": "https://www.dcnews.ro/news.html" },
  { "category": "Coronavirus", "initial_url": "https://www.dcnews.ro/coronavirus.html" },
  { "category": "Politica", "initial_url": "https://www.dcnews.ro/politica.html" },
  { "category": "Tehnologie", "initial_url": "https://www.dcnews.ro/tehnologie.html" },
  { "category": "Economie", "initial_url": "https://www.dcnews.ro/economie-si-afaceri.html" },
  { "category": "Sanatate", "initial_url": "https://www.dcnews.ro/sanatate-magazin.html" },
  { "category": "Lifestyle", "initial_url": "https://www.dcnews.ro/stiri-lifestyle.html" },
  { "category": "Sport", "initial_url": "https://www.dcnews.ro/stiri-sport.html" },
  { "category": "Cultura", "initial_url": "https://www.dcnews.ro/cultura-news2.html" }
]

def get_indexes():
    for category in categories:
        print(f"Running category {category['category']}")

        # load current status
        status = {
            "list": [],
            "done": []
        }

        try:
            with open(f"raw/{category['category']}.json", "r", encoding="utf8") as f:
                status = json.load(f)
        except:
            print("    Index not found!")

        print(f"   This category has {len(status['list'])} pages processed.")
        current_page = category['initial_url']
        # current_page = 'https://www.dcnews.ro/politica_10000.html'
        now_at = 0
        break_crawl = False
        encountered_url_count = 0
        while break_crawl is False:
            print(f"      Getting {current_page} ...")
            page = requests.get(url=current_page)

            # run stuff
            news_count_on_page = 0
            soup = BeautifulSoup(page.text, 'html.parser')
            h2s = soup.find_all('h2')
            for h2 in h2s:
                links = h2.findChildren("a", recursive=False)
                if len(links) != 1:
                    continue
                url = links[0]['href']
                if "_" not in url:
                    continue
                if url in status["list"]:
                    encountered_url_count += 1
                    if encountered_url_count > 200:
                        print(f"   Encountered many already processed urls, breaking crawl ...")
                        break_crawl = True
                else:
                    status["list"].append(url)
                    status["done"].append(False)
                    news_count_on_page += 1

            print(news_count_on_page)
            if news_count_on_page == 0:
                print("    NO MORE NEWS!")
                break_crawl = True

            # prep next page
            if "_" in current_page:
                now_at = int(current_page[current_page.rfind("_") + 1:].replace(".html", "")) + 1
                current_page = current_page[:current_page.rfind("_")] + f"_{now_at}.html"
            else:
                current_page = current_page.replace(".html", "_2.html")

            with open(f"raw/{category['category']}.json", "w", encoding="utf8") as f:
                json.dump(status, f)


def get_pages():
    for category in categories:
        os.makedirs(f'raw/{category["category"]}', exist_ok=True)
        print(f"Running category {category['category']}")

        # load current status
        status = {
            "list": [],
            "done": []
        }

        with open(f"raw/{category['category']}.json", "r", encoding="utf8") as f:
            status = json.load(f)

        print(f"   This category has {len(status['list'])} links.")

        print(f"    Getting {len(status['list'])} news items:")
        cnt = 0

        for url in tqdm(status["list"]):
            index = status["list"].index(url)
            id = hashlib.md5(url.encode('utf-8')).hexdigest()

            if status["done"][index] == True or os.path.exists(f"{category['category']}/{id}.json"):
                status["done"][index] = True
                continue
            cnt += 1

            try:
                #print(url)
                page = requests.get(url=url)
                soup = BeautifulSoup(page.text, 'html.parser')
                h1 = soup.find('h1')
                # title
                title = h1.getText().strip()
                #print(title)

                # category
                subcategory = []
                breadcrumbs = soup.find("div", {"class": "breadcrumbs"})
                links = breadcrumbs.findChildren("a", recursive=False)
                for link in links:
                    subcategory.append(link.getText().strip())
                #print(subcategory)

                # date
                edate = soup.find("span", {"style": "font-size:14px;color:#656d78;float:right;"}).getText()
                date = edate[edate.find("|") + 1:].strip()
                #print(date)

                # text
                text = []
                ps = soup.find_all("p", {"style": "text-align:left;"})
                for p in ps:
                    # print(p)
                    #print(p.getText().strip())
                    text.append(p.getText().strip())

                # tags
                tags = []
                ts = soup.find_all("div", {"class": "article_tag"})
                for t in ts:
                    tags.append(t.getText().strip())
                #print(tags)

                data = {
                    "title": title,
                    "text": text,
                    "tags": tags,
                    "date": date,
                    "subcategory": subcategory
                }
                # break
                with open(f"raw/{category['category']}/{id}.json", "w", encoding="utf8") as f:
                    json.dump(data, f, indent=4)
                # break
                #print(f"Done {cnt} news in the {category['category']} category.")
                status["done"][index] = True
                if cnt % 100 == 0:
                    with open(f"raw/{category['category']}.json", "w", encoding="utf8") as f:
                        json.dump(status, f)
            except Exception as ex:
                status["done"][index] = "?"
                print(ex)

        with open(f"raw/{category['category']}.json", "w", encoding="utf8") as f:
            json.dump(status, f)

get_indexes()
get_pages()

