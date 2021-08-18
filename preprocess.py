import os, sys
from tqdm.autonotebook import tqdm as tqdm
import json, string

min_title_len = 10
min_text_len = 400
min_sentences_len = 1

categories = [
  { "category": "Tehnologie", "initial_url": "https://www.dcnews.ro/tehnologie.html" },
  { "category": "Economie", "initial_url": "https://www.dcnews.ro/economie-si-afaceri.html" },
  { "category": "Sanatate", "initial_url": "https://www.dcnews.ro/sanatate-magazin.html" },
  { "category": "Lifestyle", "initial_url": "https://www.dcnews.ro/stiri-lifestyle.html" },
  { "category": "Sport", "initial_url": "https://www.dcnews.ro/stiri-sport.html" },
  { "category": "Stiri", "initial_url": "https://www.dcnews.ro/news.html" },
  { "category": "Coronavirus", "initial_url": "https://www.dcnews.ro/coronavirus.html" },
  { "category": "Politica", "initial_url": "https://www.dcnews.ro/politica.html" },
  { "category": "Cultura", "initial_url": "https://www.dcnews.ro/cultura-news2.html" }
]

import re

regex = r"([a-z1-9îșțăâ}\)\]]{2,}\.)([A-ZÎȘȚĂÂ]{1,})"  # r"([a-z]{4,}\.)([A-Z][a-z]{1,})"
subst = "\\g<1> \\g<2>"

# filter and convert each category in the preprocessed folder
def preprocess ():
    for category in categories:
        files = os.listdir("raw/"+category['category'])
        os.makedirs("preprocessed/"+category['category'], exist_ok=True)

        for file in tqdm(files):
            with open("raw/"+category['category']+"/"+file,"r",encoding="utf8") as f:
                data = json.load(f)

            data["title"] = data["title"].replace("EXCLUSIV", "").strip()

            if " " not in data["title"] or len(data["title"])<10:
                continue

            if data["title"][-1] in string.punctuation and data["title"][-1] not in "`'\"":
                data["title"] = data["title"][:-1].strip()
            if data["title"][0] == ":":
                data["title"] = data["title"][1:].strip()

            data["title"] = data["title"].replace("FOTO", "").strip()

            if len(data["text"]) == 0:
                continue

            text = ""
            for p in data["text"]:
                if p.strip() == "":
                    continue

                if p.lower().endswith("video") or p.lower().endswith("video:"):
                    continue

                if "citeste si" in p.lower() or "citește și" in p.lower() or "galerie foto" in p.lower():
                    continue

                if p.startswith("Vezi și"):
                    continue

                if p.startswith("Citește") or p.startswith("Citeste"):
                    continue

                if p.endswith("-"):
                    continue

                p = p.replace("UPDATE:","").replace("UPDATE","").replace("  "," ").strip()

                text += p + "\n"


            if sum([1 for _ in re.finditer(regex, text, re.MULTILINE)]) > 0:
                #print("\n\nFound: ")
                #print(text)
                #print()
                text = re.sub(regex, subst, text, 0, re.UNICODE)
                #print(text)
                #print("---------------------")
                #break

            text = text.replace("Ş","Ș").replace("ş","ș").replace("Ţ","Ț").replace("ţ","ț")
            data["text"] = text.strip()

            if len(data["text"]) < 200:
                continue

            if len(data["subcategory"])>0:
                if data["subcategory"][0].startswith("Cele"):
                    data["subcategory"] = data["subcategory"][1:]

            with open("preprocessed/"+category['category']+"/"+file, "w", encoding="utf8") as f:
                json.dump(data,f,ensure_ascii=False,indent=4)


def aggregate_in_single_file ():
    all = []
    for category in categories:
        files = os.listdir("preprocessed/" + category['category'])

        for file in tqdm(files,desc="preprocessed/" + category['category']):
            with open("preprocessed/" + category['category'] + "/" + file, "r", encoding="utf8") as f:
                data = json.load(f)
            data["category"] = category['category']
            all.append(data)

    with open("preprocessed/all.json", "w", encoding="utf8") as f:
        json.dump(all, f, ensure_ascii=False, indent=4)
    print(len(all))

#preprocess()

aggregate_in_single_file()