import os
import csv
import time
from typing import List, Tuple

import openai
from openai.error import RateLimitError
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
from tqdm.auto import tqdm

'''
A label generator of training data for story classifier using ChatGPT
'''

# Config
openai.api_key = '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'
data_dir = "./data50"
label_path = "./raw_label50.csv"
batch_size = 5


def generate_label(article) -> str:
    '''
    Deprecated: need to implement retry mechanism to prevent openai.RateLimitError
    '''

    messages = [
        {"role": "system",
         "content": "你是一個要分類文章是否為故事的的助手。以下是故事的定義：A series of events incited by conflict, which drives the protagonist to find a solution, or a series of events developed from disbalance to rebalance. 接下來我會給你一篇文章，是的話回答1，不是的回答0，不要有多餘解釋。"}, 
        {"role": "user",
         "content": f"文章：{article}"},
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )

    label = response["choices"][0]["message"]["content"]
    return label


@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(3), reraise=True, retry=retry_if_exception_type(RateLimitError))
def generate_batched_labels(articles: List[str]) -> List[str]:
    '''
    To balance between the cost and efficiency, batch multple articles into one request
    Returns string is usually (but not guaranteed) labels each of which seperated by a new line character
    '''

    system_prompt = f"你是一個要分類文章是否為故事的的助手。以下是故事的定義：A series of events incited by conflict, which drives the protagonist to find a solution, or a series of events developed from disbalance to rebalance. 接下來我會給你{len(articles)}篇文章，對於每篇文章請根據故事的定義判斷是否為故事，是的話回答1，不是的回答0，不要有多餘解釋，每篇文章的答案請用一個空白符號分開。"

    user_prompt = ""
    for i, article in enumerate(articles):
        user_prompt += f"{i+1}: {article}\n"
    
    messages = [
        {"role": "system",
        "content": system_prompt,},
        {"role": "user",
         "content": user_prompt},
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )

    content = response["choices"][0]["message"]["content"]
    labels = content.split(" ")

    return labels


def get_articles(data_dir) -> Tuple[int, str]:
    '''
    A generator that outputs (article id, article)
    '''

    # sort the filenames
    filenames = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    filenames = sorted(filenames, key=lambda f: int(f.split(".")[0]))    # key: 13.txt -> 13

    for f in tqdm(filenames):
        fid = int(f.split(".")[0])
        with open(os.path.join(data_dir, f), "r") as fp:
            article = fp.read()
        yield fid, article


def main():
    labels = {}
    batched_article_ids = []
    batched_articles = []

    # generate labels
    articles = get_articles(data_dir)
    for article_id, article in articles:
        batched_article_ids.append(article_id)
        batched_articles.append(article)

        #  batch large enough, send request to chatGPT
        if len(batched_articles) >= batch_size:

            # Unexpected Error:
            # - gpt answer format error (seldom)
            #     + case 1: length not matched
            #     + case 2: content not 0 or 1
            #     + sol: pass
            # - rate limit error (sometimes)
            #     + cause:
            #         * quota exceeded
            #         * openai server overloaded
            #     + sol: retry some times then pass
            try:
                batched_labels = generate_batched_labels(batched_articles)
                if len(batched_labels) == len(batched_article_ids):
                    for aid, label in zip(batched_article_ids, batched_labels):
                        labels[aid] = label
            except:
                pass

            batched_articles, batched_article_ids = [], []
    

    # save labels
    with open(label_path, "w", newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(["id", "label"])

        for article_id in sorted(labels.keys()):
            writer.writerow([article_id, labels[article_id]])

if __name__ == "__main__":
    main()