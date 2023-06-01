import os
import csv

from tqdm.auto import tqdm
import openai

'''
A label generator of training data for story classifier using ChatGPT
'''

# Config
openai.api_key = 'YOUR_API_KEY'
data_dir = "./data2"
label_path = "./raw_label.csv"


def generate_label(article) -> str:
    messages = [
        {"role": "system",
         "content": "你是一個要分類內容是否為故事的的助手，是的話回答1，不是的回答0，不需要多餘的解釋。以下是故事的定義：A series of events incited by conflict, which drives the protagonist/hero to find a solution, or a series of events developed from disbalance to rebalance."},
        {"role": "user",
         "content": f"內容：{article}"},
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )

    label = response["choices"][0]["message"]["content"]
    return label

                
def main():
    labels = {}

    # generate labels
    for filename in tqdm(os.listdir(data_dir)):
        if filename.endswith(".txt"):
            article_id = filename.split(".")[0]
            file_path = os.path.join(data_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                article = file.read()
                label = generate_label(article)
                labels[int(article_id)] = label

    # save
    with open(label_path, "w", newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(["id", "label"])

        for article_id in sorted(labels.keys()):
            writer.writerow([article_id, labels[article_id]])

if __name__ == "__main__":
    main()