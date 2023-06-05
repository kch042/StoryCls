# 2023 AI Final Project - Story Classifier

A transformer-based Chinese story classfier.

## Data
- source: [臺灣文化部「國民記憶庫：臺灣故事島」](https://data.gov.tw/dataset/24967?page=1)
- split:
    - train
        - num: 11035
        - label: by ChatGPT
    - evaluation: 500
        - num: 500
        - label: by human


## Label Generation from ChatGPT
Inside `gptapi.py`, we encapsulate

- story definition
- output format
- `batch_size` stories

into 1 request to ChatGPT


For example, one request might contain messages like
```
[{  'role': 'system', 
    'content': '你是一個要分類文章是否為故事的的助手。以下是故事的定義：A series of events incited by conflict, which drives the protagonist to find a solution, or a series of events developed from disbalance to rebalance. 接下來我會給你2篇文章，對於每篇文章請根據故事的定義判斷是否為故事，是的話回答1，不是的回答0，不要有多餘解釋，每篇文章的答案請用一個換行符分開。'}, 
    
{   'role': 'user', 
    'content': 
    '1: 劉金獅先生，1935年出生於宜蘭，1962年被捕，歷經3天3夜疲勞審訊，於1963年以台獨叛亂罪判刑10年。在景美看守所期間擔任洗衣工廠外役。1972年出獄後，積極投入黨外運動，並組織「台灣政治受難者聯誼總會」。在景美看守所期間為1968-1972年。
    2: 陳松先生，1943年出生於浙江寧波，1965年因中國饑荒偷渡來臺，一度成了「反共義士」。1966年「共匪特務」判刑6年，在獄中擔任縫衣工廠外役，1972年出獄。在景美看守所期間為1967年-1972年。'
}]
```

We batch multiple articles into 1 request (instead of 1 request per article) to 
- reduce the api cost
- increase the label generation speed
- prevent rate limit error (openai has a limit on the number of requests per minute)


## Model
Fine-tuned on the Chinese roberta model [`hfl/chinese-roberta-wwm-ext`](https://huggingface.co/hfl/chinese-roberta-wwm-ext) from huggingface


## Result
- metric: accuracy
- test set: 0.562


## Usage

### Label Generation
Look up `gptapi.py`, modify `openai.api_key`, `data_dir` , `label_path`, and then run
```
$ python gptapi.py
```

### Training
```
$ ./run.sh
```