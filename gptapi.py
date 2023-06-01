import os
import openai
import csv

openai.api_key = 'The_API_KEY'

def generate_response(article):
    prompt = f"Following is the definition of a story: A series of events incited by conflict, which drives the protagonist/hero to find a solution, or a series of events developed from disbalance to rebalance. Please determine if the following article is a story or not. If it is, please return 1; if it isn’t, please return 0. Here is the article: {article}\nChatGPT: "

    response = openai.Completion.create(
        engine='text-davinci-003',  
        prompt=prompt,
        max_tokens=200, 
        temperature=0.7,
        n=1,
        stop=None,
    )
        
    chatbot_response = response.choices[0].text.strip().split('ChatGPT:', 1)[-1].strip()
    return chatbot_response


csv_file = 'C:/Users/USER/Desktop/result.csv'
with open(csv_file, 'w', newline='') as cfile:
    writer = csv.writer(cfile)
    writer.writerow(['Article Number', 'Label'])

    number = 0
    directory = "C:/Users/USER/Desktop/data"
    file_list = os.listdir(directory)
    for filename in file_list:
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                article = file.read()
                label = generate_response(article)
                writer.writerow([number, label])
                number = number + 1
                
                
                
