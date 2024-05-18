# from keybert import KeyBERT
import openai 
import webbrowser
import os
from konlpy.tag import Okt
import re
from krwordrank.word import KRWordRank
from dotenv import load_dotenv
import cv2
import json
from pathlib import Path
from base64 import b64decode
from selenium import webdriver


load_dotenv()
Dalle_Key = os.environ.get('MySecret')

def split_noun_sentences(text):
    okt = Okt()
    sentences = text.replace(". ",".")
    sentences = re.sub(r'([^\n\s\.\?!]+[^\n\.\?!]*[\.\?!])', r'\1\n', sentences).strip().split("\n")
    
    result = []
    for sentence in sentences:
        if len(sentence) == 0:
            continue
        sentence_pos = okt.pos(sentence, stem=True)
        nouns = [word for word, pos in sentence_pos if pos == 'Noun']
        if len(nouns) == 1:
            continue
        result.append(' '.join(nouns) + '.')
        
    return result

doc = "엄마가 집에서 맛있는 카레를 해줘서 기분이 너무 좋았던 남자"
#'a man who go to school flying dragon'
#'''a man who felt so good because my mom made delicious curry at home.'''
#"""
#A man who learned physical education, science, math, and Korean.
#"""
#doc = "엄마가 집에서 맛있는 카레를 해줘서 기분이 너무 좋았던 사람"

'''
min_count = 0   # 단어의 최소 출현 빈도수 (그래프 생성 시)
max_length = 10 # 단어의 최대 길이
wordrank_extractor = KRWordRank(min_count=min_count, max_length=max_length)
beta = 0.8    # PageRank의 decaying factor beta
max_iter = 20
texts = split_noun_sentences(doc)
keywords, rank, graph = wordrank_extractor.extract(texts, beta, max_iter)

inputPrompt = ""
for word, r in sorted(keywords.items(), key=lambda x:x[1], reverse=True):
    inputPrompt = inputPrompt + word + ","
    print('%8s:\t%.4f' % (word, r))


print(inputPrompt)
'''

# inputPrompt = "기분, 우울, 학교, 비, 우산"

client = openai.OpenAI(api_key = Dalle_Key)

user_info = "korean,8 age,man"
end_of_prompt = "no language,within simple cartoon,only draw,no color, I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use it AS-IS"
# simple sketch
# no color

response = client.images.generate(
  model="dall-e-3",
  prompt=user_info+doc+end_of_prompt,#user_info+doc+end_of_prompt,#user_info+inputPrompt+end_of_prompt,
  size="1024x1024",
  quality="standard",
  n=1,
)
print(response.data[0].url)
webbrowser.open(response.data[0].url)
response = client.images.generate(
  model="dall-e-3",
  prompt=user_info+doc+end_of_prompt,#user_info+doc+end_of_prompt,#user_info+inputPrompt+end_of_prompt,
  size="1024x1024",
  quality="standard",
  n=1,
)



print(response.data[0].url)
webbrowser.open(response.data[0].url)
'''
webdriver.
browser = webdriver.Chrome('chromedriver', chrome_options=options)
url = 'https://m.post.naver.com/viewer/postView.nhn?volumeNo=31538876&memberNo=6408050'
browser.maximize_window()
browser.get(url)
bs = BeautifulSoup(browser.page_source, 'lxml')
'''


