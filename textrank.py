# from keybert import KeyBERT
import openai 
import webbrowser
import os
from konlpy.tag import Okt
import re
from krwordrank.word import KRWordRank
from dotenv import load_dotenv


load_dotenv()
Dalle_Key = os.environ.get('MySecret')

def read_file_without_newlines(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        content = content.replace('\n', '')
    return content

# 파일 경로를 지정하세요.
file_path = './result.txt'

# 파일 내용을 문자열로 읽어서 저장합니다.
file_content = read_file_without_newlines(file_path)

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


client = openai.OpenAI(api_key = Dalle_Key)

user_info = "korean,8 age,man"
end_of_prompt = "within simple no color cartoon,only drawing,I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use it AS-IS"
# simple sketch
# no color

min_count = 0   # 단어의 최소 출현 빈도수 (그래프 생성 시)
max_length = 10 # 단어의 최대 길이
wordrank_extractor = KRWordRank(min_count=min_count, max_length=max_length)
beta = 0.8    # PageRank의 decaying factor beta
max_iter = 20
texts = split_noun_sentences(file_content)
keywords, rank, graph = wordrank_extractor.extract(texts, beta, max_iter)
inputPrompt = ""
for word, r in sorted(keywords.items(), key=lambda x:x[1], reverse=True):
    inputPrompt = inputPrompt + word + ","
    print('%8s:\t%.4f' % (word, r))

response = client.images.generate(
  model="dall-e-3",
  prompt=user_info+inputPrompt+end_of_prompt,#user_info+doc+end_of_prompt,#user_info+inputPrompt+end_of_prompt,
  size="1024x1024",
  quality="standard",
  n=1,
)
print(response.data[0].url)



