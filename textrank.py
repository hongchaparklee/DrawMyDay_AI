# from keybert import KeyBERT
import openai 
import webbrowser
import os
from konlpy.tag import Okt
import re
from krwordrank.word import KRWordRank


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

doc = """
오늘은 비가 많이 내렸다. 
학교가 끝나고 엄마가 우산을 쓰고 나를 데리러왔다. 
기분이 우울했는데 엄마가 집에서 맛있는 카레를 해줘서 기분이 너무 좋았다.
"""
#doc = "엄마가 집에서 맛있는 카레를 해줘서 기분이 너무 좋았다."

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
print("g")

# inputPrompt = "기분, 우울, 학교, 비, 우산"


client = openai.OpenAI(api_key = "제가 준 비밀키를 복붙하세요!!")

user_info = "9 years old asian boy"
end_of_prompt = "draw within cartoon"

response = client.images.generate(
  model="dall-e-3",
  prompt=user_info+inputPrompt+end_of_prompt,
  size="1024x1024",
  quality="standard",
  n=1,
)

webbrowser.open(response.data[0].url)

response = client.images.generate(
  model="dall-e-3",
  prompt=user_info+doc+end_of_prompt,
  size="1024x1024",
  quality="standard",
  n=1,
)

webbrowser.open(response.data[0].url)
