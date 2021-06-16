import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Komoran
import re


class RetrievalEncoder():
  def __init__(self):
    self.data = pd.read_csv('./data/200924_korean_ver1.csv')
    self.embedder = SentenceTransformer('distiluse-base-multilingual-cased')
    self.corpus = list(map(str, self.data['question']))
    self.corpus = np.unique(self.corpus)
    self.komoran = Komoran()
    cleaned_corpus= self.clean_text(self.corpus)
    self.dense_embedding = np.array(self.embedder.encode(cleaned_corpus))

    tokenized = []
    for i in range(len(self.corpus)):
      tokenized.append(' '.join(self.komoran.morphs(cleaned_corpus[i])))

# Sparse embedding
    self.tfidfv = TfidfVectorizer(tokenizer=self.identity_tokenizer, ngram_range=(1, 2)).fit(tokenized)
    self.sparse_embedding = self.tfidfv.transform(tokenized).toarray()


  def identity_tokenizer(self, text):
        return text


  def clean_text(self, texts):
      '''
      :param texts: sentences(question)
      :return: cleaned sentences
      '''
      temp = []
      for i in range(0, len(texts)):
          review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"]', '',str(texts[i])) #remove punctuation
          review = re.sub(r'\d+','', str(texts[i]))# remove number
          review = review.lower() #lower case
          review = re.sub(r'\s+', ' ', review) #remove extra space
          review = re.sub(r'<[^>]+>','',review) #remove Html tags
          review = re.sub(r'\s+', ' ', review) #remove spaces
          review = re.sub(r"^\s+", '', review) #remove space from start
          review = re.sub(r'\s+$', '', review) #remove space from the end
          temp.append(review)
      return temp


  def dense_searching(self, query, closest_n=10):
      '''
      :param query: input query
      :param closest_n: top K
      :return: [(idx, distance)] * closest_n
      '''

      q_desne_embedding = self.embedder.encode([query])
      distances = spatial.distance.cdist([q_desne_embedding[0]], self.dense_embedding, "cosine")[0]

      results = zip(range(len(distances)), distances)
      results = sorted(results, key=lambda x: x[1])
      results = results[0: closest_n]

      return results
    

  def sparse_searching(self, query, dense_candidate, closest_n=5):
      '''
      :param query: input query
      :param dense_candidate: dense_searching's output
      :param closest_n:
      :return: None
      '''

      ranked_sparse_embedding = []
      ranked_sparse_corpus = []
      # dense searching의 출력 값인 Top K개의 idx에 해당하는 sparse vector와
      # input query의 sparse vector와의 유사도 계산
      for idx, _ in dense_candidate:
          ranked_sparse_embedding.append(self.sparse_embedding[idx])
          ranked_sparse_corpus.append(self.corpus[idx])

      q_tokenized = self.komoran.morphs(query)
      q_sparse_embedding = self.tfidfv.transform([' '.join(q_tokenized)]).toarray()

      distances = spatial.distance.cdist(q_sparse_embedding, ranked_sparse_embedding, "cosine")[0]
      results = zip(range(len(distances)), distances)
      results = sorted(results, key=lambda x: x[1])
      best_idx = results[0:closest_n][0][0]

      return ranked_sparse_corpus[best_idx]


  def find_answer(self, query):
      result = self.dense_searching(query, 10)
      best_q = self.sparse_searching(query, result, 5)
      best_answer = self.data.loc[self.data.loc[:, 'question'] == best_q, 'answer']
      index = best_answer.index
      # print(index)
      return self.data.loc[int(index[0]), 'answer']