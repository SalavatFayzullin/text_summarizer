import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import re
import requests
import json
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

# Загрузка переменных окружения из файла .env
load_dotenv()

class Summarizer:
    def __init__(self):
        # Загрузка необходимых данных NLTK для запасного варианта
        try:
            nltk.download('punkt', quiet=True)
        except:
            print("Error downloading NLTK data, but continuing anyway")
        
        # Проверка, установлен ли HF_API_TOKEN в переменных окружения
        self.hf_token = os.getenv('HF_API_TOKEN')
        
        if self.hf_token and self.hf_token != "your_huggingface_token_here":
            # Инициализация клиента API Inference с токеном
            self.client = InferenceClient(token=self.hf_token)
            print(f"Using Hugging Face Inference API with authentication: {self.hf_token[:5]}...")
        else:
            # Анонимный клиент - с ограничением скорости, но работает для демонстрации
            self.client = InferenceClient()
            print("Using Hugging Face Inference API anonymously (rate limited)")
            
        # Модель по умолчанию для суммаризации
        self.model = "facebook/bart-large-cnn"
        
        print(f"LLM summarization initialized using model: {self.model}")
        
        # Загрузка необходимых данных NLTK
        try:
            nltk.download('stopwords', quiet=True)
        except:
            print("Error downloading NLTK stopwords, but continuing anyway")
        
        print("Using extractive summarization with NLTK and TF-IDF")
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            # Упрощенный набор стоп-слов, если загрузка NLTK не удалась
            self.stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
                              'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 
                              'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
                              "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 
                              'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
                              'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 
                              'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
                              'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 
                              'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
                              'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
                              'through', 'during', 'before', 'after', 'above', 'below', 'to', 
                              'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 
                              'again', 'further', 'then', 'once', 'here', 'there', 'when', 
                              'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
                              'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 
                              'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 
                              'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 
                              'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', 
                              "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 
                              'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 
                              'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 
                              'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 
                              'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}
        
    def preprocess_text(self, text):
        """Очистка и предварительная обработка текста"""
        # Преобразование в нижний регистр
        text = text.lower()
        # Удаление специальных символов и цифр
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def sentence_similarity(self, sentences):
        """Вычисление схожести предложений с использованием TF-IDF"""
        # Создание векторизатора TF-IDF
        vectorizer = TfidfVectorizer(stop_words=list(self.stop_words))
        
        try:
            # Генерация матрицы TF-IDF
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Вычисление косинусного сходства
            similarity_matrix = tfidf_matrix @ tfidf_matrix.T
            
            return similarity_matrix.toarray()
        except:
            # Запасной вариант для очень коротких текстов или при ошибках
            return np.ones((len(sentences), len(sentences)))
    
    def rank_sentences(self, sentences, similarity_matrix):
        """Ранжирование предложений с использованием алгоритма, подобного TextRank"""
        # Инициализация оценок
        scores = np.ones(len(sentences))
        
        # Алгоритм, подобный TextRank (упрощенный)
        damping = 0.85
        iterations = 10
        
        for _ in range(iterations):
            new_scores = np.ones(len(sentences)) * (1 - damping)
            
            for i in range(len(sentences)):
                for j in range(len(sentences)):
                    if i != j:
                        new_scores[i] += damping * (similarity_matrix[i, j] * scores[j] / 
                                                   sum(similarity_matrix[j, :] + 1e-10))
            
            scores = new_scores
            
        return scores
    
    def llm_summarize(self, text, max_length=150, min_length=40):
        """Генерация резюме с использованием Inference API от Hugging Face"""
        try:
            # Использование InferenceClient для суммаризации
            summary = self.client.summarization(
                text, 
                model=self.model,
                parameters={
                    "max_length": max_length,
                    "min_length": min_length,
                    "do_sample": False
                }
            )
            return summary
        except Exception as e:
            print(f"Error in LLM summarization: {e}")
            return None
    
    def extractive_fallback(self, text, max_sentences=3):
        """Извлечение ключевых предложений в качестве запасного варианта, когда API не работает"""
        try:
            # Простая экстрактивная суммаризация - первые несколько предложений
            sentences = sent_tokenize(text)
            return ' '.join(sentences[:min(max_sentences, len(sentences))])
        except Exception as e:
            print(f"Error in extractive fallback: {e}")
            return "Error generating summary."
    
    def summarize(self, text, max_length=150, min_length=40):
        """
        Создание абстрактивного резюме входного текста с использованием LLM
        
        Аргументы:
            text (str): Текст для суммаризации
            max_length (int): Максимальная длина резюме (в токенах)
            min_length (int): Минимальная длина резюме (в токенах)
            
        Возвращает:
            str: Сгенерированное резюме
        """
        # Обработка пустого текста
        if not text or len(text.strip()) == 0:
            return "No text provided for summarization."
        
        # Обработка очень коротких текстов
        if len(text.split()) < 20:
            return text
            
        try:
            # Сначала попробовать суммаризацию с помощью LLM
            llm_summary = self.llm_summarize(text, max_length, min_length)
            
            if llm_summary:
                return llm_summary
            
            # Если LLM не сработал, используем запасной экстрактивный метод
            print("LLM summarization failed, using extractive fallback")
            return self.extractive_fallback(text)
            
        except Exception as e:
            print(f"Error in summarization: {e}")
            # Запасной вариант - простой экстрактивный метод
            return self.extractive_fallback(text) 