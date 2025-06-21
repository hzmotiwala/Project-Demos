import re
import string
from typing import List, Dict, Any
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import os

class TextProcessor:
    def __init__(self):
        self._setup_nltk()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Add custom stopwords for feedback analysis
        self.stop_words.update([
            'app', 'application', 'software', 'product', 'service',
            'would', 'could', 'should', 'might', 'may', 'really', 'very',
            'much', 'many', 'one', 'two', 'first', 'last', 'also', 'even'
        ])
    
    def _setup_nltk(self):
        """Download required NLTK data if not present."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                nltk.download('punkt', quiet=True)
            except:
                pass
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            try:
                nltk.download('stopwords', quiet=True)
            except:
                pass
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            try:
                nltk.download('wordnet', quiet=True)
            except:
                pass
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize a single text string."""
        if not text or not text.strip():
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\-\']', ' ', text)
        
        # Fix multiple whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove extra punctuation
        text = re.sub(r'[\.]{2,}', '.', text)
        text = re.sub(r'[\!]{2,}', '!', text)
        text = re.sub(r'[\?]{2,}', '?', text)
        
        return text.strip()
    
    def split_feedback(self, text: str) -> List[str]:
        """Split raw text into individual feedback items."""
        if not text or not text.strip():
            return []
        
        # First, try to split by double newlines (paragraph breaks)
        paragraphs = text.split('\n\n')
        feedback_items = []
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If paragraph is very long, try to split by single newlines
            if len(paragraph) > 500:
                lines = paragraph.split('\n')
                for line in lines:
                    line = line.strip()
                    if len(line) > 20:  # Minimum meaningful feedback length
                        feedback_items.append(line)
            else:
                feedback_items.append(paragraph)
        
        # If no double newlines found, split by single newlines
        if len(feedback_items) <= 1 and '\n' in text:
            lines = text.split('\n')
            feedback_items = [line.strip() for line in lines if line.strip() and len(line.strip()) > 20]
        
        # If still only one item and it's very long, try to split by sentences
        if len(feedback_items) == 1 and len(feedback_items[0]) > 1000:
            try:
                sentences = sent_tokenize(feedback_items[0])
                # Group sentences into chunks of 2-3 sentences
                feedback_items = []
                current_chunk = []
                
                for sentence in sentences:
                    current_chunk.append(sentence)
                    if len(current_chunk) >= 2 and len(' '.join(current_chunk)) > 100:
                        feedback_items.append(' '.join(current_chunk))
                        current_chunk = []
                
                if current_chunk:
                    feedback_items.append(' '.join(current_chunk))
            except:
                # If sentence tokenization fails, keep original
                pass
        
        return feedback_items if feedback_items else [text]
    
    def clean_feedback_batch(self, feedback_items: List[str], min_length: int = 20) -> List[str]:
        """Clean a batch of feedback items and filter by minimum length."""
        cleaned_items = []
        
        for item in feedback_items:
            cleaned = self.clean_text(item)
            if len(cleaned) >= min_length:
                cleaned_items.append(cleaned)
        
        return cleaned_items
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract important keywords from text."""
        try:
            # Tokenize and clean
            tokens = word_tokenize(text.lower())
            
            # Remove stopwords and punctuation
            keywords = [
                self.lemmatizer.lemmatize(token) 
                for token in tokens 
                if token not in self.stop_words 
                and token not in string.punctuation
                and len(token) > 2
                and token.isalpha()
            ]
            
            # Count frequency and return most common
            from collections import Counter
            keyword_freq = Counter(keywords)
            
            return [keyword for keyword, freq in keyword_freq.most_common(max_keywords)]
            
        except Exception as e:
            # Fallback to simple word extraction
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            return list(set(words))[:max_keywords]
    
    def extract_phrases(self, text: str, phrase_length: int = 2) -> List[str]:
        """Extract meaningful phrases from text."""
        try:
            # Tokenize
            tokens = word_tokenize(text.lower())
            
            # Remove stopwords and punctuation
            clean_tokens = [
                token for token in tokens 
                if token not in self.stop_words 
                and token not in string.punctuation
                and len(token) > 2
                and token.isalpha()
            ]
            
            # Generate n-grams
            phrases = []
            for i in range(len(clean_tokens) - phrase_length + 1):
                phrase = ' '.join(clean_tokens[i:i + phrase_length])
                phrases.append(phrase)
            
            # Remove duplicates and return
            return list(set(phrases))[:10]
            
        except Exception as e:
            # Simple fallback
            words = text.lower().split()
            phrases = []
            for i in range(len(words) - 1):
                if len(words[i]) > 2 and len(words[i+1]) > 2:
                    phrases.append(f"{words[i]} {words[i+1]}")
            return list(set(phrases))[:10]
    
    def preprocess_for_clustering(self, feedback_items: List[str]) -> List[str]:
        """Preprocess feedback items specifically for clustering analysis."""
        processed_items = []
        
        for item in feedback_items:
            # Clean the text
            cleaned = self.clean_text(item)
            
            try:
                # Tokenize and lemmatize
                tokens = word_tokenize(cleaned.lower())
                processed_tokens = [
                    self.lemmatizer.lemmatize(token) 
                    for token in tokens 
                    if token not in self.stop_words 
                    and token not in string.punctuation
                    and len(token) > 2
                    and token.isalpha()
                ]
                
                processed_text = ' '.join(processed_tokens)
                
            except Exception as e:
                # Fallback processing
                words = re.findall(r'\b[a-zA-Z]{3,}\b', cleaned.lower())
                processed_text = ' '.join(words)
            
            if processed_text.strip():
                processed_items.append(processed_text)
        
        return processed_items
    
    def detect_language(self, text: str) -> str:
        """Simple language detection (primarily English)."""
        # Count English words vs non-English characters
        english_pattern = re.compile(r'[a-zA-Z\s\.,!?]+')
        english_chars = len(english_pattern.findall(text))
        total_chars = len(text)
        
        if total_chars == 0:
            return "unknown"
        
        english_ratio = english_chars / total_chars
        return "english" if english_ratio > 0.7 else "other"
    
    def get_text_statistics(self, text: str) -> Dict[str, Any]:
        """Get basic statistics about the text."""
        try:
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            
            return {
                "character_count": len(text),
                "word_count": len(words),
                "sentence_count": len(sentences),
                "avg_words_per_sentence": len(words) / len(sentences) if sentences else 0,
                "language": self.detect_language(text)
            }
        except:
            # Fallback statistics
            words = text.split()
            sentences = text.split('.')
            
            return {
                "character_count": len(text),
                "word_count": len(words),
                "sentence_count": len(sentences),
                "avg_words_per_sentence": len(words) / len(sentences) if sentences else 0,
                "language": self.detect_language(text)
            }
