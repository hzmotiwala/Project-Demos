import os
import json
import re
from typing import List, Dict, Any
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter

class FeedbackAnalyzer:
    def __init__(self):
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "your-openai-api-key")
        )
        self.model = "gpt-4o"
    
    def analyze_sentiment_batch(self, feedback_items: List[str]) -> Dict[str, Any]:
        """Analyze sentiment for a batch of feedback items."""
        try:
            # Prepare batch prompt
            batch_text = "\n".join([f"{i+1}. {text}" for i, text in enumerate(feedback_items)])
            
            prompt = f"""
            Analyze the sentiment of each feedback item below. For each item, provide:
            1. A sentiment classification (positive, neutral, or negative)
            2. A sentiment score between -1 (very negative) and 1 (very positive)
            3. Key emotional indicators found in the text
            
            Feedback items:
            {batch_text}
            
            Respond with a JSON object in this exact format:
            {{
                "results": [
                    {{
                        "item_number": 1,
                        "sentiment": "positive/neutral/negative",
                        "sentiment_score": 0.7,
                        "emotional_indicators": ["word1", "word2"]
                    }}
                ]
            }}
            """
            
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert sentiment analyst."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            # Fallback to simple sentiment analysis
            results = []
            for i, text in enumerate(feedback_items):
                score = self._simple_sentiment_score(text)
                sentiment = "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"
                results.append({
                    "item_number": i + 1,
                    "sentiment": sentiment,
                    "sentiment_score": score,
                    "emotional_indicators": []
                })
            return {"results": results}
    
    def extract_themes_clustering(self, feedback_items: List[str], num_themes: int = 5) -> List[Dict[str, Any]]:
        """Extract themes using TF-IDF and K-means clustering."""
        try:
            # Vectorize the text
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            
            tfidf_matrix = vectorizer.fit_transform(feedback_items)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=min(num_themes, len(feedback_items)), random_state=42)
            clusters = kmeans.fit_predict(tfidf_matrix)
            
            # Extract themes
            feature_names = vectorizer.get_feature_names_out()
            themes = []
            
            for i in range(kmeans.n_clusters):
                # Get top features for this cluster
                cluster_center = kmeans.cluster_centers_[i]
                top_indices = cluster_center.argsort()[-10:][::-1]
                top_keywords = [feature_names[idx] for idx in top_indices]
                
                # Count items in this cluster
                cluster_count = np.sum(clusters == i)
                
                # Generate theme name using AI
                theme_name = self._generate_theme_name(top_keywords)
                
                themes.append({
                    "theme": theme_name,
                    "keywords": top_keywords[:5],
                    "frequency": int(cluster_count),
                    "cluster_id": i
                })
            
            # Sort by frequency
            themes.sort(key=lambda x: x['frequency'], reverse=True)
            return themes
            
        except Exception as e:
            # Fallback to keyword-based theme extraction
            return self._fallback_theme_extraction(feedback_items, num_themes)
    
    def _generate_theme_name(self, keywords: List[str]) -> str:
        """Generate a descriptive theme name from keywords using AI."""
        try:
            prompt = f"""
            Based on these keywords from user feedback clustering: {', '.join(keywords)}
            
            Generate a concise, descriptive theme name (2-4 words) that captures the main topic.
            Examples: "User Interface Issues", "Payment Problems", "Customer Support", "App Performance"
            
            Respond with just the theme name, no explanation.
            """
            
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=20
            )
            
            theme_name = response.choices[0].message.content.strip()
            return theme_name if len(theme_name.split()) <= 4 else ' '.join(keywords[:2]).title()
            
        except:
            # Fallback to keyword-based naming
            return ' '.join(keywords[:2]).title()
    
    def _fallback_theme_extraction(self, feedback_items: List[str], num_themes: int) -> List[Dict[str, Any]]:
        """Fallback theme extraction using simple keyword frequency."""
        # Common feedback topics
        topic_keywords = {
            "User Interface": ["ui", "interface", "design", "layout", "button", "screen", "visual"],
            "Performance": ["slow", "fast", "speed", "lag", "performance", "loading", "crash"],
            "Features": ["feature", "functionality", "option", "tool", "capability"],
            "Support": ["help", "support", "customer", "service", "response", "assistance"],
            "Bugs": ["bug", "error", "issue", "problem", "broken", "fix", "glitch"],
            "Usability": ["easy", "difficult", "confusing", "intuitive", "user-friendly"],
            "Content": ["content", "information", "data", "text", "image"],
            "Payment": ["payment", "price", "cost", "billing", "subscription", "money"]
        }
        
        theme_counts = {}
        all_text = ' '.join(feedback_items).lower()
        
        for theme, keywords in topic_keywords.items():
            count = sum(all_text.count(keyword) for keyword in keywords)
            if count > 0:
                theme_counts[theme] = count
        
        # Sort and return top themes
        sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
        
        themes = []
        for i, (theme, count) in enumerate(sorted_themes[:num_themes]):
            themes.append({
                "theme": theme,
                "keywords": topic_keywords[theme][:5],
                "frequency": count,
                "cluster_id": i
            })
        
        return themes
    
    def generate_action_items(self, feedback_items: List[str], themes: List[Dict], sentiment_results: Dict) -> List[Dict[str, Any]]:
        """Generate actionable insights and recommendations."""
        try:
            # Prepare context for AI
            context = {
                "total_feedback": len(feedback_items),
                "top_themes": [theme["theme"] for theme in themes[:5]],
                "sentiment_distribution": self._calculate_sentiment_distribution(sentiment_results),
                "sample_feedback": feedback_items[:5]  # Sample for context
            }
            
            prompt = f"""
            Based on this user feedback analysis, generate actionable insights and recommendations:
            
            Context:
            - Total feedback items: {context['total_feedback']}
            - Top themes: {', '.join(context['top_themes'])}
            - Sentiment distribution: {context['sentiment_distribution']}
            
            Sample feedback:
            {chr(10).join([f"- {fb}" for fb in context['sample_feedback']])}
            
            Generate 3-5 specific, actionable recommendations. For each recommendation, provide:
            1. Category (e.g., "UI/UX", "Performance", "Customer Support")
            2. Priority (High/Medium/Low)
            3. Issue description
            4. Specific recommendation
            5. Expected impact
            
            Respond in JSON format:
            {{
                "action_items": [
                    {{
                        "category": "UI/UX",
                        "priority": "High",
                        "issue": "Users find navigation confusing",
                        "recommendation": "Redesign main navigation with user testing",
                        "expected_impact": "Improved user experience and reduced confusion",
                        "evidence_count": 15
                    }}
                ]
            }}
            """
            
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a product management expert who specializes in turning user feedback into actionable insights."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            return result["action_items"]
            
        except Exception as e:
            # Fallback to rule-based action items
            return self._generate_fallback_actions(themes, sentiment_results)
    
    def _generate_fallback_actions(self, themes: List[Dict], sentiment_results: Dict) -> List[Dict[str, Any]]:
        """Generate fallback action items based on themes and sentiment."""
        actions = []
        
        # Rule-based action generation
        theme_actions = {
            "user interface": {
                "category": "UI/UX",
                "priority": "High",
                "issue": "Users experiencing interface difficulties",
                "recommendation": "Conduct UI/UX audit and implement improvements"
            },
            "performance": {
                "category": "Technical",
                "priority": "High",
                "issue": "Performance issues reported",
                "recommendation": "Optimize application performance and loading times"
            },
            "support": {
                "category": "Customer Support",
                "priority": "Medium",
                "issue": "Support-related concerns",
                "recommendation": "Improve support response times and documentation"
            }
        }
        
        for theme in themes[:3]:
            theme_key = theme["theme"].lower()
            for key, action in theme_actions.items():
                if key in theme_key:
                    action_item = action.copy()
                    action_item["evidence_count"] = theme["frequency"]
                    action_item["expected_impact"] = f"Address {theme['frequency']} reported issues"
                    actions.append(action_item)
                    break
        
        return actions
    
    def _calculate_sentiment_distribution(self, sentiment_results: Dict) -> Dict[str, int]:
        """Calculate sentiment distribution from results."""
        distribution = {"positive": 0, "neutral": 0, "negative": 0}
        
        for result in sentiment_results.get("results", []):
            sentiment = result.get("sentiment", "neutral")
            distribution[sentiment] += 1
        
        return distribution
    
    def _simple_sentiment_score(self, text: str) -> float:
        """Simple rule-based sentiment scoring as fallback."""
        positive_words = ["good", "great", "excellent", "amazing", "love", "perfect", "awesome", "fantastic"]
        negative_words = ["bad", "terrible", "awful", "hate", "horrible", "worst", "disappointing", "frustrating"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def analyze_feedback_batch(self, feedback_items: List[str], num_themes: int = 5) -> Dict[str, Any]:
        """Main method to analyze a batch of feedback items."""
        # Step 1: Sentiment Analysis
        sentiment_results = self.analyze_sentiment_batch(feedback_items)
        
        # Step 2: Theme Extraction
        themes = self.extract_themes_clustering(feedback_items, num_themes)
        
        # Step 3: Generate Action Items
        action_items = self.generate_action_items(feedback_items, themes, sentiment_results)
        
        # Step 4: Compile detailed analysis
        detailed_analysis = []
        for i, feedback in enumerate(feedback_items):
            sentiment_result = next(
                (r for r in sentiment_results["results"] if r["item_number"] == i + 1),
                {"sentiment": "neutral", "sentiment_score": 0.0, "emotional_indicators": []}
            )
            
            # Find relevant themes for this feedback item
            item_themes = self._find_item_themes(feedback, themes)
            
            detailed_analysis.append({
                "text": feedback,
                "sentiment": sentiment_result["sentiment"],
                "sentiment_score": sentiment_result["sentiment_score"],
                "themes": item_themes,
                "key_phrases": sentiment_result.get("emotional_indicators", [])
            })
        
        # Step 5: Calculate summary statistics
        sentiment_distribution = self._calculate_sentiment_distribution(sentiment_results)
        average_sentiment = np.mean([r["sentiment_score"] for r in sentiment_results["results"]])
        
        return {
            "total_feedback": len(feedback_items),
            "themes": themes,
            "sentiment_summary": {
                "distribution": sentiment_distribution,
                "average_sentiment": float(average_sentiment)
            },
            "action_items": action_items,
            "detailed_analysis": detailed_analysis
        }
    
    def _find_item_themes(self, feedback_text: str, themes: List[Dict]) -> List[str]:
        """Find which themes are relevant to a specific feedback item."""
        feedback_lower = feedback_text.lower()
        relevant_themes = []
        
        for theme in themes:
            # Check if any keywords from the theme appear in the feedback
            keyword_matches = sum(1 for keyword in theme["keywords"] if keyword.lower() in feedback_lower)
            if keyword_matches > 0:
                relevant_themes.append(theme["theme"])
        
        return relevant_themes[:3]  # Return top 3 most relevant themes
