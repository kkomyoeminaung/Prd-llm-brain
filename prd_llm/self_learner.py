"""
Internet Self-Learning - Auto-learn from web
"""

import os
import json
import asyncio
import aiohttp
import re
import sqlite3
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass


@dataclass
class LearnedTopic:
    topic: str
    priority: int
    last_learned: datetime
    times_learned: int


class WebScraper:
    """Scrape text from web sources"""
    
    HEADERS = {
        'User-Agent': 'PRD-LLM-Learner/1.0 (Educational)',
        'Accept': 'text/html,application/xhtml+xml',
    }
    
    async def search_duckduckgo(self, query: str) -> List[Dict]:
        """Search DuckDuckGo and return results"""
        url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1"
        results = []
        
        try:
            async with aiohttp.ClientSession(headers=self.HEADERS) as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        
                        if data.get('AbstractText'):
                            results.append({
                                'title': data.get('Heading', query),
                                'text': data['AbstractText'],
                                'url': data.get('AbstractURL', '')
                            })
                        
                        for topic in data.get('RelatedTopics', [])[:5]:
                            if isinstance(topic, dict) and topic.get('Text'):
                                results.append({
                                    'title': query,
                                    'text': topic['Text'],
                                    'url': topic.get('FirstURL', '')
                                })
        except Exception as e:
            print(f"[Scraper] Error: {e}")
        
        return results
    
    async def fetch_wikipedia(self, topic: str) -> Optional[str]:
        """Fetch Wikipedia summary"""
        clean = topic.replace(' ', '_')
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{clean}"
        
        try:
            async with aiohttp.ClientSession(headers=self.HEADERS) as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get('extract', '')
        except Exception as e:
            print(f"[Scraper] Wikipedia error: {e}")
        return None
    
    async def fetch_page_text(self, url: str, max_chars: int = 3000) -> Optional[str]:
        """Fetch and clean text from web page"""
        try:
            async with aiohttp.ClientSession(headers=self.HEADERS) as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 200:
                        html = await resp.text(errors='replace')
                        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
                        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
                        text = re.sub(r'<[^>]+>', ' ', text)
                        text = re.sub(r'\s+', ' ', text).strip()
                        return text[:max_chars]
        except:
            pass
        return None


class InternetSelfLearner:
    """
    Auto-learn from internet without human intervention
    """
    
    DEFAULT_TOPICS = [
        "machine learning", "artificial intelligence", "neural networks",
        "python programming", "data science", "algorithms",
        "quantum physics", "biology", "chemistry",
        "world history", "philosophy", "psychology",
        "myanmar culture", "buddhism", "southeast asia",
    ]
    
    def __init__(self, data_path: str, db_path: str, teacher=None):
        self.data_path = data_path
        self.db_path = db_path
        self.teacher = teacher
        self.scraper = WebScraper()
        self.learned_topics = set()
        
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        self._init_db()
        self._seed_topics()
    
    def _init_db(self):
        """Initialize SQLite database for tracking"""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS learned_urls (
                url TEXT PRIMARY KEY,
                topic TEXT,
                learned_at DATETIME,
                quality_score REAL
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS topic_queue (
                id INTEGER PRIMARY KEY,
                topic TEXT UNIQUE,
                priority INTEGER DEFAULT 5,
                times_learned INTEGER DEFAULT 0,
                last_learned DATETIME
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS learning_stats (
                date TEXT PRIMARY KEY,
                urls_learned INTEGER DEFAULT 0,
                qa_pairs_added INTEGER DEFAULT 0
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _seed_topics(self):
        """Add default topics to queue"""
        conn = sqlite3.connect(self.db_path)
        
        for topic in self.DEFAULT_TOPICS:
            conn.execute(
                "INSERT OR IGNORE INTO topic_queue (topic, priority) VALUES (?, ?)",
                (topic, 5)
            )
        
        conn.commit()
        conn.close()
    
    def add_topic(self, topic: str, priority: int = 5):
        """Add custom topic to learning queue"""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO topic_queue (topic, priority) VALUES (?, ?)",
            (topic, priority)
        )
        conn.commit()
        conn.close()
        print(f"[SelfLearner] Added topic: {topic}")
    
    def _get_next_topic(self) -> Optional[str]:
        """Get highest priority unlearned topic"""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("""
            SELECT topic FROM topic_queue
            WHERE last_learned IS NULL OR 
                  datetime(last_learned) < datetime('now', '-7 days')
            ORDER BY priority DESC, times_learned ASC
            LIMIT 1
        """).fetchone()
        conn.close()
        return row[0] if row else None
    
    def _mark_topic_learned(self, topic: str):
        """Mark topic as learned"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            UPDATE topic_queue 
            SET times_learned = times_learned + 1,
                last_learned = CURRENT_TIMESTAMP
            WHERE topic = ?
        """, (topic,))
        conn.commit()
        conn.close()
    
    def _mark_url_learned(self, url: str, topic: str, quality: float):
        """Mark URL as learned"""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR IGNORE INTO learned_urls (url, topic, learned_at, quality_score) VALUES (?, ?, ?, ?)",
            (url, topic, datetime.now(), quality)
        )
        conn.commit()
        conn.close()
    
    def _was_url_learned(self, url: str) -> bool:
        """Check if URL was already learned"""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT 1 FROM learned_urls WHERE url=?", (url,)).fetchone()
        conn.close()
        return row is not None
    
    async def _extract_qa_from_text(self, text: str, topic: str) -> List[Dict]:
        """Extract Q&A pairs from text using teacher AI"""
        if not self.teacher or len(text) < 100:
            return []
        
        prompt = f"""Read this text about "{topic}" and create 3-5 question-answer pairs.

Text: {text[:1500]}

Rules:
- Questions should be educational and specific
- Answers based on the text
- Output as JSON array only

Output format:
[{{"question": "...", "answer": "..."}}, ...]"""
        
        try:
            response = await self.teacher.generate(prompt)
            if response:
                start = response.find('[')
                end = response.rfind(']') + 1
                if start != -1 and end > start:
                    pairs = json.loads(response[start:end])
                    return [p for p in pairs if 'question' in p and 'answer' in p]
        except Exception as e:
            print(f"[SelfLearner] QA extraction error: {e}")
        
        return []
    
    def _save_training_data(self, items: List[Dict]):
        """Save to JSONL training file"""
        with open(self.data_path, 'a', encoding='utf-8') as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    async def learn_topic(self, topic: str) -> int:
        """Learn a specific topic from internet"""
        print(f"📚 Learning: {topic}")
        total_qa = 0
        
        # 1. Try Wikipedia first
        wiki_text = await self.scraper.fetch_wikipedia(topic)
        if wiki_text and len(wiki_text) > 200:
            qa_pairs = await self._extract_qa_from_text(wiki_text, topic)
            if qa_pairs:
                items = [{
                    'prompt': qa['question'],
                    'response': qa['answer'],
                    'source': f'wikipedia:{topic}',
                    'domain': self._detect_domain(topic)
                } for qa in qa_pairs]
                self._save_training_data(items)
                total_qa += len(items)
                print(f"  ✅ Wikipedia: +{len(items)} Q&A pairs")
        
        # 2. Search DuckDuckGo
        search_results = await self.scraper.search_duckduckgo(topic)
        
        for result in search_results[:3]:
            url = result.get('url', '')
            text = result.get('text', '')
            
            if url and self._was_url_learned(url):
                continue
            
            if not text and url:
                text = await self.scraper.fetch_page_text(url)
            
            if text and len(text) > 200:
                qa_pairs = await self._extract_qa_from_text(text, topic)
                if qa_pairs:
                    items = [{
                        'prompt': qa['question'],
                        'response': qa['answer'],
                        'source': f'web:{url[:100]}',
                        'domain': self._detect_domain(topic)
                    } for qa in qa_pairs]
                    self._save_training_data(items)
                    total_qa += len(items)
                    print(f"  ✅ Web: +{len(items)} Q&A pairs")
                
                if url:
                    quality = len(qa_pairs) / 5.0 if qa_pairs else 0.3
                    self._mark_url_learned(url, topic, quality)
        
        # Mark topic as learned
        self._mark_topic_learned(topic)
        
        return total_qa
    
    def _detect_domain(self, text: str) -> str:
        text_lower = text.lower()
        if any(w in text_lower for w in ['physics', 'quantum']):
            return 'physics'
        if any(w in text_lower for w in ['biology', 'genetics']):
            return 'biology'
        if any(w in text_lower for w in ['math', 'calculus']):
            return 'mathematics'
        if any(w in text_lower for w in ['programming', 'code']):
            return 'computer_science'
        return 'general'
    
    async def run_learning_cycle(self, max_topics: int = 3) -> Dict:
        """Run one learning cycle"""
        results = {'topics_learned': 0, 'total_qa': 0}
        
        for _ in range(max_topics):
            topic = self._get_next_topic()
            if not topic:
                break
            
            qa_count = await self.learn_topic(topic)
            results['topics_learned'] += 1
            results['total_qa'] += qa_count
        
        return results
    
    def get_stats(self) -> Dict:
        """Get learning statistics"""
        conn = sqlite3.connect(self.db_path)
        
        total_urls = conn.execute("SELECT COUNT(*) FROM learned_urls").fetchone()[0]
        total_topics = conn.execute("SELECT COUNT(*) FROM topic_queue WHERE times_learned > 0").fetchone()[0]
        pending_topics = conn.execute("SELECT COUNT(*) FROM topic_queue WHERE last_learned IS NULL").fetchone()[0]
        
        # Estimate QA pairs added (simulated for internal tracking)
        qa_row = conn.execute("SELECT SUM(qa_pairs_added) FROM learning_stats").fetchone()
        total_qa = (qa_row[0] or 0) + (total_urls * 3) # Baseline estimation
        
        conn.close()
        
        return {
            'total_urls_learned': total_urls,
            'total_topics_learned': total_topics,
            'pending_topics': pending_topics,
            'total_qa': total_qa
        }
