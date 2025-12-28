"""
Smarter Support/Resistance + Pattern detection Streamlit app — single-file fixed + ML classifier.
This version integrates the FINAL Mind-Free, High-Accuracy EMA Pullback Signal.
"""

from __future__ import annotations
import os
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import pickle
import io
import requests
import time
from bs4 import BeautifulSoup
from nsepy import get_history
from plotly.subplots import make_subplots
from newsapi import NewsApiClient
import re
import html

# sklearn imports guarded by try/except to show friendly message if missing
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
except Exception as e:
    _SKLEARN_IMPORT_ERROR = str(e)
    RandomForestClassifier = None
    train_test_split = None
    cross_val_score = None
    accuracy_score = None
    confusion_matrix = None
    classification_report = None
    roc_auc_score = None
    StandardScaler = None
    Pipeline = None
else:
    _SKLEARN_IMPORT_ERROR = None

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Accept-Language': 'en-US,en;q=0.9',
}

DATA_DIR = "stock_data_cache"
os.makedirs(DATA_DIR, exist_ok=True)
SUMMARY_FILE = os.path.join(DATA_DIR, "liquidity_sweep_summary.csv")
TIMESTAMP_FILE = os.path.join(DATA_DIR, "last_simulation_timestamp.txt")

import logging

logging.basicConfig(
    filename='liquidity_sweep_debug.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    filemode='w'
)

logging.info("Streamlit app has started")

def scrape_google_news(ticker: str, num_articles: int = 15) -> List[Dict[str, Any]]:
    """
    Scrapes recent news from Google News RSS feed.
    Fetches extra articles to account for duplicates that will be removed later.
    """
    try:
        import feedparser
        
        clean_ticker = ticker.replace('.NS', '').replace('.BO', '')
        
        url = f'https://news.google.com/rss/search?q={clean_ticker}+stock&hl=en-IN&gl=IN&ceid=IN:en'
        
        feed = feedparser.parse(url)
        articles = []
        
        for entry in feed.entries[:num_articles]:
            pub_date = datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') else datetime.now()
            
            title = entry.title
            source = 'Google News'
            
            if ' - ' in title:
                parts = title.rsplit(' - ', 1)
                if len(parts) == 2:
                    title = parts[0].strip()
                    source = parts[1].strip()
            
            description = None
            
            if hasattr(entry, 'summary') and entry.summary:
                soup = BeautifulSoup(entry.summary, 'html.parser')
                text_content = soup.get_text(separator=' ', strip=True)
                
                if text_content.startswith(entry.title):
                    text_content = text_content[len(entry.title):].strip()
                
                if source in text_content:
                    text_content = text_content.replace(source, '').strip()
                
                if text_content and len(text_content) > 20:
                    description = text_content
            
            if not description or len(description) < 20:
                description = f"Read the latest news about {clean_ticker} stock."
            
            description = ' '.join(description.split())
            if len(description) > 300:
                description = description[:297] + '...'
            
            articles.append({
                'title': clean_html_text(title),
                'description': description,
                'url': entry.link,
                'source': {'name': source},
                'publishedAt': pub_date.strftime('%Y-%m-%d %H:%M:%S'),
                'sentiment': None,
                'news_source': 'Google'  # Track which scraper found this
            })
        
        return articles
    
    except Exception as e:
        logging.error(f"Error scraping Google News: {e}")
        return []


def scrape_bing_news(ticker: str, num_articles: int = 15) -> List[Dict[str, Any]]:
    """
    Scrapes news from Bing News search (better descriptions than Google News RSS).
    Fetches extra articles to account for duplicates that will be removed later.
    """
    try:
        from urllib.parse import quote
        
        clean_ticker = ticker.replace('.NS', '').replace('.BO', '')
        search_query = quote(f'{clean_ticker} stock news')
        
        url = f'https://www.bing.com/news/search?q={search_query}&format=rss'
        
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.content, 'xml')
        
        articles = []
        items = soup.find_all('item')[:num_articles]
        
        for item in items:
            title = item.find('title').text if item.find('title') else 'No title'
            description = item.find('description').text if item.find('description') else ''
            link = item.find('link').text if item.find('link') else ''
            pub_date = item.find('pubDate').text if item.find('pubDate') else ''
            source = item.find('source').text if item.find('source') else 'Bing News'
            
            # Clean description
            description = clean_html_text(description)
            if not description or len(description) < 20:
                description = f"Latest update on {clean_ticker}. Click to read the full article."
            
            # Parse date
            try:
                if pub_date:
                    # Bing uses RFC 2822 format
                    dt = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %Z')
                    pub_date = dt.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    pub_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            except:
                try:
                    # Try alternative format
                    dt = datetime.strptime(pub_date.split(' GMT')[0], '%a, %d %b %Y %H:%M:%S')
                    pub_date = dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    pub_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            articles.append({
                'title': clean_html_text(title),
                'description': description,
                'url': link,
                'source': {'name': source},
                'publishedAt': pub_date,
                'sentiment': None,
                'news_source': 'Bing'  # Track which scraper found this
            })
        
        return articles
    
    except Exception as e:
        logging.error(f"Error scraping Bing News: {e}")
        return []


def calculate_article_relevance(article: Dict[str, Any], 
                                target_ticker: str,
                                company_name: Optional[str] = None) -> float:
    """
    Calculate how relevant an article is to the target ticker.
    NOW INCLUDES EXCHANGE AND COMPANY NAME VALIDATION!
    
    Returns a score from 0.0 (not relevant) to 1.0 (highly relevant).
    """
    score = 0.0
    base_ticker, target_exchange = extract_exchange_from_ticker(target_ticker)
    
    title = article.get('title', '').upper()
    description = article.get('description', '').upper()
    source = article.get('source', {})
    if isinstance(source, dict):
        source_name = source.get('name', '').upper()
    else:
        source_name = str(source).upper()
    
    full_text = f"{title} {description} {source_name}"
    
    # ==========================================
    # EXCHANGE VALIDATION (Critical!)
    # ==========================================
    
    # Check if article mentions wrong exchange
    wrong_exchange_indicators = {
        'NSE': ['ASX', 'AUSTRALIA', 'SYDNEY', 'LSE', 'LONDON', 'NYSE:', 'NASDAQ:'],
        'BSE': ['ASX', 'AUSTRALIA', 'SYDNEY', 'LSE', 'LONDON', 'NYSE:', 'NASDAQ:'],
        'US': ['NSE', 'BSE', 'MUMBAI', 'INDIA', 'ASX', 'AUSTRALIA', 'LSE', 'LONDON'],
        'ASX': ['NSE', 'BSE', 'MUMBAI', 'INDIA', 'NYSE:', 'NASDAQ:', 'LSE', 'LONDON'],
    }
    
    # If article mentions wrong exchange, return very low score
    if target_exchange in wrong_exchange_indicators:
        for wrong_indicator in wrong_exchange_indicators[target_exchange]:
            if wrong_indicator in full_text:
                logging.info(f"Article mentions wrong exchange '{wrong_indicator}' for {target_ticker}")
                return 0.05  # Very low relevance score
    
    # Boost score if article mentions correct exchange
    correct_exchange_indicators = {
        'NSE': ['NSE', 'NATIONAL STOCK EXCHANGE', 'MUMBAI', 'INDIA', 'INDIAN STOCK'],
        'BSE': ['BSE', 'BOMBAY STOCK EXCHANGE', 'MUMBAI', 'INDIA', 'INDIAN STOCK'],
        'US': ['NYSE', 'NASDAQ', 'S&P 500', 'DOW JONES', 'WALL STREET', 'US STOCK'],
        'ASX': ['ASX', 'AUSTRALIAN STOCK', 'SYDNEY', 'AUSTRALIA'],
    }
    
    if target_exchange in correct_exchange_indicators:
        for correct_indicator in correct_exchange_indicators[target_exchange]:
            if correct_indicator in full_text:
                score += 0.15  # Bonus for mentioning correct exchange
                break
    
    # ==========================================
    # COMPANY NAME MATCHING (More Reliable!)
    # ==========================================
    
    if company_name:
        # Check if company name appears in title or description
        if company_name in title:
            score += 0.4  # Strong signal
        elif company_name in description:
            score += 0.25  # Good signal
        
        # Check for partial company name matches (e.g., "ABB India" vs "ABB Australia")
        company_words = company_name.split()
        if len(company_words) > 1:
            # Multi-word company names are more specific
            matching_words = sum(1 for word in company_words if word in title)
            if matching_words >= len(company_words) * 0.6:  # 60% of words match
                score += 0.2
    
    # ==========================================
    # TICKER MATCHING (Less Reliable Alone)
    # ==========================================
    
    # Title relevance
    if base_ticker in title:
        # Check if ticker is in first half of title (more prominent)
        title_position = title.find(base_ticker) / max(len(title), 1)
        if title_position < 0.3:
            score += 0.3  # Prominent in title
        else:
            score += 0.2  # Mentioned in title
    
    # Description relevance
    if base_ticker in description:
        desc_position = description.find(base_ticker) / max(len(description), 1)
        if desc_position < 0.2:
            score += 0.2  # Early in description
        else:
            score += 0.1  # Mentioned in description
    
    # Count how many times the ticker appears
    ticker_count_title = title.count(base_ticker)
    ticker_count_desc = description.count(base_ticker)
    
    if ticker_count_title >= 2:
        score += 0.1
    if ticker_count_desc >= 2:
        score += 0.05
    
    # ==========================================
    # MULTI-COMPANY PENALTY
    # ==========================================
    
    # Penalty for articles that mention many other company names
    common_tickers = ['APPLE', 'GOOGLE', 'AMAZON', 'MICROSOFT', 'META', 
                     'TESLA', 'NVIDIA', 'ALPHABET', 'RELIANCE', 'TCS',
                     'INFOSYS', 'WIPRO', 'HDFC', 'ICICI']
    
    other_ticker_count = sum(1 for t in common_tickers 
                            if t != base_ticker and t in full_text)
    
    if other_ticker_count >= 4:
        score *= 0.3  # Severe penalty for market roundup articles
    elif other_ticker_count >= 3:
        score *= 0.5  # Strong penalty
    elif other_ticker_count >= 2:
        score *= 0.7  # Moderate penalty
    
    return min(score, 1.0)


def filter_relevant_articles(articles: List[Dict[str, Any]], 
                            target_ticker: str, 
                            min_relevance: float = 0.3) -> List[Dict[str, Any]]:
    """
    Filter articles based on relevance to the target ticker.
    NOW WITH COMPANY NAME LOOKUP!
    """
    # Get company name once for all articles
    company_name = get_company_name_from_ticker(target_ticker)
    base_ticker, exchange = extract_exchange_from_ticker(target_ticker)
    
    if company_name:
        logging.info(f"Filtering news for {company_name} ({base_ticker}, {exchange})")
    else:
        logging.warning(f"Could not get company name for {target_ticker}, using ticker only")
    
    scored_articles = []
    
    for article in articles:
        relevance = calculate_article_relevance(article, target_ticker, company_name)
        if relevance >= min_relevance:
            article['relevance_score'] = relevance
            scored_articles.append(article)
        else:
            # Log why article was filtered out for debugging
            logging.debug(f"Filtered out article (relevance {relevance:.2f}): {article.get('title', 'No title')[:50]}")
    
    # Sort by relevance score (highest first)
    scored_articles.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
    
    return scored_articles


def fetch_stock_news(ticker: str, api_key: Optional[str] = None, num_articles: int = 10) -> List[Dict[str, Any]]:
    """
    Fetches recent news about a stock using multiple sources and combines them.
    NOW WITH EXCHANGE VALIDATION AND COMPANY NAME MATCHING!
    """
    base_ticker, exchange = extract_exchange_from_ticker(ticker)
    
    # Get company name for better matching
    company_name = get_company_name_from_ticker(ticker)
    
    # Use company name in search if available, otherwise use ticker
    search_term = company_name if company_name else base_ticker
    
    all_articles = []
    
    # Try NewsAPI first if key is provided
    if api_key:
        try:
            newsapi = NewsApiClient(api_key=api_key)
            to_date = datetime.now()
            from_date = to_date - timedelta(days=7)
            
            # Add exchange context to search
            if exchange in ['NSE', 'BSE']:
                search_query = f'{search_term} India stock'
            elif exchange == 'ASX':
                search_query = f'{search_term} Australia stock'
            else:
                search_query = f'{search_term} stock'
            
            response = newsapi.get_everything(
                q=search_query,
                from_param=from_date.strftime('%Y-%m-%d'),
                to=to_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='publishedAt',
                page_size=num_articles * 2
            )
            
            if response['status'] == 'ok' and response.get('articles'):
                all_articles.extend(response['articles'][:num_articles * 2])
        
        except Exception as e:
            logging.warning(f"NewsAPI fetch failed: {e}. Falling back to free sources.")
    
    # Fetch from Bing News
    try:
        # Add exchange context
        if exchange in ['NSE', 'BSE']:
            search_term_bing = f'{base_ticker} India'
        elif exchange == 'ASX':
            search_term_bing = f'{base_ticker} Australia'
        else:
            search_term_bing = base_ticker
            
        bing_articles = scrape_bing_news(search_term_bing, num_articles * 2)
        if bing_articles:
            all_articles.extend(bing_articles)
            logging.info(f"Fetched {len(bing_articles)} articles from Bing News")
    except Exception as e:
        logging.warning(f"Bing News scraping failed: {e}")
    
    # Fetch from Google News
    try:
        # Add exchange context
        if exchange in ['NSE', 'BSE']:
            search_term_google = f'{base_ticker} India'
        elif exchange == 'ASX':
            search_term_google = f'{base_ticker} Australia'
        else:
            search_term_google = base_ticker
            
        google_articles = scrape_google_news(search_term_google, num_articles * 2)
        if google_articles:
            all_articles.extend(google_articles)
            logging.info(f"Fetched {len(google_articles)} articles from Google News")
    except Exception as e:
        logging.warning(f"Google News scraping failed: {e}")
    
    # Remove duplicates
    unique_articles = remove_duplicate_articles(all_articles)
    
    # **Filter for relevance WITH exchange validation**
    relevant_articles = filter_relevant_articles(unique_articles, ticker, min_relevance=0.3)
    
    logging.info(f"Filtered to {len(relevant_articles)} relevant articles from {len(unique_articles)} unique articles for {ticker}")
    
    # Sort by date (newest first)
    relevant_articles.sort(key=lambda x: x.get('publishedAt', ''), reverse=True)
    
    return relevant_articles[:num_articles]


def display_news_section(ticker: str, news_api_key: Optional[str] = None):
    """
    Displays a formatted news section in Streamlit.
    NOW WITH EXCHANGE INFO!
    """
    base_ticker, exchange = extract_exchange_from_ticker(ticker)
    company_name = get_company_name_from_ticker(ticker)
    
    if company_name:
        st.subheader(f'📰 Recent News for {company_name} ({base_ticker}, {exchange})')
    else:
        st.subheader(f'📰 Recent News for {base_ticker} ({exchange})')
    
    with st.spinner('Fetching latest news from multiple sources...'):
        articles = fetch_stock_news(ticker, api_key=news_api_key, num_articles=10)
    
    if not articles:
        st.warning(f'No relevant news found for {base_ticker} on {exchange} exchange.')
        return
    
    # Show source breakdown
    bing_count = sum(1 for a in articles if a.get('news_source') == 'Bing')
    google_count = sum(1 for a in articles if a.get('news_source') == 'Google')
    newsapi_count = sum(1 for a in articles if a.get('news_source') not in ['Bing', 'Google'])
    
    st.caption(f"📊 Sources: Bing News ({bing_count}) • Google News ({google_count})" + 
               (f" • NewsAPI ({newsapi_count})" if newsapi_count > 0 else ""))
    
    # Show average relevance
    avg_relevance = sum(a.get('relevance_score', 0) for a in articles) / len(articles) if articles else 0
    st.caption(f"🎯 Average Relevance Score: {avg_relevance:.2f} | 🌍 Exchange: {exchange}")
    
    # Analyze sentiment for each article
    for article in articles:
        title = clean_html_text(article.get('title', ''))
        description = clean_html_text(article.get('description', ''))
        
        article['title'] = title
        article['description'] = description
        
        combined_text = f"{title} {description}"
        sentiment = analyze_news_sentiment(combined_text)
        article['sentiment'] = sentiment
    
    # Calculate overall sentiment
    avg_sentiment = sum(a['sentiment']['score'] for a in articles) / len(articles)
    
    # Display overall sentiment metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('Total Articles', len(articles))
    with col2:
        sentiment_label = '🟢 Positive' if avg_sentiment > 0.2 else '🔴 Negative' if avg_sentiment < -0.2 else '🟡 Neutral'
        st.metric('Overall Sentiment', sentiment_label)
    with col3:
        positive_count = sum(1 for a in articles if a['sentiment']['score'] > 0.3)
        st.metric('Positive News', f"{positive_count}/{len(articles)}")
    with col4:
        st.metric('Avg Relevance', f"{avg_relevance:.2f}")
    
    st.markdown('---')
    
    # Display articles
    for i, article in enumerate(articles, 1):
        title = article.get('title', 'No title')
        relevance = article.get('relevance_score', 0)
        
        # Add source badge to title
        news_source_badge = ""
        if article.get('news_source') == 'Bing':
            news_source_badge = " 🅱️"
        elif article.get('news_source') == 'Google':
            news_source_badge = " 🔍"
        
        # Add relevance indicator to title
        relevance_indicator = "🎯🎯🎯" if relevance > 0.7 else "🎯🎯" if relevance > 0.5 else "🎯"
        
        with st.expander(f"📄 {relevance_indicator} {title}{news_source_badge}", expanded=(i <= 3)):
            
            # Show relevance score
            st.markdown(f"**Relevance:** {relevance:.2f} | **Sentiment:** {article['sentiment']['label']} (Score: {article['sentiment']['score']:.2f})")
            
            # Article details
            col1, col2 = st.columns([3, 1])
            
            with col1:
                source = article.get('source', 'Unknown')
                if isinstance(source, dict):
                    source = source.get('name', 'Unknown')
                st.markdown(f"**Source:** {source}")
                
                if article.get('publishedAt'):
                    pub_date = article['publishedAt']
                    if isinstance(pub_date, str):
                        try:
                            if 'T' in pub_date:
                                pub_date = datetime.strptime(pub_date.split('.')[0], '%Y-%m-%d %H:%M:%S').strftime('%B %d, %Y at %I:%M %p')
                            else:
                                pub_date = datetime.strptime(pub_date, '%Y-%m-%d %H:%M:%S').strftime('%B %d, %Y at %I:%M %p')
                        except:
                            pub_date = pub_date.split('T')[0] if 'T' in pub_date else pub_date
                    st.markdown(f"**Published:** {pub_date}")
            
            with col2:
                if article.get('url'):
                    st.markdown(f"[🔗 Read More]({article['url']})")
            
            # Description
            description = article.get('description', 'No description available.')
            if description and description != 'No description available.' and len(description) > 10:
                if len(description) > 500:
                    description = description[:497] + '...'
                st.markdown(f"_{description}_")
            
            st.markdown('---')


def remove_duplicate_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Removes duplicate articles based on title similarity.
    Uses fuzzy matching to catch near-duplicates.
    """
    if not articles:
        return []
    
    unique = []
    seen_titles = []
    
    for article in articles:
        title = article.get('title', '').lower().strip()
        
        # Skip if title is too short or empty
        if len(title) < 10:
            continue
        
        # Check for exact or near-duplicate
        is_duplicate = False
        for seen_title in seen_titles:
            # Simple similarity check: if 80% of words match, consider duplicate
            title_words = set(title.split())
            seen_words = set(seen_title.split())
            
            if not title_words or not seen_words:
                continue
            
            overlap = len(title_words & seen_words)
            similarity = overlap / max(len(title_words), len(seen_words))
            
            if similarity > 0.8:  # 80% similarity threshold
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique.append(article)
            seen_titles.append(title)
    
    return unique


def analyze_news_sentiment(text: str) -> Dict[str, Any]:
    """
    Simple sentiment analysis using keyword matching.
    For production, consider using TextBlob or a proper NLP model.
    """
    if not text:
        return {'score': 0, 'label': 'Neutral'}
    
    text_lower = text.lower()
    
    # Positive keywords
    positive_words = ['gain', 'profit', 'surge', 'rally', 'bullish', 'growth', 
                     'upgrade', 'beat', 'strong', 'positive', 'outperform',
                     'expansion', 'success', 'breakthrough', 'record']
    
    # Negative keywords
    negative_words = ['loss', 'drop', 'fall', 'decline', 'bearish', 'downgrade',
                     'miss', 'weak', 'negative', 'underperform', 'layoff',
                     'lawsuit', 'investigation', 'fraud', 'crash']
    
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    # Calculate sentiment score
    total = pos_count + neg_count
    if total == 0:
        return {'score': 0, 'label': 'Neutral', 'confidence': 0.5}
    
    score = (pos_count - neg_count) / total
    
    if score > 0.3:
        label = '🟢 Positive'
        confidence = min(0.5 + (score * 0.5), 1.0)
    elif score < -0.3:
        label = '🔴 Negative'
        confidence = min(0.5 + (abs(score) * 0.5), 1.0)
    else:
        label = '🟡 Neutral'
        confidence = 0.5
    
    return {'score': score, 'label': label, 'confidence': confidence}

def get_company_name_from_ticker(ticker: str) -> Optional[str]:
    """
    Fetch the actual company name for a ticker to improve matching accuracy.
    Caches results to avoid repeated API calls.
    """
    try:
        # Try to get company info from yfinance
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get long name or short name
        company_name = info.get('longName') or info.get('shortName')
        
        if company_name:
            return company_name.upper()
        
    except Exception as e:
        logging.warning(f"Could not fetch company name for {ticker}: {e}")
    
    return None


def extract_exchange_from_ticker(ticker: str) -> Tuple[str, str]:
    """
    Extract the base ticker and exchange suffix.
    
    Returns:
    --------
    Tuple of (base_ticker, exchange)
    
    Examples:
    ---------
    'RELIANCE.NS' -> ('RELIANCE', 'NSE')
    'ABB.NS' -> ('ABB', 'NSE')
    'ABB.AX' -> ('ABB', 'ASX')
    'AAPL' -> ('AAPL', 'US')
    """
    ticker = ticker.upper().strip()
    
    # Map of exchange suffixes to exchange names
    exchange_map = {
        '.NS': 'NSE',
        '.BO': 'BSE',
        '.AX': 'ASX',  # Australia
        '.L': 'LSE',   # London
        '.TO': 'TSX',  # Toronto
        '.HK': 'HKEX', # Hong Kong
        '.T': 'TSE',   # Tokyo
        '.SI': 'SGX',  # Singapore
    }
    
    for suffix, exchange in exchange_map.items():
        if ticker.endswith(suffix):
            base_ticker = ticker.replace(suffix, '')
            return base_ticker, exchange
    
    # Default to US exchanges if no suffix
    return ticker, 'US'

# Add this helper function after the imports section
def clean_html_text(text: str) -> str:
    """
    Removes HTML tags and decodes HTML entities from text.
    """
    if not text:
        return "No description available"
    
    try:
        # Decode HTML entities (e.g., &lt; becomes <)
        text = html.unescape(text)
        
        # Remove HTML tags using BeautifulSoup
        soup = BeautifulSoup(text, 'html.parser')
        clean_text = soup.get_text(separator=' ', strip=True)
        
        # Remove extra whitespace
        clean_text = ' '.join(clean_text.split())
        
        return clean_text if clean_text else "No description available"
    
    except Exception as e:
        logging.error(f"Error cleaning HTML text: {e}")
        return "No description available"

# --- New Function: EMA Pullback Signal ---

def detect_ema_pullback_with_state(hist: pd.DataFrame,
                                   target_profit_pct: float = 10.0,
                                   stop_loss_pct: float = 20.0) -> pd.DataFrame:
    try:
        hist = hist.copy()

        if hist.index.duplicated().any():
            logging.warning(f"Found {hist.index.duplicated().sum()} duplicate dates, keeping last")
            hist = hist[~hist.index.duplicated(keep='last')]
        
        # Ensure data is sorted by date
        hist = hist.sort_index()

        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in hist.columns for col in required_cols):
            # Return empty signals if data is invalid
            hist['Pullback_Signal'] = False
            hist['Entry_Price'] = np.nan
            hist['Target_Price'] = np.nan
            hist['Stop_Price'] = np.nan
            hist['Position_Active'] = False
            hist['Exit_Type'] = ''
            return hist

        # Ensure index is DatetimeIndex and timezone-naive
        if not isinstance(hist.index, pd.DatetimeIndex):
            hist.index = pd.to_datetime(hist.index, errors='coerce')

        if hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)

        # Calculate EMAs using TradingView's method - return numpy arrays
        def pine_ema_exact(series, length):
            alpha = 2 / (length + 1)
            result = np.zeros(len(series))
            result[:] = np.nan

            if len(series) >= length:
                result[length - 1] = series.iloc[:length].mean()
                for i in range(length, len(series)):
                    result[i] = alpha * series.iloc[i] + (1 - alpha) * result[i - 1]

            return result

        # Calculate EMAs as numpy arrays
        ema_f_vals = pine_ema_exact(hist['Close'], 10)
        ema_m_vals = pine_ema_exact(hist['Close'], 20)
        ema_s_vals = pine_ema_exact(hist['Close'], 50)

        # Store as columns
        hist['EMA_F'] = ema_f_vals
        hist['EMA_M'] = ema_m_vals
        hist['EMA_S'] = ema_s_vals

        # Extract all values as numpy arrays
        n = len(hist)
        low_vals = hist['Low'].to_numpy()
        high_vals = hist['High'].to_numpy()
        close_vals = hist['Close'].to_numpy()
        open_vals = hist['Open'].to_numpy()

        # Calculate signal conditions
        trend_aligned = (
            (ema_f_vals > ema_m_vals) &
            (ema_m_vals > ema_s_vals) &
            ~np.isnan(ema_f_vals) &
            ~np.isnan(ema_m_vals) &
            ~np.isnan(ema_s_vals)
        )

        pullback_touch = (
            (low_vals <= ema_m_vals) &
            (high_vals >= ema_m_vals) &
            ~np.isnan(low_vals) &
            ~np.isnan(high_vals) &
            ~np.isnan(ema_m_vals)
        )

        reversal_confirmation = (
            (close_vals > open_vals) &
            ~np.isnan(close_vals) &
            ~np.isnan(open_vals)
        )

        signal_today = trend_aligned & pullback_touch & reversal_confirmation

        # Initialize state arrays
        position_active = np.zeros(n, dtype=bool)
        entry_price = np.full(n, np.nan)
        target_price = np.full(n, np.nan)
        stop_price = np.full(n, np.nan)
        buy_signal = np.zeros(n, dtype=bool)
        exit_type = np.array([''] * n, dtype=object)

        # State variables
        current_entry = np.nan
        current_in_trade = False

        # Iterate through bars - start from 50 to ensure EMAs are valid
        for i in range(50, n):
            try:
                # Extract scalar values safely
                prev_signal = bool(signal_today[i-1])
                current_close = float(close_vals[i])
                current_open = float(open_vals[i])
            except (IndexError, TypeError, ValueError):
                continue

            # Skip if NaN
            if np.isnan(current_close) or np.isnan(current_open):
                continue

            position_active[i] = current_in_trade

            # Check for EXIT conditions
            if current_in_trade:
                entry_price[i] = current_entry
                target_price[i] = current_entry * (1 + target_profit_pct / 100)
                stop_price[i] = current_entry * (1 - stop_loss_pct / 100)

                # Check for TARGET hit
                if current_close >= target_price[i]:
                    current_in_trade = False
                    current_entry = np.nan
                    position_active[i] = False
                    exit_type[i] = 'Target'

                # Check for STOP LOSS hit
                elif current_close < stop_price[i]:
                    current_in_trade = False
                    current_entry = np.nan
                    position_active[i] = False
                    exit_type[i] = 'Stop'

            # --- REVISED SECTION IN detect_ema_pullback_with_state ---

            # Replace the existing "Check for NEW ENTRY" block with this:
            if prev_signal:
                buy_signal[i] = True  # Always record the signal
                
                # Only reset entry/target/stop if NOT currently in a trade 
                # OR if you want the newest signal to overwrite the old levels:
                if not current_in_trade:
                    current_entry = current_open
                    current_in_trade = True
                    entry_price[i] = current_entry
                    target_price[i] = current_entry * (1 + target_profit_pct / 100)
                    stop_price[i] = current_entry * (1 - stop_loss_pct / 100)

        # Add columns to dataframe
        hist['Pullback_Signal'] = buy_signal
        hist['Entry_Price'] = entry_price
        hist['Target_Price'] = target_price
        hist['Stop_Price'] = stop_price
        hist['Position_Active'] = position_active
        hist['Exit_Type'] = exit_type

        return hist

    except Exception as e:
        # If anything fails, return hist with empty signal columns
        logging.error(f"Error in detect_ema_pullback_with_state: {e}")
        hist['Pullback_Signal'] = False
        hist['Entry_Price'] = np.nan
        hist['Target_Price'] = np.nan
        hist['Stop_Price'] = np.nan
        hist['Position_Active'] = False
        hist['Exit_Type'] = ''
        return hist


def get_active_signals(hist: pd.DataFrame) -> pd.DataFrame:
    """
    Extract only the rows with active buy signals.

    Parameters:
    -----------
    hist : pd.DataFrame
        Processed historical data with signals

    Returns:
    --------
    pd.DataFrame with only signal rows and relevant columns
    """
    signals = hist[hist['Pullback_Signal']].copy()

    if not signals.empty:
        # Add useful information
        signals['Signal_Date'] = signals.index
        signals['Entry_At_Open'] = signals['Entry_Price']
        signals['Target'] = signals['Target_Price']

        # Use Stop_Price if it exists, otherwise fall back to Freeze_Price for backward compatibility
        if 'Stop_Price' in signals.columns:
            signals['Risk'] = signals['Stop_Price']
        elif 'Freeze_Price' in signals.columns:
            signals['Risk'] = signals['Freeze_Price']
        else:
            # Calculate stop loss if neither exists
            signals['Risk'] = signals['Entry_Price'] * 0.8  # 20% stop loss

        # Calculate potential profit
        signals['Potential_Profit_%'] = (
            (signals['Target'] - signals['Entry_Price']) / signals['Entry_Price'] * 100
        )

        # Select relevant columns
        return signals[[
            'Signal_Date', 'Entry_At_Open', 'Target', 'Risk',
            'Potential_Profit_%', 'Close'
        ]]

    return pd.DataFrame()


def backtest_strategy(hist: pd.DataFrame, initial_capital: float = 100000.0) -> dict:
    """
    Backtest the strategy and return performance metrics.

    Parameters:
    -----------
    hist : pd.DataFrame
        Historical data with signals already detected
    initial_capital : float
        Starting capital

    Returns:
    --------
    dict with performance metrics
    """
    trades = []
    capital = initial_capital
    position_size = 0
    entry_price = 0

    for i in range(len(hist)):
        row = hist.iloc[i]

        # Entry
        if row['Pullback_Signal'] and position_size == 0:
            entry_price = row['Entry_Price']
            position_size = (capital * 0.10) / entry_price  # 10% of equity

            # Use Stop_Price if available, otherwise Freeze_Price
            risk_price = row.get('Stop_Price') or row.get('Freeze_Price')

            trades.append({
                'entry_date': row.name,
                'entry_price': entry_price,
                'target': row['Target_Price'],
                'risk': risk_price,
                'status': 'open'
            })

        # Exit (target or stop)
        if position_size > 0 and len(trades) > 0:
            last_trade = trades[-1]

            # Target hit
            if row['Close'] >= last_trade['target']:
                profit = (last_trade['target'] - entry_price) * position_size
                capital += profit
                last_trade['exit_date'] = row.name
                last_trade['exit_price'] = last_trade['target']
                last_trade['profit'] = profit
                last_trade['status'] = 'win'
                position_size = 0

            # Stop/Risk hit
            elif last_trade['risk'] is not None and row['Close'] <= last_trade['risk']:
                loss = (last_trade['risk'] - entry_price) * position_size
                capital += loss
                last_trade['exit_date'] = row.name
                last_trade['exit_price'] = last_trade['risk']
                last_trade['profit'] = loss
                last_trade['status'] = 'loss'
                position_size = 0

    # Calculate metrics
    completed_trades = [t for t in trades if t['status'] != 'open']

    if not completed_trades:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'final_capital': capital,
            'total_return_%': 0,
            'trades': trades
        }

    wins = [t for t in completed_trades if t['status'] == 'win']
    losses = [t for t in completed_trades if t['status'] == 'loss']

    return {
        'total_trades': len(completed_trades),
        'winning_trades': len(wins),
        'losing_trades': len(losses),
        'win_rate': len(wins) / len(completed_trades) * 100 if completed_trades else 0,
        'final_capital': capital,
        'total_return_%': (capital - initial_capital) / initial_capital * 100,
        'avg_win': np.mean([t['profit'] for t in wins]) if wins else 0,
        'avg_loss': np.mean([t['profit'] for t in losses]) if losses else 0,
        'trades': trades
    }


def plot_strategy_chart(hist: pd.DataFrame, ticker: str):
    """
    Creates an enhanced chart showing the strategy signals and levels.
    This replicates the TradingView Pine Script visual output.
    """

    # Create figure with secondary y-axis for EMAs
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{ticker} - Mind-Free 10% Target System', 'Position Status')
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name='Price',
            showlegend=False
        ),
        row=1, col=1
    )

    # EMAs
    fig.add_trace(
        go.Scatter(
            x=hist.index, y=hist['EMA_F'],
            name='Fast EMA (10)',
            line=dict(color='#2962FF', width=1)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=hist.index, y=hist['EMA_M'],
            name='Medium EMA (20)',
            line=dict(color='#FF6D00', width=1)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=hist.index, y=hist['EMA_S'],
            name='Slow EMA (50)',
            line=dict(color='#E040FB', width=1)
        ),
        row=1, col=1
    )

    # Buy signals
    buy_signals = hist[hist['Pullback_Signal']]
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals['Low'] * 0.995,
                mode='markers',
                name='Buy Signal',
                marker=dict(
                    symbol='triangle-up',
                    size=15,
                    color='#00E676',
                    line=dict(color='white', width=1)
                )
            ),
            row=1, col=1
        )

    # Target price line (10%)
    target_data = hist[hist['Target_Price'].notna()]
    if not target_data.empty:
        fig.add_trace(
            go.Scatter(
                x=target_data.index,
                y=target_data['Target_Price'],
                name='10% Target',
                line=dict(color='#00E676', width=2, dash='dash'),
                mode='lines'
            ),
            row=1, col=1
        )

    # Stop loss line (20%)
    '''stop_data = hist[hist['Stop_Price'].notna()]
    if not stop_data.empty:
        fig.add_trace(
            go.Scatter(
                x=stop_data.index,
                y=stop_data['Stop_Price'],
                name='20% Stop Loss',
                line=dict(color='#FF1744', width=2, dash='dash'),
                mode='lines'
            ),
            row=1, col=1
        )'''

    # Position status (subplot)
    fig.add_trace(
        go.Scatter(
            x=hist.index,
            y=hist['Position_Active'].astype(int),
            name='Position Active',
            fill='tozeroy',
            line=dict(color='#00E676', width=0),
            fillcolor='rgba(0, 230, 118, 0.3)'
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        height=800,
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Status", row=2, col=1)

    return fig


def compare_ema_calculations(hist: pd.DataFrame, ticker: str):
    """
    Diagnostic function to compare different EMA calculation methods.
    Use this to debug discrepancies with TradingView.

    Parameters:
    -----------
    hist : pd.DataFrame
        Historical OHLCV data
    ticker : str
        Stock ticker symbol

    Returns:
    --------
    DataFrame showing EMA values side-by-side for comparison
    """
    # Method 1: Pandas default (adjust=True) - WRONG for TradingView
    ema_pandas_adjust_true = hist['Close'].ewm(span=20, adjust=True).mean()

    # Method 2: Pandas (adjust=False, min_periods=1) - WRONG initial value
    ema_pandas_adjust_false = hist['Close'].ewm(span=20, adjust=False, min_periods=1).mean()

    # Method 3: TradingView exact method (our implementation)
    ema_tradingview = pine_ema_exact(hist['Close'], 20)

    comparison = pd.DataFrame({
        'Close': hist['Close'],
        'Pandas_adjust=True': ema_pandas_adjust_true,
        'Pandas_adjust=False': ema_pandas_adjust_false,
        'TradingView_Method': ema_tradingview
    })

    # Show first 25 rows to see the difference in initialization
    print(f"\n=== EMA Comparison for {ticker} (first 25 rows) ===")
    print(comparison.head(25))

    # Show rows around potential signal dates
    print(f"\n=== Recent data (last 10 rows) ===")
    print(comparison.tail(10))

    return comparison


def pine_ema_exact(series, length):
    """
    Exact replication of TradingView's EMA calculation.
    This is extracted as a standalone function for testing.
    """
    alpha = 2 / (length + 1)
    result = np.zeros(len(series))
    result[:] = np.nan

    # First valid value is SMA
    if len(series) >= length:
        result[length - 1] = series.iloc[:length].mean()

        # Then apply iterative EMA formula
        for i in range(length, len(series)):
            result[i] = alpha * series.iloc[i] + (1 - alpha) * result[i - 1]

    return pd.Series(result, index=series.index)

# --- End New Function ---


def run_single_ticker_analysis(ticker_symbol: str, start_date: datetime, end_date: datetime, interval: str, analysis_index: int):
    ticker_base = ticker_symbol.strip().upper()
    if not ticker_base.endswith(".NS"):
        full_ticker = f"{ticker_base}.NS"
    else:
        full_ticker = ticker_base

    with st.expander(f"Analysis for {full_ticker}", expanded=False):
        try:
            time.sleep(1)
            hist = get_historical_data(full_ticker, start_date, end_date, interval)

            if hist.empty:
                st.error(f"❌ Could not fetch data for {full_ticker}")
                update_summary_table(ticker_base, '❌ (Fetch Failed)', 'N/A')
                return

            if len(hist) < 50:
                st.warning(f"⚠️ Insufficient data for {full_ticker}")
                update_summary_table(ticker_base, '❌ (Insufficient Data)', 'N/A')
                return

            # ADD THIS: Ensure timezone-naive index
            if hasattr(hist.index, 'tz') and hist.index.tz is not None:
                hist.index = hist.index.tz_localize(None)
            
            # ADD THIS: Verify required columns
            if not all(col in hist.columns for col in ['Open', 'High', 'Low', 'Close']):
                st.error(f"❌ Missing required price columns for {full_ticker}")
                update_summary_table(ticker_base, '❌ (Invalid Data)', 'N/A')
                return

            # Use the CORRECTED signal detection with stop_loss_pct parameter
            hist = detect_ema_pullback_with_state(hist,
                target_profit_pct=10.0,
                stop_loss_pct=20.0  # Changed from drawdown_freeze_pct
            )

            # Count signals
            signal_count = hist['Pullback_Signal'].sum()
            st.info(f"Found {signal_count} buy signals for {full_ticker}")

            if signal_count > 0:
                # Show signal details
                signals_df = get_active_signals(hist)
                st.dataframe(signals_df)

                # Backtest results
                backtest_results = backtest_strategy(hist)

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Trades", backtest_results['total_trades'])
                col2.metric("Win Rate", f"{backtest_results['win_rate']:.1f}%")
                col3.metric("Total Return", f"{backtest_results['total_return_%']:.2f}%")
                col4.metric("Final Capital", f"₹{backtest_results['final_capital']:,.0f}")

                # Plot the enhanced chart
                fig = plot_strategy_chart(hist, full_ticker)
                st.plotly_chart(fig, use_container_width=True, key=f"strategy_chart_{ticker_symbol}_{analysis_index}")
            else:
                st.warning(f"No signals detected for {full_ticker}")

            # Update summary
            has_signal = signal_count > 0
            latest_date = hist[hist['Pullback_Signal']].index.max().strftime('%Y-%m-%d') if has_signal else 'N/A'
            status = '✅ Buy Signal' if has_signal else '❌ No Signal'
            update_summary_table(ticker_base, status, latest_date)
            st.success(f"✓ Updated summary for {ticker_base}")

        except Exception as e:
            st.error(f"❌ Error analyzing {full_ticker}: {str(e)}")
            logging.error(f"Error in run_single_ticker_analysis for {full_ticker}: {e}", exc_info=True)
            update_summary_table(ticker_base, '❌ (Error)', 'N/A')

def plot_candlestick_with_signals(hist: pd.DataFrame, ticker: str, signal_column: str, analysis_index: int): # ADD analysis_index
    """Plots a candlestick chart with markers for the new signal_column."""
    fig = go.Figure(data=[go.Candlestick(
        x=hist.index,
        open=hist['Open'],
        high=hist['High'],
        low=hist['Low'],
        close=hist['Close'],
        name='OHLC'
    )])

    # --- UPDATED LINE TO FIND THE NEW SIGNAL ---
    signals = hist[hist[signal_column]]
    # ------------------------------------------

    if not signals.empty:
        fig.add_trace(go.Scatter(
            x=signals.index,
            y=signals['High'],
            mode='markers',
            name='Pullback Buy Signal', # Renamed display name
            marker=dict(symbol='triangle-up', color='green', size=10) # Changed marker color to green for Buy
        ))

    fig.update_layout(
        title=f'{ticker} Candlestick Chart with Pullback Buy Signals', # Renamed title
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{ticker}_{analysis_index}") # <--- USE INDEX IN KEY

@st.cache_data(ttl=3600)  # Cache for 1 hour or until session ends
def load_top_100_sweeps():
    top_100_stocks = get_top_100_stocks()
    # NOTE: This function call is now misleading, but kept for compatibility.
    summary_df = build_liquidity_sweep_summary(top_100_stocks, batch_size=5, delay_seconds=5)
    return summary_df

def process_and_store_ticker(ticker: str, start_date: datetime, end_date: datetime) -> (bool, str):
    hist = get_historical_data(ticker, start_date, end_date, interval='1d')

    if hist.empty or len(hist) < 20:
        print(f"[WARN] Insufficient data for {ticker}")
        return False, None

    try:
        hist = enrich_with_trend_and_direction(hist)
        # NOTE: Using the old detect_liquidity_sweeps function for pattern detection only.
        hist = detect_liquidity_sweeps(hist, volume_multiplier=3.0, atr_multiplier=1.5)

        print(f"[DEBUG] Liquidity Sweep column around 2025-09-07 for {ticker}:")
        print(hist.loc['2025-09-05':'2025-09-10'][['Open', 'High', 'Low', 'Close', 'Volume', 'Liquidity_Sweep']])
        high_conf_sweeps = detect_high_confidence_sweeps(
            hist, supports=[], resistances=[], strict_mode=True, min_confidence=0.5
        )
        high_conf_sweeps = annotate_sweep_types(hist, high_conf_sweeps, lookahead=3)
        print(f"[DEBUG] High-confidence sweeps for {ticker}:")
        if not high_conf_sweeps.empty:
            print(high_conf_sweeps[['Date', 'Liquidity_Sweep', 'Confidence_Score', 'Sweep_Type']].tail(10))
        else:
            print("[DEBUG] No high-confidence sweeps detected.")


        hist.to_csv(f"{DATA_DIR}/{ticker}.csv")

        if not high_conf_sweeps.empty:
            if 'Date' in high_conf_sweeps.columns:
                high_conf_sweeps['Date'] = pd.to_datetime(high_conf_sweeps['Date'], errors='coerce').dt.date
                valid_dates = high_conf_sweeps['Date'].dropna()

                if not valid_dates.empty:
                    latest_sweep_date = valid_dates.max().strftime('%Y-%m-%d')
                    print(f"[INFO] {ticker}: Latest high-confidence sweep on {latest_sweep_date}")
                    return True, latest_sweep_date
                else:
                    print(f"[WARN] No valid sweep dates for {ticker}")

        print(f"[INFO] {ticker}: No high-confidence liquidity sweeps detected.")
        return False, None

    except Exception as e:
        print(f"[ERROR] Failed to process {ticker}: {type(e).__name__} - {e}")
        return False, None

def update_summary_table(ticker: str, status: str, signal_date: str):
    """
    Loads the summary CSV, updates or adds a row for the given ticker,
    and saves it back to disk.
    """
    # Load existing data or create an empty DataFrame
    if os.path.exists(SUMMARY_FILE):
        summary_df = pd.read_csv(SUMMARY_FILE)
    else:
        # Renamed header to reflect the new signal type
        summary_df = pd.DataFrame(columns=['Ticker', 'Latest Signal Status', 'Latest Signal Date', 'Last Updated'])

    # Prepare the new data row
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = {
        'Ticker': ticker,
        'Latest Signal Status': status,
        'Latest Signal Date': signal_date,
        'Last Updated': now
    }

    # Remove the old entry for the ticker, if it exists
    summary_df = summary_df[summary_df['Ticker'] != ticker]

    # Add the new/updated entry
    summary_df = pd.concat([summary_df, pd.DataFrame([new_row])], ignore_index=True)

    # Save the updated table
    summary_df.to_csv(SUMMARY_FILE, index=False)
    logging.info(f"Updated summary for {ticker}.")

def show_summary_table():
    """Displays the summary table in the Streamlit app, sorted by the latest signal date."""
    if os.path.exists(SUMMARY_FILE):
        # Renamed header to reflect the new signal type
        st.subheader("📊 EMA Pullback Signal Summary")
        summary_df = pd.read_csv(SUMMARY_FILE)

        # Convert date column to datetime for proper sorting, placing non-dates (N/A) at the end
        summary_df['Sortable_Date'] = pd.to_datetime(summary_df['Latest Signal Date'], errors='coerce')

        # Sort by the new sortable date column, newest first
        summary_df = summary_df.sort_values(by='Sortable_Date', ascending=False)

        # Drop the temporary column before displaying
        summary_df = summary_df.drop(columns=['Sortable_Date'])

        st.dataframe(summary_df)
    else:
        st.info("ℹ️ No summary data has been saved yet. Run an analysis to create the summary.")

END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=45)
CUT_OFF_DATE = (datetime.utcnow() - timedelta(days=90)).replace(hour=0, minute=0, second=0, microsecond=0)

def check_liquidity_sweep_for_summary(ticker: str) -> (bool, str):
    """Check for signals and return status and latest signal date."""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=90)

    hist = get_historical_data(f"{ticker}.NS", start_date, end_date, interval='1d')
    if hist.empty or len(hist) < 50:
        return False, 'N/A'

    try:
        hist = detect_ema_pullback_with_state(hist, target_profit_pct=10.0, stop_loss_pct=20.0)
        signals = hist[hist['Pullback_Signal']]

        if signals.empty:
            return False, 'N/A'

        latest_signal_date = signals.index.max().strftime('%Y-%m-%d')
        return True, latest_signal_date

    except Exception as e:
        logging.error(f"Error in check_liquidity_sweep_for_summary for {ticker}: {e}")
        return False, 'N/A'

def get_high_conf_sweeps_for_ticker(ticker):
    hist = get_historical_data(ticker, START_DATE, END_DATE, interval='1d')
    if hist.empty or len(hist) < 20:
        return pd.DataFrame()  # Return empty DataFrame

    try:
        hist = enrich_with_trend_and_direction(hist)
        # NOTE: This function still uses the old Liquidity_Sweep detection,
        # but the main scanner logic now uses the new Pullback_Signal.
        hist = detect_liquidity_sweeps(hist, volume_multiplier=3.0, atr_multiplier=1.5)
        high_conf_sweeps = detect_high_confidence_sweeps(hist, supports=[], resistances=[], strict_mode=True, min_confidence=0.5)
        high_conf_sweeps = annotate_sweep_types(hist, high_conf_sweeps, lookahead=3)

        if 'Date' not in high_conf_sweeps.columns:
            high_conf_sweeps = high_conf_sweeps.reset_index()
            high_conf_sweeps['Date'] = pd.to_datetime(high_conf_sweeps['Datetime']).dt.tz_localize(None)

        high_conf_sweeps['Date'] = pd.to_datetime(high_conf_sweeps['Date']).dt.tz_localize(None)

        return high_conf_sweeps

    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return pd.DataFrame()

# --- 1. Define the Scraper Function ---
@st.cache_data(ttl=3600) # Cache for 1 hour
def get_top_100_stocks() -> List[str]:
    """Scrapes and VALIDATES the top 100 stock tickers."""
    base_url = "https://www.screener.in/screens/885655/top-100-stocks/?page="
    tickers = []
    st.write("Scraping tickers from screener.in...")

    try:
        for page in range(1, 6): # Scrape first 4 pages
            url = base_url + str(page)
            response = requests.get(url, headers=HEADERS)
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'class': 'data-table'})
            if table and table.find('tbody'):
                for row in table.find('tbody').find_all('tr'):
                    if len(row.find_all('td')) > 1:
                        company_link = row.find_all('td')[1].find('a')
                        if company_link and 'href' in company_link.attrs:
                            ticker = company_link['href'].split('/')[2]
                            tickers.append(ticker.strip())

        # --- VALIDATION STEP ---
        st.write(f"Found {len(tickers)} tickers. Now validating with Yahoo Finance...")
        valid_tickers = []
        progress_bar = st.progress(0)
        for i, ticker in enumerate(list(set(tickers))):
            try:
                # A quick check to see if the ticker is valid on yfinance
                data = yf.download(f"{ticker}.NS", period="5d", progress=False)
                if not data.empty:
                    valid_tickers.append(ticker)
            except Exception:
                continue # Skip invalid tickers
            finally:
                progress_bar.progress((i + 1) / len(tickers))

        st.success(f"Validated {len(valid_tickers)} tickers.")
        return valid_tickers

    except Exception as e:
        st.error(f"Failed to scrape top 100 stocks: {e}")
        return []

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_historical_data(ticker: str, start: datetime, end: datetime, interval: str, max_retries: int = 3) -> pd.DataFrame:
    """
    Fetches historical stock data from Yahoo Finance with retry logic.
    """
    for attempt in range(max_retries):
        try:
            # Add delay between retries
            if attempt > 0:
                wait_time = 2 ** attempt  # Exponential backoff: 2, 4, 8 seconds
                logging.info(f"Retry {attempt + 1} for {ticker} after {wait_time}s delay")
                time.sleep(wait_time)

            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                progress=False,
                auto_adjust=False,
                timeout=10
            )

            if df.empty:
                logging.warning(f"Empty DataFrame returned for {ticker} (attempt {attempt + 1})")
                continue
            
            # FIX: Handle multi-level columns that yfinance sometimes returns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Ensure index is timezone-naive DatetimeIndex
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # Remove duplicate dates (keep last)
            if df.index.duplicated().any():
                logging.warning(f"Removing {df.index.duplicated().sum()} duplicate dates for {ticker}")
                df = df[~df.index.duplicated(keep='last')]
            
            # Sort by date
            df = df.sort_index()
            
            # Drop rows with NaN in critical columns
            critical_cols = ['Open', 'High', 'Low', 'Close']
            available_cols = [col for col in critical_cols if col in df.columns]
            if available_cols:
                df = df.dropna(subset=available_cols)

            logging.info(f"Successfully fetched {len(df)} rows for {ticker}")
            return df

        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed for {ticker}: {e}")
            if attempt == max_retries - 1:
                logging.error(f"All retries exhausted for {ticker}")
                return pd.DataFrame()

    return pd.DataFrame()

def check_liquidity_sweep(ticker: str, interval: str = '1d') -> bool:
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=45)  # Ensure sufficient data

    # NOTE: This function is now DEPRECATED. The app now uses check_liquidity_sweep_for_summary
    # which calls the new detect_ema_pullback_signal. This function definition is kept
    # to avoid breaking older references, but its logic is outdated.

    # We will pass this through as a simple placeholder using the new logic for stability
    hist = get_historical_data(f"{ticker}.NS", start_date, end_date, interval)

    if hist.empty or len(hist) < 50:
        return False

    hist['ATR'] = compute_atr(hist)
    hist = detect_ema_pullback_with_state(hist, target_profit_pct=10.0)
    return hist['Pullback_Signal'].any()


@st.cache_data(ttl=3600)
def build_liquidity_sweep_summary(tickers: list, batch_size: int = 5, delay_seconds: int = 5):
    summary = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]

        print(f"[INFO] Processing batch {i//batch_size + 1} of {len(tickers)//batch_size + 1}")

        for ticker in batch:
            print(f"[INFO] Fetching liquidity sweep for {ticker}")
            # Use the new summary check function
            has_signal, latest_date = check_liquidity_sweep_for_summary(ticker)
            summary.append({
                'Ticker': ticker,
                'Latest Signal Status': '✅' if has_signal else '❌',
                'Latest Signal Date': latest_date
            })

            time.sleep(delay_seconds)  # Delay between each call

    return pd.DataFrame(summary)

def determine_sweep_type(hist: pd.DataFrame, idx: int, lookahead: int = 3) -> str:
    if idx + lookahead >= len(hist):
        return 'Neutral'  # Not enough future data

    future_close = hist['Close'].iloc[idx + lookahead]
    current_close = hist['Close'].iloc[idx]
    change_pct = (future_close - current_close) / current_close

    threshold = 0.005  # 0.5% price move threshold

    if change_pct >= threshold:
        return 'Buy'
    elif change_pct <= -threshold:
        return 'Sell'
    else:
        return 'Neutral'

def annotate_sweep_types(hist: pd.DataFrame, high_conf_sweeps: pd.DataFrame, lookahead: int = 3) -> pd.DataFrame:
    sweep_types = []

    hist_dates = hist.index.tolist()  # Ordered list of timestamps

    for sweep_date in high_conf_sweeps.index:
        try:
            idx = hist_dates.index(sweep_date)
        except ValueError:
            sweep_types.append('Neutral')  # If the date is missing, fallback
            continue

        sweep_types.append(determine_sweep_type(hist, idx, lookahead))

    high_conf_sweeps = high_conf_sweeps.copy()
    high_conf_sweeps['Sweep_Type'] = sweep_types
    return high_conf_sweeps

def plot_volume_obv_with_sweep_types(hist: pd.DataFrame, high_conf_sweeps: pd.DataFrame, ticker: str) -> None:
    # NOTE: This plotting function is for the old liquidity sweep data and is mostly kept as is,
    # but the main candlestick chart uses the new signal.
    # It still plots high_conf_sweeps (which uses the old Liquidity_Sweep flag).
    fig = go.Figure()

    # Volume Bars
    fig.add_trace(go.Bar(
        x=hist.index,
        y=hist['Volume'],
        name='Volume',
        marker_color='rgba(128, 128, 128, 0.5)'
    ))

    # OBV Line
    obv = compute_obv(hist)
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=obv,
        name='OBV',
        yaxis='y2',
        line=dict(color='orange')
    ))

    # Buy Sweeps (using old column names)
    buys = high_conf_sweeps[high_conf_sweeps['Sweep_Type'] == 'Buy']
    fig.add_trace(go.Scatter(
        x=buys.index,
        y=buys['Volume'],
        mode='markers+text',
        name='Buy Sweep',
        marker=dict(color='green', size=buys['Confidence_Score'] * 20, symbol='triangle-up'),
        text=buys['Confidence_Score'].round(2).astype(str),
        textposition="top center",
        yaxis='y1'
    ))

    # Sell Sweeps
    sells = high_conf_sweeps[high_conf_sweeps['Sweep_Type'] == 'Sell']
    fig.add_trace(go.Scatter(
        x=sells.index,
        y=sells['Volume'],
        mode='markers+text',
        name='Sell Sweep',
        marker=dict(color='red', size=sells['Confidence_Score'] * 20, symbol='triangle-down'),
        text=sells['Confidence_Score'].round(2).astype(str),
        textposition="bottom center",
        yaxis='y1'
    ))

    # Neutral Sweeps (Gray)
    neutrals = high_conf_sweeps[high_conf_sweeps['Sweep_Type'] == 'Neutral']
    fig.add_trace(go.Scatter(
        x=neutrals.index,
        y=neutrals['Volume'],
        mode='markers+text',
        name='Neutral Sweep',
        marker=dict(color='blue', size=neutrals['Confidence_Score'] * 20, symbol='circle'),
        text=neutrals['Confidence_Score'].round(2).astype(str),
        textposition="middle center",
        yaxis='y1'
    ))

    fig.update_layout(
        title_text="Volume & OBV with Buy/Sell Liquidity Sweeps",
        yaxis=dict(title="Volume"),
        yaxis2=dict(
            title="OBV",
            overlaying="y",
            side="right"
        ),
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True, key=f"volume_obv_{ticker}") # <--- ADDED KEY

    if not high_conf_sweeps.empty:
        latest_sweep = high_conf_sweeps.iloc[-1]
        latest_date = latest_sweep.name.strftime('%Y-%m-%d')
        latest_status = latest_sweep['Sweep_Type']

        # Use color coding for the status
        if latest_status == 'Buy':
            color = 'green'
        elif latest_status == 'Sell':
            color = 'red'
        else:
            color = 'orange'

        st.markdown(f"**{ticker}:** Latest signal was **<span style='color:{color};'>{latest_status}</span>** on **{latest_date}**", unsafe_allow_html=True)
    else:
        st.markdown(f"**{ticker}:** No high-confidence liquidity sweeps detected in this period.")

def enrich_with_trend_and_direction(hist: pd.DataFrame) -> pd.DataFrame:
    """Adds moving averages and trend direction, safely handling short data histories."""
    hist = hist.copy()
    if len(hist) >= 50:
        hist['MA_50'] = hist['Close'].rolling(window=50).mean()
    else:
        hist['MA_50'] = np.nan

    if len(hist) >= 200:
        hist['MA_200'] = hist['Close'].rolling(window=200).mean()
        hist['MA_50'] = hist['MA_50'].fillna(method='bfill')
        hist['MA_200'] = hist['MA_200'].fillna(method='bfill')
        hist['Trend'] = np.where(hist['MA_50'] > hist['MA_200'], 'Uptrend', 'Downtrend')
    else:
        hist['MA_200'] = np.nan
        hist['Trend'] = 'Neutral'

    hist['Sweep_Direction'] = np.where(hist['Close'].diff() > 0, 'Up', 'Down')
    return hist

def score_liquidity_sweep(row, supports, resistances) -> float:
    score = 0.0
    price = row['Close']
    proximity_thresh = 0.01

    for _, level in supports:
        if abs(price - level) / price < proximity_thresh:
            score += 0.3

    for _, level in resistances:
        if abs(price - level) / price < proximity_thresh:
            score += 0.3

    vol_ratio = row['Volume'] / row['Median_Volume'] if row['Median_Volume'] > 0 else 1
    score += min(0.3, (vol_ratio - 1) * 0.1)

    if row['Trend'] == 'Uptrend' and row['Sweep_Direction'] == 'Up':
        score += 0.2
    elif row['Trend'] == 'Downtrend' and row['Sweep_Direction'] == 'Down':
        score += 0.2

    atr_pct = row['ATR'] / row['Close']
    price_move_pct = row['Price_Change_Pct']
    if price_move_pct > atr_pct * 1.5:
        score += 0.2

    return round(min(score, 1.0), 2)



def plot_volume_obv_with_sweeps(hist: pd.DataFrame, high_conf_sweeps: pd.DataFrame, ticker: str) -> None:
    # Deprecated function kept for compatibility
    pass

def plot_candlestick_with_high_conf_sweeps(hist: pd.DataFrame, supports: list, resistances: list, ticker: str, high_conf_sweeps: pd.DataFrame) -> None:
    # Deprecated function kept for compatibility
    pass

def detect_high_confidence_sweeps(hist: pd.DataFrame, supports: list, resistances: list, strict_mode: bool, min_confidence: float = 0.5) -> pd.DataFrame:
    # Deprecated function kept for compatibility
    # This function relies on 'Liquidity_Sweep' which is no longer the main signal.
    sweeps = hist[hist['Liquidity_Sweep']].copy()

    if strict_mode:
        sweeps['Confidence_Score'] = sweeps.apply(lambda row: score_liquidity_sweep(row, supports, resistances), axis=1)
        high_conf_sweeps = sweeps[sweeps['Confidence_Score'] >= min_confidence]
    else:
        # In relaxed mode, treat all sweeps equally
        sweeps['Confidence_Score'] = 0.5
        high_conf_sweeps = sweeps

    return high_conf_sweeps

def plot_candlestick_with_liquidity_sweeps(hist: pd.DataFrame, supports: list, resistances: list, ticker: str) -> None:
    # Deprecated function kept for compatibility
    pass

def detect_liquidity_sweeps(hist: pd.DataFrame, volume_multiplier: float, atr_multiplier: float) -> pd.DataFrame:
    """
    Final, completely bulletproof version that handles all edge cases.
    (Kept for compatibility, but the main signal uses detect_ema_pullback_signal)
    """
    hist = hist.copy()
    source_cols = ['High', 'Low', 'Close', 'Volume']
    if not all(col in hist.columns for col in source_cols):
        logging.warning(f"Cannot detect sweeps; DataFrame missing one of: {source_cols}")
        hist['Liquidity_Sweep'] = False
        return hist

    # Calculate indicators
    hist['ATR'] = compute_atr(hist)
    hist['Median_Volume'] = hist['Volume'].rolling(window=20).median()
    hist['Price_Change_Pct'] = hist['Close'].pct_change().abs()

    # Fill NaN values
    hist['ATR'] = hist['ATR'].bfill().ffill().fillna(0)
    hist['Median_Volume'] = hist['Median_Volume'].bfill().ffill().fillna(0)
    hist['Price_Change_Pct'] = hist['Price_Change_Pct'].fillna(0)

    volume_series = hist['Volume'].squeeze()
    median_vol_series = hist['Median_Volume'].squeeze()

    volume_arr = volume_series.values if hasattr(volume_series, 'values') else np.array(volume_series)
    median_vol_arr = median_vol_series.values if hasattr(median_vol_series, 'values') else np.array(median_vol_series)

    volume_check = volume_arr > (median_vol_arr * volume_multiplier)

    close_series = hist['Close'].squeeze()
    atr_series = hist['ATR'].squeeze()
    price_change_series = hist['Price_Change_Pct'].squeeze()

    close_arr = close_series.values if hasattr(close_series, 'values') else np.array(close_series)
    atr_arr = atr_series.values if hasattr(atr_series, 'values') else np.array(atr_series)
    price_change_arr = price_change_series.values if hasattr(price_change_series, 'values') else np.array(price_change_series)

    safe_close_arr = np.where(close_arr == 0, 1, close_arr)
    atr_ratio_arr = (atr_arr / safe_close_arr) * atr_multiplier
    atr_check = price_change_arr > atr_ratio_arr

    liquidity_sweep_arr = volume_check & atr_check

    hist['Liquidity_Sweep'] = liquidity_sweep_arr

    return hist

def plot_candlestick_chart(hist: pd.DataFrame, supports: list, resistances: list, ticker: str) -> None:
    # Deprecated function kept for compatibility
    pass

@st.cache_data
def scrape_nse_insider_trading_dynamic(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    session = requests.Session()
    session.get('https://www.nseindia.com', headers=HEADERS)
    time.sleep(2)  # Get cookies

    all_data = []
    page = 1
    keep_scraping = True

    while keep_scraping:
        url = f'https://www.nseindia.com/api/corporate-insider-trading?symbol={ticker}&pageNo={page}'
        response = session.get(url, headers=HEADERS)
        if response.status_code != 200:
            print(f"Failed to fetch page {page}, status code {response.status_code}")
            break

        json_data = response.json()
        data = json_data.get('data', [])

        if not data:
            break

        for item in data:
            txn_date = datetime.strptime(item['transactionDate'], '%d-%b-%Y')
            if txn_date < start_date:
                keep_scraping = False
                break
            if start_date <= txn_date <= end_date:
                all_data.append({
                    'Date': txn_date,
                    'Insider_Name': item.get('reportingPersonName'),
                    'Buy_Sell': item.get('transactionType'),
                    'Shares': int(item.get('transactionQuantity', 0)),
                    'Reporting_Person_Type': item.get('reportingPersonType'),
                    'Remarks': item.get('remarks'),
                })

        page += 1
        time.sleep(1)  # Politeness delay

    df = pd.DataFrame(all_data)
    if not df.empty:
        df.set_index('Date', inplace=True)
    return df

# -----------------------------
# Utilities
# -----------------------------

def add_features(hist):
    """Add technical features to the dataframe with proper pandas handling."""
    # VWAP calculation
    if 'Volume' in hist.columns and 'High' in hist.columns and 'Low' in hist.columns:
        volume_sum = hist['Volume'].sum()
        if volume_sum > 0:  # Check scalar value, not Series
            hist['VWAP'] = (hist['Volume'] * (hist['High'] + hist['Low'] + hist['Close']) / 3).cumsum() / hist['Volume'].cumsum()
        else:
            hist['VWAP'] = hist['Close']  # Fallback if no volume data
    else:
        hist['VWAP'] = hist['Close']

    # OBV calculation
    hist['OBV'] = (np.sign(hist['Close'].diff()) * hist['Volume']).fillna(0).cumsum()

    # Support and Resistance (rolling min/max)
    hist['Support'] = hist['Low'].rolling(window=10).min()
    hist['Resistance'] = hist['High'].rolling(window=10).max()

    # Double Bottom Flag
    hist['Double_Bottom_Flag'] = (
        (hist['Low'].shift(2) > hist['Low'].shift(1)) &
        (hist['Low'].shift(1) < hist['Low'])
    ).astype(int)

    # Target for ML (1 if next day goes up)
    hist['Target'] = (hist['Close'].shift(-1) > hist['Close']).astype(int)

    # Fill NaN values using pandas method
    hist = hist.fillna(method='bfill')
    hist = hist.fillna(0)  # Fill any remaining NaNs

    return hist

# -----------------------------
# ML Model Inference
# -----------------------------
@st.cache_resource
def load_model():
    return lgb.Booster(model_file='lgb_model.txt')

def predict_movement(hist, model):
    features = ['VWAP', 'OBV', 'Support', 'Resistance', 'Double_Bottom_Flag']
    X = hist[features]
    hist['Predicted_Prob_Up'] = model.predict(X)
    return hist

def _safe_polyfit(x: List[float], y: List[float]) -> Tuple[float, float]:
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size < 2 or np.allclose(x, x[0]):
        return 0.0, float(np.nanmean(y) if y.size else 0.0)
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _line_value_at(slope: float, intercept: float, idx: int) -> float:
    return float(slope * idx + intercept)


def _line_values_for_indexes(indexes: List[int], slope: float, intercept: float) -> List[float]:
    return (slope * np.asarray(indexes) + intercept).tolist()


def _pct(a: float, b: float) -> float:
    return float(abs(a - b) / max(1e-9, abs(b)))


def _within(a: float, b: float, tol: float = 0.02) -> bool:
    return _pct(a, b) <= tol


@dataclass
class Pattern:
    name: str
    points: Dict[str, Any]
    explanation: str

# -----------------------------
# NEW: Price Forecasting
# -----------------------------

def generate_forecast(hist: pd.DataFrame, forecast_periods: int = 30, trend_lookback: int = 60) -> Optional[pd.Series]:
    """
    Generates a simple linear trend forecast. This is the final, robust version.
    It uses a standard business day range for maximum reliability.
    """
    # Check for sufficient historical data
    if len(hist) < trend_lookback:
        st.warning(f"Forecast skipped: Not enough data. Have {len(hist)}, need {trend_lookback}.")
        return None

    # Isolate recent data for the trend calculation
    recent_hist = hist.iloc[-trend_lookback:]

    # Ensure the index is a valid DatetimeIndex before proceeding
    if not isinstance(recent_hist.index, pd.DatetimeIndex):
        st.error("Forecast failed: The data's index is not in a recognized date format.")
        return None

    # Prepare data for linear regression (trend line)
    x = np.arange(len(recent_hist))
    y = recent_hist['Close'].values

    # Fit the trend line and handle any potential errors
    try:
        slope, intercept = np.polyfit(x, y, 1)
    except Exception as e:
        st.warning(f"Forecast failed: Could not calculate a trend line. Error: {e}")
        return None

    # Calculate the future price values based on the trend
    future_x = np.arange(len(recent_hist), len(recent_hist) + forecast_periods)
    forecast_values = slope * future_x + intercept

    # Get the last date from the historical data
    last_date = recent_hist.index[-1]

    # Generate the next 30 business days for the forecast's x-axis
    # This is the most reliable method and avoids frequency inference issues.
    future_dates = pd.bdate_range(start=last_date, periods=forecast_periods + 1, freq='B')[1:]

    return pd.Series(forecast_values, index=future_dates, name='Forecast')


# -----------------------------
# Technical indicators
# -----------------------------

def compute_avg_support_resistance(hist, lookback=10, confirm_touches=2):
    supports, resistances = find_support_resistance(hist, lookback=lookback, confirm_touches=confirm_touches)

    avg_support = np.mean([level for _, level in supports]) if supports else hist['Close'].min()
    avg_resistance = np.mean([level for _, level in resistances]) if resistances else hist['Close'].max()

    # Add same value for all rows
    hist['Support_Level'] = avg_support
    hist['Resistance_Level'] = avg_resistance
    hist['MA_5'] = hist['Close'].rolling(window=5).mean()
    hist['MA_20'] = hist['Close'].rolling(window=20).mean()
    hist['Momentum'] = hist['Close'] - hist['Close'].shift(5)
    hist['Volatility'] = hist['Close'].rolling(window=10).std()
    hist['Return'] = hist['Close'].pct_change()


    return hist

def compute_atr(hist: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    This is the final, most robust version of ATR calculation.
    It avoids intermediate DataFrames to prevent any dimensional errors.
    """
    hist = hist.copy()
    high = hist['High']
    low = hist['Low']
    close = hist['Close']

    # Calculate the three True Range components
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    # Calculate True Range by finding the maximum of the three components row by row
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))

    # Calculate the Average True Range
    atr = true_range.rolling(window=period).mean()

    return atr


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Robustly calculates the Relative Strength Index (RSI)."""
    delta = series.diff()
    up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.squeeze().fillna(50)

# -----------------------------
# Scoring / sizing (unchanged)
# -----------------------------
# (Omitted here for brevity in the explanation; actual functions are below and present.)

# -----------------------------
# Volume helpers & confirmations
# -----------------------------

def volume_surged(df: pd.DataFrame, idx: int, multiplier: float = 1.2, window: int = 20) -> bool:
    if idx <= 0:
        return False
    if 'Volume' not in df.columns:
        return False
    start = max(0, idx - window)
    baseline = df['Volume'].iloc[start:idx]
    if baseline.empty:
        return False
    median = baseline.median()
    return float(df['Volume'].iat[idx]) >= multiplier * float(median)


def _default_confirm_result() -> Tuple[bool, Optional[float], Optional[float], Optional[float], Optional[str]]:
    return False, None, None, None, None

# (Confirm functions unchanged)

def confirm_double_bottom(df: pd.DataFrame, details: Dict[str, Any]) -> Tuple[bool, Optional[float], Optional[float], Optional[float], Optional[str]]:
    neckline = details.get('neckline')
    if neckline is None:
        return _default_confirm_result()
    latest_idx = len(df) - 1
    if df['Close'].iat[latest_idx] > neckline and volume_surged(df, latest_idx):
        troughs = details.get('troughs')
        trough_price = float(min(troughs)) if troughs else float(df['Close'].min())
        target = float(neckline + (neckline - trough_price))
        stop_loss = float(trough_price * 0.99)
        return True, float(neckline), target, stop_loss, 'bullish'
    return _default_confirm_result()


def confirm_double_top(df: pd.DataFrame, details: Dict[str, Any]) -> Tuple[bool, Optional[float], Optional[float], Optional[float], Optional[str]]:
    neckline = details.get('neckline')
    peaks = details.get('peaks')
    if neckline is None:
        return _default_confirm_result()
    latest_idx = len(df) - 1
    if df['Close'].iat[latest_idx] < neckline and volume_surged(df, latest_idx):
        peak_price = float(max(peaks)) if peaks else float(df['Close'].max())
        target = float(neckline - (peak_price - neckline))
        stop_loss = float(peak_price * 1.01)
        return True, float(neckline), target, stop_loss, 'bearish'
    return _default_confirm_result()


def confirm_head_shoulders(df: pd.DataFrame, details: Dict[str, Any]) -> Tuple[bool, Optional[float], Optional[float], Optional[float], Optional[str]]:
    neckline = details.get('neckline')
    head = details.get('head')
    if neckline is None or head is None:
        return _default_confirm_result()
    latest_idx = len(df) - 1
    if df['Close'].iat[latest_idx] < neckline and volume_surged(df, latest_idx):
        target = float(neckline - (head - neckline))
        stop_loss = float(head * 1.01)
        return True, float(neckline), target, stop_loss, 'bearish'
    return _default_confirm_result()


def confirm_inverse_head_shoulders(df: pd.DataFrame, details: Dict[str, Any]) -> Tuple[bool, Optional[float], Optional[float], Optional[float], Optional[str]]:
    neckline = details.get('neckline')
    head = details.get('head')
    if neckline is None or head is None:
        return _default_confirm_result()
    latest_idx = len(df) - 1
    if df['Close'].iat[latest_idx] > neckline and volume_surged(df, latest_idx):
        target = float(neckline + (neckline - head))
        stop_loss = float(head * 0.99)
        return True, float(neckline), target, stop_loss, 'bullish'
    return _default_confirm_result()


def confirm_bull_flag(df: pd.DataFrame, details: Dict[str, Any]) -> Tuple[bool, Optional[float], Optional[float], Optional[float], Optional[str]]:
    resistance = details.get('resistance')
    pole_height = details.get('pole_height')
    if resistance is None or pole_height is None:
        return _default_confirm_result()
    latest_idx = len(df) - 1
    if df['Close'].iat[latest_idx] > resistance and volume_surged(df, latest_idx):
        stop_loss = float(df['Close'].iloc[-20]) if len(df) >= 20 else float(df['Close'].min())
        return True, float(resistance), float(resistance + pole_height), stop_loss, 'bullish'
    return _default_confirm_result()


def confirm_bear_flag(df: pd.DataFrame, details: Dict[str, Any]) -> Tuple[bool, Optional[float], Optional[float], Optional[float], Optional[str]]:
    support = details.get('support')
    pole_height = details.get('pole_height')
    if support is None or pole_height is None:
        return _default_confirm_result()
    latest_idx = len(df) - 1
    if df['Close'].iat[latest_idx] < support and volume_surged(df, latest_idx):
        stop_loss = float(df['Close'].iloc[-20]) if len(df) >= 20 else float(df['Close'].max())
        return True, float(support), float(support - pole_height), stop_loss, 'bearish'
    return _default_confirm_result()


def confirm_triangle(df: pd.DataFrame, details: Dict[str, Any]) -> Tuple[bool, Optional[float], Optional[float], Optional[float], Optional[str]]:
    support = details.get('support')
    resistance = details.get('resistance')
    height = details.get('height')
    if support is None or resistance is None:
        return _default_confirm_result()
    latest_idx = len(df) - 1
    if df['Close'].iat[latest_idx] > resistance and volume_surged(df, latest_idx):
        target = float(resistance + height) if height is not None else None
        return True, float(resistance), target, float(support), 'bullish'
    if df['Close'].iat[latest_idx] < support and volume_surged(df, latest_idx):
        target = float(support - height) if height is not None else None
        return True, float(support), target, float(resistance), 'bearish'
    return _default_confirm_result()

# -----------------------------
# Swing points
# -----------------------------

def swing_points(series: pd.Series, lookback: int = 3) -> Tuple[List[int], List[int]]:
    vals = np.asarray(series)
    highs: List[int] = []
    lows: List[int] = []
    n = len(vals)
    for i in range(lookback, n - lookback):
        window = vals[i - lookback: i + lookback + 1]
        center = vals[i]
        if center == window.max():
            highs.append(i)
        if center == window.min():
            lows.append(i)
    return highs, lows

# -----------------------------
# Pattern detectors
# -----------------------------
# (Same detectors present as before)

def detect_head_shoulders(df: pd.DataFrame, lookback_sw: int = 3, shoulder_tol: float = 0.04) -> List[Pattern]:
    close = df['Close'].reset_index(drop=True)
    highs, lows = swing_points(close, lookback=lookback_sw)
    patterns: List[Pattern] = []

    for i in range(len(highs) - 2):
        l, h, r = highs[i], highs[i + 1], highs[i + 2]
        if not (l < h < r):
            continue
        L, H, R = close.iloc[l], close.iloc[h], close.iloc[r]
        if H <= max(L, R):
            continue
        if not _within(L, R, tol=shoulder_tol):
            continue
        troughs_between = [j for j in lows if l < j < h]
        troughs_between2 = [j for j in lows if h < j < r]
        if not troughs_between or not troughs_between2:
            continue
        nl1 = min(troughs_between, key=lambda j: close.iloc[j])
        nl2 = min(troughs_between2, key=lambda j: close.iloc[j])
        slope, intercept = _safe_polyfit([nl1, nl2], [float(close.iloc[nl1]), float(close.iloc[nl2])])
        brk_idx = None
        for k in range(r + 1, min(r + 25, len(close))):
            nl = _line_value_at(slope, intercept, k)
            if close.iloc[k] < nl:
                brk_idx = k
                break
        patterns.append(Pattern(
            name='Head and Shoulders',
            points={
                'peaks_idx': [l, h, r],
                'peaks_price': [float(L), float(H), float(R)],
                'neckline_idx': [nl1, nl2],
                'neckline_price': [float(close.iloc[nl1]), float(close.iloc[nl2])],
                'breakout_idx': brk_idx,
            },
            explanation=(f"Three peaks with middle highest; shoulders within {int(shoulder_tol*100)}% tolerance. "
                          f"{'Breakout below neckline detected' if brk_idx is not None else 'No confirmed breakout in window'}.")
        ))

    for i in range(len(lows) - 2):
        l, h, r = lows[i], lows[i + 1], lows[i + 2]
        if not (l < h < r):
            continue
        L, Hm, R = close.iloc[l], close.iloc[h], close.iloc[r]
        if Hm >= min(L, R):
            continue
        if not _within(L, R, tol=shoulder_tol):
            continue
        peaks_between = [j for j in highs if l < j < h]
        peaks_between2 = [j for j in highs if h < j < r]
        if not peaks_between or not peaks_between2:
            continue
        nl1 = max(peaks_between, key=lambda j: close.iloc[j])
        nl2 = max(peaks_between2, key=lambda j: close.iloc[j])
        slope, intercept = _safe_polyfit([nl1, nl2], [float(close.iloc[nl1]), float(close.iloc[nl2])])
        brk_idx = None
        for k in range(r + 1, min(r + 25, len(close))):
            nl = _line_value_at(slope, intercept, k)
            if close.iloc[k] > nl:
                brk_idx = k
                break
        patterns.append(Pattern(
            name='Inverse Head and Shoulders',
            points={
                'troughs_idx': [l, h, r],
                'troughs_price': [float(L), float(Hm), float(R)],
                'neckline_idx': [nl1, nl2],
                'neckline_price': [float(close.iloc[nl1]), float(close.iloc[nl2])],
                'breakout_idx': brk_idx,
            },
            explanation=(f"Three troughs with middle lowest; shoulders within {int(shoulder_tol*100)}% tolerance. "
                          f"{'Breakout above neckline detected' if brk_idx is not None else 'No confirmed breakout in window'}.")
        ))

    return patterns

# (Triangles, flags, double tops/bottoms detectors are same as before; omitted here to keep snippet shorter)

def detect_triangles(df: pd.DataFrame, lookback_sw: int = 3, min_pts: int = 3, window: int = 120) -> List[Pattern]:
    close = df['Close'].reset_index(drop=True)
    highs, lows = swing_points(close, lookback=lookback_sw)
    patterns: List[Pattern] = []
    n = len(close)
    step = max(15, window // 6)
    for start in range(0, max(1, n - window), step):
        end = start + window
        sh = [h for h in highs if start <= h < end]
        sl = [l for l in lows if start <= l < end]
        if len(sh) < min_pts or len(sl) < min_pts:
            continue
        up_slope, up_int = _safe_polyfit(sh, close.iloc[sh])
        lo_slope, lo_int = _safe_polyfit(sl, close.iloc[sl])
        up_y_start, up_y_end = _line_value_at(up_slope, up_int, start), _line_value_at(up_slope, up_int, end - 1)
        lo_y_start, lo_y_end = _line_value_at(lo_slope, lo_int, start), _line_value_at(lo_slope, lo_int, end - 1)
        converging = (up_y_end < up_y_start) and (lo_y_end > lo_y_start) and (lo_y_end < up_y_end)
        if not converging:
            continue
        if abs(up_slope + lo_slope) < 1e-5:
            tri_type = 'Symmetrical Triangle'
        elif up_slope < 0 and lo_slope >= 0:
            tri_type = 'Symmetrical/Contracting Triangle'
        elif up_slope < 0 and abs(lo_slope) < 1e-5:
            tri_type = 'Descending Triangle'
        elif abs(up_slope) < 1e-5 and lo_slope > 0:
            tri_type = 'Ascending Triangle'
        else:
            tri_type = 'Triangle (contracting)'
        patterns.append(Pattern(
            name=tri_type,
            points={'window': [start, end - 1], 'upper_line': (up_slope, up_int), 'lower_line': (lo_slope, lo_int)},
            explanation=f'Converging trendlines. Upper slope {up_slope:.5f}, lower slope {lo_slope:.5f}.'
        ))
    return patterns


def detect_flags(df: pd.DataFrame,
                 impulse_look: int = 15, impulse_min_pct: float = 0.12,
                 consolidation_min: int = 5, consolidation_max: int = 25,
                 pullback_min: float = 0.23, pullback_max: float = 0.62) -> List[Pattern]:
    close = df['Close'].reset_index(drop=True)
    patterns: List[Pattern] = []
    n = len(close)
    i = 0
    while i < n - impulse_look - consolidation_min:
        start = i
        end_imp = i + impulse_look
        if end_imp >= n:
            break
        pct = (close.iloc[end_imp] - close.iloc[start]) / close.iloc[start]
        direction = 'bull' if pct >= impulse_min_pct else ('bear' if pct <= -impulse_min_pct else None)
        if direction:
            for end_cons in range(end_imp + consolidation_min, min(end_imp + consolidation_max, n - 1)):
                cons = close.iloc[end_imp:end_cons + 1]
                x = np.arange(len(cons))
                slope, intercept = _safe_polyfit(x.tolist(), cons.values.tolist())
                residuals = cons.values - (slope * x + intercept)
                channel_ok = residuals.std() < max(0.01 * close.iloc[end_imp], 0.02 * close.iloc[end_imp])
                if direction == 'bull':
                    retrace = abs((cons.min() - close.iloc[end_imp]) / (close.iloc[end_imp] - close.iloc[start]))
                else:
                    retrace = abs((cons.max() - close.iloc[end_imp]) / (close.iloc[end_imp] - close.iloc[start]))
                if channel_ok and pullback_min <= retrace <= pullback_max:
                    ptype = 'Bull Flag' if direction == 'bull' else 'Bear Flag'
                    patterns.append(Pattern(
                        name=ptype,
                        points={'impulse': [start, end_imp], 'consolidation': [end_imp, end_cons], 'channel_slope': slope, 'channel_intercept': intercept},
                        explanation=(f'Impulse {direction} move of {pct*100:.1f}% followed by tight {end_cons - end_imp + 1}-day consolidation.')
                    ))
                    i = end_cons
                    break
        i += 1
    return patterns


def detect_double_tops_bottoms(df: pd.DataFrame, tol: float = 0.02, min_gap: int = 10) -> List[Pattern]:
    close = df['Close'].reset_index(drop=True)
    highs, lows = swing_points(close, lookback=3)
    patterns: List[Pattern] = []
    for i in range(len(highs) - 1):
        a, b = highs[i], highs[i + 1]
        if b - a < min_gap:
            continue
        if _within(close.iloc[a], close.iloc[b], tol=tol):
            mids = close.iloc[a:b + 1]
            valley_idx = a + int(np.argmin(mids.values))
            patterns.append(Pattern(name='Double Top', points={'peaks': [a, b], 'neckline': int(valley_idx)}, explanation=f'Two highs within {int(tol*100)}% tolerance.'))
    for i in range(len(lows) - 1):
        a, b = lows[i], lows[i + 1]
        if b - a < min_gap:
            continue
        if _within(close.iloc[a], close.iloc[b], tol=tol):
            mids = close.iloc[a:b + 1]
            peak_idx = a + int(np.argmax(mids.values))
            patterns.append(Pattern(name='Double Bottom', points={'troughs': [a, b], 'neckline': int(peak_idx)}, explanation=f'Two lows within {int(tol*100)}% tolerance.'))
    return patterns


def detect_all_patterns(hist: pd.DataFrame) -> List[Pattern]:
    patterns: List[Pattern] = []
    try:
        patterns.extend(detect_head_shoulders(hist, lookback_sw=3, shoulder_tol=0.05))
    except Exception:
        pass
    try:
        patterns.extend(detect_triangles(hist, lookback_sw=3, min_pts=3, window=120))
    except Exception:
        pass
    try:
        patterns.extend(detect_flags(hist, impulse_look=15, impulse_min_pct=0.12, consolidation_min=5, consolidation_max=25))
    except Exception:
        pass
    try:
        patterns.extend(detect_double_tops_bottoms(hist, tol=0.02, min_gap=10))
    except Exception:
        pass
    return patterns

# -----------------------------
# Data Fetcher
# -----------------------------
@st.cache_data
def fetch_data(ticker: str, start: datetime.date, end: datetime.date, interval: str) -> pd.DataFrame:
    start_dt = datetime.combine(start, datetime.min.time())
    end_dt = datetime.combine(end, datetime.min.time())
    ts = yf.Ticker(ticker)
    hist = ts.history(start=start_dt, end=end_dt+timedelta(days=1), actions=True, interval=interval)
    if hist.empty:
        st.warning("No data found. Adjust your inputs.")
    return hist


def get_ticker_symbol(user_ticker: str, exchange: str) -> str:
    s = user_ticker.strip().upper()
    if '.' in s:
        return s
    if exchange == 'NSE (India)':
        return f'{s}.NS'
    if exchange == 'BSE (India)':
        return f'{s}.BO'
    return s

# -----------------------------
# Returns / Total Return
# -----------------------------

def compute_price_return(hist: pd.DataFrame, buy_date: datetime, invest_amount: float = 1000.0) -> Dict[str, Any]:
    price_col = 'Adj Close' if 'Adj Close' in hist.columns else 'Close'
    prices = hist[[price_col]].dropna()
    if prices.empty:
        raise ValueError('No price data available')
    # normalize timezone: remove tz info to avoid comparisons between tz-aware and naive datetimes
    if hasattr(prices.index, 'tz') and prices.index.tz is not None:
        prices = prices.copy()
        prices.index = prices.index.tz_convert(None)
    buy_date = pd.to_datetime(buy_date)
    idx = prices.index.searchsorted(buy_date)
    if idx >= len(prices):
        idx = len(prices) - 1
    buy_date_actual = prices.index[idx]
    buy_price = float(prices.iloc[idx][price_col])
    current_price = float(prices.iloc[-1][price_col])
    shares = invest_amount / buy_price
    value_today = shares * current_price
    return {
        'buy_date': buy_date_actual.date(),
        'buy_price': buy_price,
        'current_price': current_price,
        'shares': float(shares),
        'value_today': float(value_today),
        'return_pct': float((current_price / buy_price - 1) * 100)
    }


def compute_total_return(hist: pd.DataFrame, buy_date: datetime, invest_amount: float) -> Optional[Dict[str, Any]]:
    if hasattr(hist.index, 'tz') and hist.index.tz is not None:
        hist = hist.copy()
        hist.index = hist.index.tz_convert(None)
    buy_date = pd.to_datetime(buy_date)
    prices = hist[['Close']].copy()
    dividends = hist['Dividends'] if 'Dividends' in hist.columns else pd.Series(dtype=float)
    buy_idx = prices.index.searchsorted(buy_date)
    if buy_idx >= len(prices):
        st.error('Buy date is after available trading data.')
        return None
    buy_date_actual = prices.index[buy_idx]
    buy_price = float(prices.iloc[buy_idx]['Close'])
    shares = invest_amount / buy_price
    for div_date, div in dividends.items():
        if div_date >= buy_date_actual and div > 0:
            if div_date in prices.index:
                reinvest_price = float(prices.loc[div_date, 'Close'])
                div_cash = div * shares
                shares += div_cash / reinvest_price
    latest_date = prices.index[-1]
    latest_price = float(prices.iloc[-1]['Close'])
    current_value = shares * latest_price
    years = (latest_date - buy_date_actual).days / 365.25
    cagr = (current_value / invest_amount) ** (1 / years) - 1 if years > 0 else float('nan')
    return {
        'buy_date_actual': buy_date_actual.date(),
        'buy_price': float(buy_price),
        'shares_final': float(shares),
        'end_date': latest_date.date(),
        'end_price': float(latest_price),
        'final_value': float(current_value),
        'absolute_return': float(current_value - invest_amount),
        'pct_return': float((current_value / invest_amount - 1) * 100),
        'cagr': float(cagr)
    }

# -----------------------------
# Support/resistance & forward testing
# -----------------------------

def find_support_resistance(hist: pd.DataFrame, lookback: int = 5, confirm_touches: int = 2, tolerance: float = 0.01) -> Tuple[List[Tuple[pd.Timestamp, float]], List[Tuple[pd.Timestamp, float]]]:
    closes = hist['Close']
    support_levels: List[Tuple[pd.Timestamp, float]] = []
    resistance_levels: List[Tuple[pd.Timestamp, float]] = []
    for i in range(lookback, len(closes) - lookback):
        window = closes.iloc[i - lookback: i + lookback + 1]
        if closes.iloc[i] == window.min():
            support_levels.append((closes.index[i], float(closes.iloc[i])))
        if closes.iloc[i] == window.max():
            resistance_levels.append((closes.index[i], float(closes.iloc[i])))
    def _filter(levels: List[Tuple[pd.Timestamp, float]]) -> List[Tuple[pd.Timestamp, float]]:
        confirmed: List[Tuple[pd.Timestamp, float]] = []
        prices = closes.values
        for dt, lvl in levels:
            touches = int(np.sum(np.isclose(prices, lvl, rtol=tolerance)))
            if touches >= confirm_touches:
                confirmed.append((dt, float(lvl)))
        return confirmed
    return _filter(support_levels), _filter(resistance_levels)


def forward_test(hist: pd.DataFrame, levels: List[Tuple[pd.Timestamp, float]], test_type: str = 'support') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    results = []
    closes = hist['Close']
    for date, level in levels:
        test_points = {'1M': date + timedelta(days=30), '3M': date + timedelta(days=90), '1Y': date + timedelta(days=365)}
        outcome: Dict[str, str] = {}
        for label, future_date in test_points.items():
            idx = closes.index.searchsorted(future_date)
            if idx >= len(closes):
                continue
            future_price = float(closes.iloc[idx])
            if test_type == 'support':
                outcome[label] = 'Held' if future_price >= level else 'Broke'
            else:
                outcome[label] = 'Held' if future_price <= level else 'Broke'
        results.append({'date': date.date(), 'level': round(level, 2), **outcome})
    df = pd.DataFrame(results)
    score: Dict[str, Any] = {'Held %': None, 'Broke %': None, 'Total Levels Tested': len(df)}
    if not df.empty:
        cols = [c for c in ['1M', '3M', '1Y'] if c in df.columns]
        if cols:
            held_count = (df[cols] == 'Held').sum().sum()
            broke_count = (df[cols] == 'Broke').sum().sum()
            total = held_count + broke_count
            score['Held %'] = round(held_count / total * 100, 2) if total > 0 else None
            score['Broke %'] = round(broke_count / total * 100, 2) if total > 0 else None
    return df, score

# -----------------------------
# create_detected_patterns (smart)
# -----------------------------

def compute_pattern_score(p: Pattern, hist: pd.DataFrame, aggressiveness: float = 1.0) -> Tuple[float, Dict[str, float]]:
    # (reusing compute_pattern_score for scoring inside create_detected_patterns)
    return _compute_pattern_score_inner(p, hist, aggressiveness)


def _compute_pattern_score_inner(p: Pattern, hist: pd.DataFrame, aggressiveness: float = 1.0) -> Tuple[float, Dict[str, float]]:
    breakdown: Dict[str, float] = {
        'symmetry': 0.0,
        'touches': 0.0,
        'volume_surge': 0.0,  # renamed from 'volume'
        'obv_trend': 0.0,
        'vwap_position': 0.0,
        'vroc': 0.0,
        'ad_confirmation': 0.0,
        'trend': 0.0,
        'risk_reward': 0.0
    }

    close = hist['Close'].reset_index(drop=True)
    vol = hist['Volume'].reset_index(drop=True) if 'Volume' in hist.columns else pd.Series(np.ones(len(close)))
    atr = compute_atr(hist).reset_index(drop=True) if 'High' in hist.columns and 'Low' in hist.columns else pd.Series(np.ones(len(close))*np.nan)

    # --- Volume Surge (already partially there, make explicit) ---
    brk = p.points.get('breakout_idx')
    if brk is not None and isinstance(brk, int) and brk < len(vol):
        med = float(vol[max(0, brk-20):brk].median()) if brk > 0 else float(vol.median())
        vol_mult = float(vol.iloc[brk]) / max(1.0, med) if med > 0 else 1.0
        breakdown['volume_surge'] = min(2.0, vol_mult) - 1.0

    # --- OBV Trend ---
    obv = compute_obv(hist)
    if len(obv) > 20:
        obv_slope = np.polyfit(range(20), obv.iloc[-20:], 1)[0]
        if p.name in ('Double Bottom','Inverse Head and Shoulders','Bull Flag') and obv_slope > 0:
            breakdown['obv_trend'] = 1.0
        elif p.name in ('Double Top','Head and Shoulders','Bear Flag') and obv_slope < 0:
            breakdown['obv_trend'] = 1.0

    # --- VWAP Position ---
    vwap = compute_vwap(hist)
    if len(vwap) == len(close):
        price_now = close.iloc[-1]
        if p.name in ('Bull Flag','Double Bottom','Inverse Head and Shoulders') and price_now > vwap.iloc[-1]:
            breakdown['vwap_position'] = 1.0
        elif p.name in ('Bear Flag','Double Top','Head and Shoulders') and price_now < vwap.iloc[-1]:
            breakdown['vwap_position'] = 1.0
        elif abs(price_now - vwap.iloc[-1]) / price_now < 0.01:  # within 1%
            breakdown['vwap_position'] = 0.5

    # --- VROC ---
    vroc = compute_vroc(hist)
    if not vroc.empty:
        val = vroc.iloc[-1]
        breakdown['vroc'] = max(0.0, min(1.0, val * 5))  # amplify signal


    # --- Accumulation/Distribution Confirmation ---
    ad_line = compute_ad_line(hist)
    if len(ad_line) > 20:
        ad_slope = np.polyfit(range(20), ad_line.iloc[-20:], 1)[0]
        if p.name in ('Bull Flag','Double Bottom','Inverse Head and Shoulders') and ad_slope > 0:
            breakdown['ad_confirmation'] = 1.0
        elif p.name in ('Bear Flag','Double Top','Head and Shoulders') and ad_slope < 0:
            breakdown['ad_confirmation'] = 1.0

    breakdown: Dict[str, float] = {
        'symmetry': 0.0,
        'touches': 0.0,
        'volume_surge': 0.0,
        'obv_trend': 0.0,
        'vwap_position': 0.0,
        'vroc': 0.0,
        'ad_confirmation': 0.0,
        'trend': 0.0,
        'risk_reward': 0.0
    }


    if p.name in ('Head and Shoulders', 'Inverse Head and Shoulders'):
        peaks = p.points.get('peaks_price') or p.points.get('troughs_price')
        if peaks and len(peaks) == 3:
            left, mid, right = peaks
            s = 1 - min(1.0, _pct(left, right))
            breakdown['symmetry'] = float(s)
    elif p.name in ('Double Bottom', 'Double Top'):
        pts = p.points.get('peaks') or p.points.get('troughs')
        if pts and len(pts) >= 2:
            a, b = pts[:2]
            breakdown['symmetry'] = float(1 - min(1.0, _pct(close.iloc[a], close.iloc[b])))

    lvl = None
    if 'neckline_price' in p.points and p.points.get('neckline_price'):
        lvl = float(np.mean(p.points['neckline_price']))
    elif 'neckline' in p.points and p.points.get('neckline') is not None:
        idx = int(p.points['neckline'])
        lvl = float(close.iloc[idx])
    elif 'resistance' in p.points:
        lvl = float(p.points['resistance'])
    if lvl is not None:
        near = np.isclose(close.values, lvl, rtol=0.01)
        touches = int(np.sum(near))
        breakdown['touches'] = float(min(1.0, touches / 5.0))

    brk = p.points.get('breakout_idx')
    if brk is not None and isinstance(brk, int) and brk < len(vol):
        med = float(vol[max(0, brk-20):brk].median()) if brk > 0 else float(vol.median())
        vol_mult = float(vol.iloc[brk]) / max(1.0, med) if med > 0 else 1.0
        breakdown['volume_surge'] = float(min(2.0, vol_mult) - 1.0)
    else:
        recent_med = float(vol[-20:].median()) if len(vol) >= 20 else float(vol.median())
        overall_med = float(vol.median()) if len(vol)>0 else 1.0
        breakdown['volume_surge'] = float(min(1.0, recent_med / max(1.0, overall_med)))

    ma_short = hist['Close'].rolling(50, min_periods=10).mean()
    ma_long = hist['Close'].rolling(200, min_periods=10).mean()
    if len(ma_short.dropna()) and len(ma_long.dropna()):
        latest_ma_short = ma_short.iloc[-1]
        latest_ma_long = ma_long.iloc[-1]
        if p.name in ('Double Bottom', 'Inverse Head and Shoulders', 'Bull Flag'):
            breakdown['trend'] = 1.0 if latest_ma_short > latest_ma_long else 0.0
        elif p.name in ('Double Top', 'Head and Shoulders', 'Bear Flag'):
            breakdown['trend'] = 1.0 if latest_ma_short < latest_ma_long else 0.0
        else:
            breakdown['trend'] = 0.5
    else:
        breakdown['trend'] = 0.5

    rr = 0.0
    if p.points.get('breakout_idx') is not None:
        bi = int(p.points.get('breakout_idx'))
        price_at = float(close.iloc[bi]) if bi < len(close) else float(close.iloc[-1])
    else:
        price_at = float(close.iloc[-1])

    target = p.points.get('target') or p.points.get('target')
    stop = p.points.get('stop_loss') or p.points.get('stop_loss')
    atr_last = float(atr.iloc[-1]) if (isinstance(atr, pd.Series) and not atr.isna().all()) else (0.0)

    if target is not None and stop is not None:
        dist_target = abs(target - price_at)
        dist_stop = abs(price_at - stop)
        if dist_stop > 0:
            rr = float(min(3.0, dist_target / dist_stop))
        else:
            rr = 0.0
    elif atr_last and atr_last > 0:
        dist_stop = max(atr_last, 0.5 * abs(price_at*0.01))
        dist_target = max(atr_last*1.5, 0.01*abs(price_at))
        rr = float(min(3.0, dist_target / dist_stop))

    breakdown['risk_reward'] = float(min(1.0, rr / 3.0))

    weights = {
        'symmetry': 0.1,
        'touches': 0.1,
        'volume_surge': 0.2,  # must match key
        'obv_trend': 0.1,
        'vwap_position': 0.1,
        'vroc': 0.1,
        'ad_confirmation': 0.1,
        'trend': 0.1,
        'risk_reward': 0.1
    }

    raw_score = sum(breakdown[k] * weights[k] for k in breakdown)
    score = float(max(0.0, min(1.0, raw_score))) * 100.0

    if aggressiveness > 1.0 and breakdown['volume_surge'] > 0.5:
        score = min(100.0, score + (aggressiveness - 1.0) * 5.0)

    return score, breakdown


def suggested_position_size(price: float, stop_loss: float, portfolio_value: float, risk_pct: float = 0.01) -> Tuple[float, float]:
    if stop_loss is None or stop_loss <= 0 or price <= 0:
        return 0.0, 0.0
    risk_amount = portfolio_value * risk_pct
    per_share_risk = abs(price - stop_loss)
    if per_share_risk <= 0:
        return 0.0, 0.0
    shares = risk_amount / per_share_risk
    position_value = shares * price
    return float(position_value), float(shares)


def create_detected_patterns(patterns: List[Pattern], hist: pd.DataFrame, smart: bool = True, aggressiveness: float = 1.0, portfolio_value: float = 100000.0) -> Dict[str, Dict[str, Any]]:
    basic = {}
    for p in patterns:
        basic[p.name + f'_{len(basic)}'] = {
            'pattern': p,
            'points': p.points,
            'explanation': p.explanation,
            'ready': bool(p.points.get('breakout_idx'))
        }
    detected: Dict[str, Dict[str, Any]] = {}
    for key, entry in basic.items():
        p: Pattern = entry['pattern']
        d: Dict[str, Any] = {**entry}
        score, breakdown = compute_pattern_score(p, hist, aggressiveness=aggressiveness) if smart else (None, {})
        d['confidence_score'] = float(score) if score is not None else None
        d['score_breakdown'] = breakdown
        stop = p.points.get('stop_loss') or p.points.get('stop')
        target = p.points.get('target') or p.points.get('target')
        atr_series = compute_atr(hist) if 'High' in hist.columns and 'Low' in hist.columns else pd.Series(dtype=float)
        price_now = float(hist['Close'].iloc[-1])
        if stop is None and not atr_series.empty:
            atr_last = float(atr_series.iloc[-1])
            stop = float(price_now - (0.8 * atr_last)) if 'Bull' in p.name or 'Bottom' in p.name or 'Inverse' in p.name else float(price_now + (0.8 * atr_last))
        if target is None and not atr_series.empty:
            atr_last = float(atr_series.iloc[-1])
            target = float(price_now + (2.0 * atr_last)) if 'Bull' in p.name or 'Bottom' in p.name or 'Inverse' in p.name else float(price_now - (2.0 * atr_last))
        d['stop_loss'] = float(stop) if stop is not None else None
        d['target'] = float(target) if target is not None else None
        if smart and d['stop_loss'] is not None and d['confidence_score'] is not None:
            base_risk = 0.01 * (aggressiveness)
            risk_frac = base_risk * (d['confidence_score'] / 100.0)
            pos_val, shares = suggested_position_size(price_now, d['stop_loss'], portfolio_value, risk_frac)
            d['suggested_position_value'] = round(pos_val, 2)
            d['suggested_shares'] = max(0.0, float(shares))
            d['risk_fraction'] = float(risk_frac)
        else:
            d['suggested_position_value'] = None
            d['suggested_shares'] = None
            d['risk_fraction'] = None
        detected[key] = d
    return detected

def extract_features_from_pattern(p: Pattern, hist: pd.DataFrame) -> Dict[str, float]:
    """Create a small, meaningful feature vector for a detected Pattern.
    Features are intentionally simple and fast to compute.
    """
    close = hist['Close'].reset_index(drop=True)
    features: Dict[str, float] = {}
    features['pattern_type'] = hash(p.name) % 1000  # coarse numeric encoding (we'll use sklearn's Pipeline or user-supplied encoding)

    # geometry / symmetry
    if p.name in ('Head and Shoulders', 'Inverse Head and Shoulders'):
        peaks = p.points.get('peaks_price') or p.points.get('troughs_price')
        if peaks and len(peaks) == 3:
            left, mid, right = peaks
            features['shoulder_symmetry'] = 1.0 - min(1.0, _pct(left, right))
            features['head_vs_shoulders'] = abs(mid - np.mean([left, right])) / max(1e-9, np.mean([left, right]))
        else:
            features['shoulder_symmetry'] = 0.0
            features['head_vs_shoulders'] = 0.0
    else:
        features['shoulder_symmetry'] = 0.0
        features['head_vs_shoulders'] = 0.0

    # touches / neckline
    lvl = None
    if 'neckline_price' in p.points and p.points.get('neckline_price'):
        lvl = float(np.mean(p.points['neckline_price']))
    elif 'neckline' in p.points and p.points.get('neckline') is not None:
        lvl_idx = int(p.points['neckline'])
        lvl = float(close.iloc[lvl_idx])
    elif 'resistance' in p.points:
        lvl = float(p.points['resistance'])
    features['lvl'] = float(lvl) if lvl is not None else 0.0
    if lvl is not None:
        features['touches'] = float(np.sum(np.isclose(close.values, lvl, rtol=0.01)))
    else:
        features['touches'] = 0.0

    # volume around breakout
    brk = p.points.get('breakout_idx')
    if brk is not None and 'Volume' in hist.columns and isinstance(brk, int) and brk < len(hist):
        vol = hist['Volume'].reset_index(drop=True)
        baseline = float(vol[max(0, brk-20):brk].median()) if brk>0 else float(vol.median())
        features['breakout_vol_mult'] = float(vol.iloc[brk]) / max(1.0, baseline) if baseline>0 else 1.0
    else:
        features['breakout_vol_mult'] = 1.0

    # ATR-normalized ranges and MA slope
    if 'High' in hist.columns and 'Low' in hist.columns:
        atr = compute_atr(hist).reset_index(drop=True)
        features['atr_last'] = float(atr.iloc[-1]) if not atr.isna().all() else 0.0
    else:
        features['atr_last'] = 0.0

    # distance to current price
    price_now = float(close.iloc[-1])
    features['dist_to_now'] = abs(price_now - (lvl if lvl is not None else price_now)) / max(1e-9, price_now)

    # trend alignment
    ma_short = hist['Close'].rolling(50, min_periods=10).mean()
    ma_long = hist['Close'].rolling(200, min_periods=10).mean()

    if len(ma_short.dropna()) and len(ma_long.dropna()):
        features['trend_alignment'] = 1.0 if ma_short.iloc[-1] > ma_long.iloc[-1] else 0.0
    else:
        features['trend_alignment'] = 0.5

    # pattern length / window
    if 'window' in p.points and isinstance(p.points['window'], (list, tuple)):
        s, e = p.points['window']
        features['pattern_length'] = float(max(1, e - s))
    else:
        features['pattern_length'] = 0.0

    # ensure deterministic ordering
    return features


def build_dataset_from_patterns(patterns: List[Pattern], hist: pd.DataFrame, labels: Optional[List[int]] = None) -> Tuple[pd.DataFrame, Optional[List[int]]]:
    rows = []
    for p in patterns:
        f = extract_features_from_pattern(p, hist)
        f['pattern_name'] = p.name
        rows.append(f)
    df = pd.DataFrame(rows)
    return df, (labels if labels is not None else None)

def auto_label_patterns_by_outcome(patterns: List[Pattern], hist: pd.DataFrame, outcome_window_days: int = 30, threshold_pct: float = 0.03) -> List[int]:
    """Auto-generate binary labels by checking price movement after a breakout_idx (or pattern end).
    If price moves in the expected direction by >= threshold_pct within outcome_window_days -> label=1 else 0.
    This is a weak-labeling heuristic and intended to bootstrap training only.
    """
    labels: List[int] = []
    closes = hist['Close']
    n = len(closes)

    for p in patterns:
        # default label
        lbl = 0

        # infer expected direction for common patterns
        direction = None
        if p.name in ('Head and Shoulders', 'Double Top', 'Bear Flag'):
            direction = 'down'
        elif p.name in ('Inverse Head and Shoulders', 'Double Bottom', 'Bull Flag'):
            direction = 'up'
        else:
            # for triangles/other try to infer direction from breakout_idx vs neckline if available
            try:
                brk = p.points.get('breakout_idx')
                if brk is not None and isinstance(brk, int) and brk < n:
                    # compare breakout price to neckline/resistance/support if present
                    val = None
                    if 'neckline_price' in p.points and p.points.get('neckline_price'):
                        val = float(np.mean(p.points['neckline_price']))
                    elif 'neckline' in p.points and p.points.get('neckline') is not None:
                        ni = int(p.points['neckline'])
                        if 0 <= ni < n:
                            val = float(closes.iloc[ni])
                    if val is not None:
                        brk_price = float(closes.iloc[brk])
                        direction = 'up' if brk_price > val else 'down' if brk_price < val else None
            except Exception:
                direction = None

        # choose index to check: breakout_idx if present, else pattern end / last point we can infer
        idx: Optional[int] = None
        if p.points.get('breakout_idx') is not None and isinstance(p.points.get('breakout_idx'), int):
            idx = int(p.points.get('breakout_idx'))
        elif 'window' in p.points and isinstance(p.points['window'], (list, tuple)):
            # window stores relative indexes into the series used during detection
            s, e = p.points['window']
            idx = int(min(e, n - 1))
        elif 'peaks_idx' in p.points and isinstance(p.points['peaks_idx'], (list, tuple)):
            idx = int(min(max(p.points['peaks_idx']), n - 1))
        elif 'troughs_idx' in p.points and isinstance(p.points['troughs_idx'], (list, tuple)):
            idx = int(min(max(p.points['troughs_idx']), n - 1))
        else:
            idx = n - 1

        # safety bounds
        if idx is None or idx < 0 or idx >= n:
            labels.append(0)
            continue

        start_price = float(closes.iloc[idx])
        # find the future index corresponding to the outcome window
        future_dt = hist.index[idx] + timedelta(days=outcome_window_days)
        fidx = closes.index.searchsorted(future_dt)
        if fidx >= n:
            # not enough future data -> weak label 0
            labels.append(0)
            continue
        future_price = float(closes.iloc[fidx])
        move = (future_price - start_price) / max(1e-9, start_price)

        if direction == 'up' and move >= threshold_pct:
            lbl = 1
        elif direction == 'down' and move <= -threshold_pct:
            lbl = 1
        else:
            # if direction unknown, use absolute move magnitude as proxy
            if direction is None and abs(move) >= threshold_pct:
                lbl = 1
            else:
                lbl = 0

        labels.append(int(lbl))

    return labels


def train_classifier_from_dataframe(X: pd.DataFrame, y: List[int], n_estimators: int = 100, test_size: float = 0.2, random_state: int = 42) -> Tuple[Any, Dict[str, Any]]:
    if _SKLEARN_IMPORT_ERROR is not None:
        raise ImportError(f"scikit-learn import failed: {_SKLEARN_IMPORT_ERROR}. Install via `pip install scikit-learn`.")
    # trivial preprocessing: one-hot encode pattern_name and scale numeric features
    X_proc = X.copy()
    # convert pattern_name into categorical using simple get_dummies (fast and deterministic)
    if 'pattern_name' in X_proc.columns:
        dummies = pd.get_dummies(X_proc['pattern_name'], prefix='ptype')
        X_proc = pd.concat([X_proc.drop(columns=['pattern_name']), dummies], axis=1)
    # fill NaNs
    X_proc = X_proc.fillna(0.0)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_proc.values, np.array(y), test_size=test_size, random_state=random_state, stratify=np.array(y) if len(set(y))>1 else None)

    # pipeline (scale -> RF)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1))
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, 'predict_proba') else None

    metrics = {
        'accuracy': float(accuracy_score(y_test, preds)),
        'confusion_matrix': confusion_matrix(y_test, preds).tolist(),
        'classification_report': classification_report(y_test, preds, output_dict=True),
        'roc_auc': float(roc_auc_score(y_test, probs)) if probs is not None and len(set(y_test)) > 1 else None
    } # <-- Ensure this brace is the final closing element for the dictionary.
    return pipeline, metrics


def predict_patterns_with_model(pipeline: Any, patterns: List[Pattern], hist: pd.DataFrame) -> List[Tuple[Pattern, float, int]]:
    # return list of (pattern, prob, pred)
    rows = []
    for p in patterns:
        f = extract_features_from_pattern(p, hist)
        f['pattern_name'] = p.name
        rows.append(f)
    Xdf = pd.DataFrame(rows).fillna(0.0)
    # apply same encoding used in training
    if 'pattern_name' in Xdf.columns:
        dummies = pd.get_dummies(Xdf['pattern_name'], prefix='ptype')
        Xdf = pd.concat([Xdf.drop(columns=['pattern_name']), dummies], axis=1)
    # ensure pipeline can accept columns order mismatch by using DataFrame.values (pipeline expects numeric array)
    Xvals = Xdf.values
    probs = pipeline.predict_proba(Xvals)[:, 1] if hasattr(pipeline, 'predict_proba') else pipeline.predict(Xvals)
    preds = (probs >= 0.5).astype(int) if hasattr(pipeline, 'predict_proba') else probs.astype(int)
    out = []
    for p, pr, pv in zip(patterns, probs, preds):
        out.append((p, float(pr), int(pv))) # <-- Ensure this parenthesis is closed correctly.
    return out

# -----------------------------
# Advanced Volume Indicators
# -----------------------------

def compute_obv(df: pd.DataFrame) -> pd.Series:
    """Calculates On-Balance Volume."""
    return (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Volume Weighted Average Price"""
    pv = (df['Close'] * df['Volume']).cumsum()
    v = df['Volume'].cumsum().replace(0, np.nan)
    return pv / v

def compute_vroc(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Volume Rate of Change"""
    return df['Volume'].pct_change(periods=period).fillna(0)

def compute_ad_line(df: pd.DataFrame) -> pd.Series:
    """Accumulation/Distribution Line"""
    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / \
          (df['High'] - df['Low']).replace(0, np.nan)
    ad = (clv.fillna(0) * df['Volume']).cumsum()
    return ad

# -----------------------------
# Streamlit UI
# -----------------------------

def main() -> None:
    st.title('📈 Smart Stock Market Trading Signals — with ML classifier')
    st.sidebar.header('Simulation Inputs')

    # NEW: Add mode selection at the top
    analysis_mode = st.sidebar.radio(
        "Analysis Mode",
        ["Single Stock Analysis", "Multi-Stock Scanner"],
        index=0
    )

    if analysis_mode == "Multi-Stock Scanner":
        # ===== MULTI-STOCK SCANNER MODE =====
        st.header("📊 Multi-Stock Pullback Signal Scanner") # Renamed header

        # --- FIX: Define the local fallback list here. ---
        LOCAL_FALLBACK_STOCKS = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']

        # We call the cached function here. If the cache is empty, it runs the scraper.
        try:
            top_stocks = get_top_100_stocks()
            default_stock_list = '\n'.join(top_stocks) if top_stocks else '\n'.join(LOCAL_FALLBACK_STOCKS)
            st.info(f"Loaded {len(top_stocks)} tickers from live scrape/cache. Edit the list below as needed.")
        except Exception:
            # Fallback to hardcoded list if scraping/validation fails
            default_stock_list = '\n'.join(LOCAL_FALLBACK_STOCKS)
            st.warning("Could not fetch the live Top 100 list. Using a default list.")

        # Allow user to customize the list
        stock_input = st.text_area(
            "Stocks to Analyze (one per line)",
            value=default_stock_list,
            height=150
        )

        stocks_to_analyze = [s.strip().upper() for s in stock_input.split('\n') if s.strip()]

        st.info(f"Will analyze {len(stocks_to_analyze)} stocks")

        # Scan parameters
        col1, col2 = st.columns(2)
        with col1:
            scan_start_date = st.date_input(
                'Scan Start Date',
                datetime.today() - timedelta(days=90),
                key='scan_start'
            )
        with col2:
            scan_end_date = st.date_input(
                'Scan End Date',
                datetime.today(),
                key='scan_end'
            )

        exchange = st.selectbox('Exchange', ['NSE (India)', 'BSE (India)'], key='scan_exchange')
        interval = st.selectbox('Interval', ['1d', '1h'], index=0, key='scan_interval')

        # NOTE: Old Liquidity Sweep parameters are irrelevant but kept for UI stability
        st.subheader("Signal Filters (Optimized for Pullback)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("Fast EMA: 10")
        with col2:
            st.markdown("Medium EMA: 20")
        with col3:
            st.markdown("Slow EMA: 50")


        # Run the scan
        if st.button('🚀 Run Multi-Stock Scan', type='primary'):
            progress_bar = st.progress(0)
            status_text = st.empty()

            results_container = st.container()

            # Add a warning about rate limits
            st.warning("⏱️ Analysis may take several minutes due to rate limiting. Please be patient...")

            successful = 0
            failed = 0

            for idx, ticker_symbol in enumerate(stocks_to_analyze):
                status_text.text(f"Processing {idx+1}/{len(stocks_to_analyze)}: {ticker_symbol}")
                progress_bar.progress((idx) / len(stocks_to_analyze))

                # Use the existing run_single_ticker_analysis function
                with results_container:
                    try:
                        run_single_ticker_analysis(
                            ticker_symbol=ticker_symbol,
                            start_date=scan_start_date,
                            end_date=scan_end_date,
                            interval=interval,
                            analysis_index=idx # <--- ADD THIS
                        )
                        successful += 1
                    except Exception as e:
                        st.error(f"Failed to analyze {ticker_symbol}: {e}")
                        failed += 1

                # CRITICAL: Longer delay between stocks to avoid rate limiting
                if idx < len(stocks_to_analyze) - 1:  # Don't delay after last stock
                    time.sleep(5)  # Increased from 2 to 5 seconds

            progress_bar.progress(1.0)
            status_text.text(f"✅ Scan complete! Success: {successful}, Failed: {failed}")

            st.success(f"Successfully analyzed {successful}/{len(stocks_to_analyze)} stocks")

            # Display the updated summary table
            st.markdown("---")
            show_summary_table()

    else:

        exchange = st.sidebar.selectbox('Select Exchange', ['NSE (India)', 'BSE (India)', 'NASDAQ/NYSE (US)'])
        user_ticker = st.sidebar.text_input('Enter Stock Symbol', 'MOTHERSON')
        ticker = get_ticker_symbol(user_ticker, exchange)
        st.write(f'Fetching data for: **{ticker}**')

        today = datetime.today()
        buy_date = st.sidebar.date_input('Buy Date', today - timedelta(days=30), min_value=datetime(1900, 1, 1), max_value=today)
        invest_amount = st.sidebar.number_input('Investment Amount (₹)', value=1000.0, min_value=10.0, step=10.0)
        start_date = st.sidebar.date_input('Data Start', today - timedelta(days=90), min_value=datetime(1900, 1, 1), max_value=today)
        end_date = st.sidebar.date_input('Data End', today, min_value=start_date, max_value=today)
        interval = st.sidebar.selectbox('Select Interval', ['1d', '1h'])

        st.sidebar.markdown('---')
        smart_mode = st.sidebar.checkbox('Enable Smart Mode (scoring & sizing)', value=True)
        aggressiveness = st.sidebar.slider('Aggressiveness (scales risk)', min_value=0.5, max_value=2.0, value=1.0, step=0.1)
        portfolio_value = st.sidebar.number_input('Portfolio Value for Sizing', value=100000.0, min_value=1000.0, step=1000.0)

        st.sidebar.markdown('---')
        st.sidebar.header('Chart Options') # New section for chart controls
        show_forecast = st.sidebar.checkbox('Show Price Trend Forecast', value=True)
        # Note: These parameters are primarily used by the old detectors, but are kept for the original ML features
        strict_mode = st.sidebar.checkbox('Enable Strict Mode for Pattern Sweeps', value=True)
        volume_multiplier = st.sidebar.slider('Volume Multiplier', 1.0, 5.0, value=3.0, step=0.1)
        atr_multiplier = st.sidebar.slider('ATR Multiplier', 1.0, 5.0, value=1.5, step=0.1)
        min_confidence = st.sidebar.slider('Minimum Confidence Threshold', 0.1, 1.0, value=0.5, step=0.05)

        st.sidebar.markdown('---')
        st.sidebar.header('ML Classifier')
        upload_labels = st.sidebar.file_uploader('Upload labeled CSV (optional). Columns: pattern_name,label (0/1) or provide full feature rows', type=['csv'])
        auto_label = st.sidebar.checkbox('Auto-generate labels from history (weak labels)', value=True)
        n_estimators = st.sidebar.number_input('RF n_estimators', value=100, min_value=10, step=10)

        if _SKLEARN_IMPORT_ERROR is not None:
            st.warning('scikit-learn is not available in this environment. ML features will be disabled. To enable install scikit-learn: `pip install scikit-learn`')

        if st.sidebar.button('Run Simulation / Analyze'):
            hist = fetch_data(ticker, start_date, end_date, interval)

            if hist.empty:
                st.error('No data returned. Check ticker or dates.')
                return

            if st.sidebar.button('🔍 Debug EMA Comparison'):
                hist_test = fetch_data(ticker, start_date, end_date, interval)

                if not hist_test.empty:
                    comparison_df = compare_ema_calculations(hist_test, ticker)

                    st.subheader("EMA Calculation Comparison")
                    st.dataframe(comparison_df)

                    # Plot comparison
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df['Close'], name='Close', line=dict(color='black', width=2)))
                    fig.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df['Pandas_adjust=False'], name='Pandas (WRONG)', line=dict(color='red', dash='dot')))
                    fig.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df['TradingView_Method'], name='TradingView (CORRECT)', line=dict(color='green')))
                    fig.update_layout(title='EMA Calculation Methods Comparison', hovermode='x unified')
                    st.plotly_chart(fig, use_container_width=True)

            # Apply the strategy
            hist = detect_ema_pullback_with_state(hist, target_profit_pct=10.0, stop_loss_pct=20.0)

            # Show signals
            st.subheader('🎯 Trading Signals')
            signals_df = get_active_signals(hist)
            if not signals_df.empty:
                st.dataframe(signals_df)
            else:
                st.info("No buy signals found in this period")

            # Backtest
            st.subheader('📊 Strategy Performance')
            backtest_results = backtest_strategy(hist, initial_capital=portfolio_value)

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Trades", backtest_results['total_trades'])
            col2.metric("Win Rate", f"{backtest_results['win_rate']:.1f}%")
            col3.metric("Total Return", f"{backtest_results['total_return_%']:.2f}%")

            col4, col5, col6 = st.columns(3)
            col4.metric("Winning Trades", backtest_results['winning_trades'])
            col5.metric("Losing Trades", backtest_results['losing_trades'])
            col6.metric("Final Capital", f"₹{backtest_results['final_capital']:,.0f}")

            if backtest_results['total_trades'] > 0:
                col7, col8 = st.columns(2)
                col7.metric("Avg Win", f"₹{backtest_results['avg_win']:,.2f}")
                col8.metric("Avg Loss", f"₹{backtest_results['avg_loss']:,.2f}")

            # Plot
            st.subheader('📈 Strategy Visualization')
            fig = plot_strategy_chart(hist, ticker)
            st.plotly_chart(fig, use_container_width=True)

            # ADD THIS LINE TO CREATE A PRISTINE COPY FOR THE FORECAST
            hist_for_forecast = hist.copy()

            if hist.empty:
                st.error('No data returned. Check ticker or dates.')
                return

            MIN_DATA_POINTS = 60
            if len(hist) < MIN_DATA_POINTS:
                st.warning(f"⚠️ Not enough data for forecast. At least {MIN_DATA_POINTS} data points are required, but only {len(hist)} are available.")
                can_forecast = False
            else:
                can_forecast = True


            hist = add_features(hist)
            hist['ATR'] = compute_atr(hist)
            #hist = detect_ema_pullback_with_state(hist, target_profit_pct=10.0)

            # ===== NEWS SECTION =====
            st.markdown('---')
            st.subheader('📰 Latest News & Sentiment Analysis')

            # Optional: Add NewsAPI key input in sidebar
            news_api_key = st.sidebar.text_input(
                'NewsAPI Key (optional)', 
                type='password',
                help='Get free key from https://newsapi.org - Leave empty to use Google News scraper'
            )

            display_news_section(ticker, news_api_key if news_api_key else None)
            st.markdown('---')

            # Compute Support/Resistance levels and inject into hist
            hist = compute_avg_support_resistance(hist, lookback=10, confirm_touches=2)

            features = ['Close', 'Volume', 'VWAP', 'OBV', 'Support_Level', 'Resistance_Level', 'MA_5', 'MA_20', 'Momentum', 'Volatility', 'Return']
            X = hist[features].fillna(0)
            y = hist['Target']

            # Ensure no NaNs in features, which can cause LGBM to fail
            X = X.replace([np.inf, -np.inf], 0)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test)

            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'verbosity': -1
            }

            # Use a list for callbacks
            callbacks = [lgb.early_stopping(stopping_rounds=10, verbose=False)]

            model = lgb.train(
                params,
                train_data,
                valid_sets=[valid_data],
                callbacks=callbacks
            )

            # Save the trained model
            model.save_model('lgb_model.txt')

            # Evaluate model accuracy
            y_test_pred_proba = model.predict(X_test)
            y_test_pred_class = (y_test_pred_proba > 0.5).astype(int)

            # st.metric("Model Accuracy", f"{accuracy_score(y_test, y_test_pred_class):.2%}")

            # Properly inject predicted probabilities into the correct slice of hist
            hist['Predicted_Prob_Up'] = np.nan  # Initialize column

            # Assign predictions only for test period
            test_start_idx = len(X_train)
            if test_start_idx < len(hist):
                hist.iloc[test_start_idx:, hist.columns.get_loc('Predicted_Prob_Up')] = y_test_pred_proba

            # normalize hist index to tz-naive early to avoid future tz issues in comparisons & plotting
            if hasattr(hist.index, 'tz') and hist.index.tz is not None:
                hist = hist.copy()
                hist.index = hist.index.tz_convert(None)

            # indicators
            if 'High' in hist.columns and 'Low' in hist.columns:
                atr_ser = compute_atr(hist)
                st.sidebar.metric('ATR (last)', f"{atr_ser.iloc[-1]:.2f}")
            rsi = compute_rsi(hist['Close'])
            st.sidebar.metric('RSI (last)', f"{rsi.iloc[-1]:.2f}")

            st.info('🔍 Scraping Insider Trading Data from NSE...')
            insider_df = scrape_nse_insider_trading_dynamic(
                ticker=user_ticker.split('.')[0],  # Example: RELIANCE from RELIANCE.NS
                start_date=start_date,
                end_date=end_date
            )

            if insider_df.empty:
                st.warning('No insider trading data found for this stock in the selected date range.')
            else:
                st.success(f'Fetched {len(insider_df)} insider transactions.')

                # Merge insider data into price data
                hist = hist.merge(insider_df, left_index=True, right_index=True, how='left')

                # Fill missing values
                hist[['Insider_Name', 'Buy_Sell', 'Shares', 'Reporting_Person_Type', 'Remarks']] = \
                    hist[['Insider_Name', 'Buy_Sell', 'Shares', 'Reporting_Person_Type', 'Remarks']].fillna('')

                # Generate signal feature
                threshold_insider = 10000  # Customize threshold if needed
                hist['Insider_Buy_Signal'] = (hist['Shares'].astype(int) > threshold_insider) & (hist['Buy_Sell'] == 'Buy')

            # returns
            st.subheader(f'📊 Price Return Simulation for {ticker}')
            try:
                price_result = compute_price_return(hist, buy_date, invest_amount)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Buy Price", f"₹{price_result['buy_price']:.2f}")
                c2.metric("Current Price", f"₹{price_result['current_price']:.2f}")
                c3.metric("Current Value", f"₹{price_result['value_today']:.2f}")
                c4.metric("Return", f"{price_result['return_pct']:.2f}%")

            except Exception as e:
                st.error(f'Price return computation failed: {e}')

            st.subheader('💰 Total Return (Dividend Reinvested)')
            try:
                total_result = compute_total_return(hist, buy_date, invest_amount)
                if total_result is not None:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Final Value", f"₹{total_result['final_value']:.2f}")
                    c2.metric("Abs. Return", f"₹{total_result['absolute_return']:.2f}")
                    c3.metric("CAGR", f"{total_result['cagr']:.2%}")
            except Exception as e:
                st.error(f'Total return computation failed: {e}')

            # S/R detection & plotting
            supports, resistances = find_support_resistance(hist, lookback=10, confirm_touches=2)

            # Enrich Data
            hist = enrich_with_trend_and_direction(hist)

            # Detect Liquidity Sweeps
            # hist = detect_liquidity_sweeps(hist, volume_multiplier=volume_multiplier, atr_multiplier=atr_multiplier)

            # The original ML features still rely on the old sweep detection, so we keep the calculation here
            hist = detect_liquidity_sweeps(hist, volume_multiplier=volume_multiplier, atr_multiplier=atr_multiplier)


            # Filter high-confidence sweeps
            high_conf_sweeps = detect_high_confidence_sweeps(hist, supports, resistances, strict_mode=strict_mode, min_confidence=min_confidence)

            # Annotate sweep types
            high_conf_sweeps = annotate_sweep_types(hist, high_conf_sweeps, lookahead=3)

            # Plot Candlestick + Sweeps (Now using the new function for the EMA Signal)
            st.subheader("📈 Candlestick with Pullback Buy Signals")
            plot_candlestick_with_signals(hist, ticker, 'Pullback_Signal', 0)

            # Plot Volume + OBV + Buy/Sell Sweeps (This still plots the old liquidity sweeps)
            st.subheader("📊 Volume & OBV with Liquidity Sweep Markers (Original)")
            plot_volume_obv_with_sweep_types(hist, high_conf_sweeps, ticker)

            # P
            fig_sr = go.Figure()
            fig_sr.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Close', line=dict(color='lightblue')))
            x_axis = list(hist.index)
            for i, (d, level) in enumerate(supports):
                fig_sr.add_trace(go.Scatter(x=x_axis, y=[level] * len(x_axis), mode='lines', name=f'Support {i+1}: {level:.2f}', line=dict(dash='dash', color='green')))
            for i, (d, level) in enumerate(resistances):
                fig_sr.add_trace(go.Scatter(x=x_axis, y=[level] * len(x_axis), mode='lines', name=f'Resistance {i+1}: {level:.2f}', line=dict(dash='dash', color='red')))
            fig_sr.update_layout(title=f'{ticker} with Support/Resistance', hovermode='x unified')

            # --- UPDATE AND DISPLAY THE PERMANENT SUMMARY TABLE ---
            # Use the new check_liquidity_sweep_for_summary function to get the status
            has_signal, latest_date = check_liquidity_sweep_for_summary(ticker)
            latest_status = 'Buy Signal' if has_signal else 'No Signal'

            # Update the permanent CSV file with the latest result
            update_summary_table(ticker, latest_status, latest_date)

            # Display the full, updated table in the app
            show_summary_table()

            col1, col2 = st.columns(2)
            with col1:
                st.subheader('🔍 Support Forward Test')
                if supports:
                    df_s, score_s = forward_test(hist, supports, test_type='support')
                    st.dataframe(df_s.head())
                    st.metric("Support Held %", f"{score_s.get('Held %', 0):.2f}%")

            with col2:
                st.subheader('🔍 Resistance Forward Test')
                if resistances:
                    df_r, score_r = forward_test(hist, resistances, test_type='resistance')
                    st.dataframe(df_r.head())
                    st.metric("Resistance Held %", f"{score_r.get('Held %', 0):.2f}%")

            # pattern detection & plotting
            st.subheader('📐 Pattern Detection (Flags, Triangles, H&S, Double Tops/Bottoms)')
            patterns = detect_all_patterns(hist)

            figp = go.Figure()
            figp.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Close', line=dict(color='lightblue')))

            # Insider Buy Signals
            if 'Insider_Buy_Signal' in hist.columns and hist['Insider_Buy_Signal'].any():
                figp.add_trace(go.Scatter(
                    x=hist.index[hist['Insider_Buy_Signal']],
                    y=hist['Close'][hist['Insider_Buy_Signal']],
                    mode='markers',
                    marker=dict(symbol='triangle-up', color='green', size=10),
                    name='Insider Buy Signal'
                ))

            def to_date(i: int) -> Optional[pd.Timestamp]:
                return hist.index[i] if 0 <= i < len(hist.index) else None

            for p in patterns:
                if p.name in ('Head and Shoulders', 'Inverse Head and Shoulders'):
                    peaks_idx = p.points.get('peaks_idx') or p.points.get('troughs_idx')
                    nl_idx = p.points.get('neckline_idx')
                    nl_prices = p.points.get('neckline_price')
                    if peaks_idx:
                        xs = [to_date(i) for i in peaks_idx]
                        ys = [hist['Close'].iloc[i] for i in peaks_idx]
                        figp.add_trace(go.Scatter(x=xs, y=ys, mode='markers+lines', name=f'{p.name} peaks'))
                    if nl_idx and nl_prices:
                        xs = [to_date(nl_idx[0]), to_date(nl_idx[1])]
                        ys = nl_prices
                        figp.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name=f'{p.name} neckline', line=dict(dash='dot')))
                elif 'Triangle' in p.name:
                    win = p.points.get('window')
                    if win:
                        start, end = win
                        xidx = list(range(start, end + 1))
                        xs = [to_date(i) for i in xidx]
                        up_slope, up_int = p.points.get('upper_line', (0.0, 0.0))
                        lo_slope, lo_int = p.points.get('lower_line', (0.0, 0.0))
                        ys_up = _line_values_for_indexes(xidx, up_slope, up_int)
                        ys_lo = _line_values_for_indexes(xidx, lo_slope, lo_int)
                        figp.add_trace(go.Scatter(x=xs, y=ys_up, mode='lines', name=f'{p.name} upper', line=dict(dash='dash')))
                        figp.add_trace(go.Scatter(x=xs, y=ys_lo, mode='lines', name=f'{p.name} lower', line=dict(dash='dash')))
                elif 'Flag' in p.name:
                    imp = p.points.get('impulse')
                    cons = p.points.get('consolidation')
                    if imp:
                        a, b = imp
                        xs = [to_date(a), to_date(b)]
                        ys = [hist['Close'].iloc[a], hist['Close'].iloc[b]]
                        figp.add_trace(go.Scatter(x=xs, y=ys, mode='lines+markers', name=f'{p.name} impulse'))
                    if cons and p.points.get('channel_slope') is not None:
                        c, d = cons
                        xidx = list(range(c, d + 1))
                        xs = [to_date(i) for i in xidx]
                        ych = _line_values_for_indexes(xidx, p.points['channel_slope'], p.points['channel_intercept'])
                        figp.add_trace(go.Scatter(x=xs, y=ych, mode='lines', name=f'{p.name} channel', line=dict(dash='dot')))
                elif p.name == 'Double Top':
                    peaks = p.points.get('peaks')
                    nline = p.points.get('neckline')
                    if peaks:
                        xs = [to_date(i) for i in peaks]
                        ys = [hist['Close'].iloc[i] for i in peaks]
                        figp.add_trace(go.Scatter(x=xs, y=ys, mode='markers+lines', name='Double Top peaks'))
                    if nline is not None:
                        x_all = list(hist.index)
                        figp.add_trace(go.Scatter(x=x_all, y=[hist['Close'].iloc[nline]]*len(x_all), mode='lines', name='Double Top neckline', line=dict(dash='dot')))
                elif p.name == 'Double Bottom':
                    troughs = p.points.get('troughs')
                    nline = p.points.get('neckline')
                    if troughs:
                        xs = [to_date(i) for i in troughs]
                        ys = [hist['Close'].iloc[i] for i in troughs]
                        figp.add_trace(go.Scatter(x=xs, y=ys, mode='markers+lines', name='Double Bottom troughs'))
                    if nline is not None:
                        x_all = list(hist.index)
                        figp.add_trace(go.Scatter(x=x_all, y=[hist['Close'].iloc[nline]]*len(x_all), mode='lines', name='Double Bottom neckline', line=dict(dash='dot')))

            figp.update_layout(title=f'{ticker} — Detected Chart Patterns', xaxis_title='Date', yaxis_title='Price', hovermode='x unified', height=500)
            st.plotly_chart(figp, use_container_width=True)

            # The corrected section for the score_breakdown (around line 1890 in the context of the larger app.py):

# ... (inside the loop that defines the table_rows)

            # show detected patterns details (score & sizing)
            detected_patterns = create_detected_patterns(patterns, hist, smart=smart_mode, aggressiveness=aggressiveness, portfolio_value=portfolio_value)
            if detected_patterns:
                table_rows = []
                for k, v in detected_patterns.items():
                    row = {
                        'Pattern': v['pattern'].name,
                        'Confidence': v.get('confidence_score'),
                        'Target': v.get('target'),
                        'Stop Loss': v.get('stop_loss'),
                        'Suggested Pos. (₹)': v.get('suggested_position_value'),
                    } # <--- This closing brace '}' was likely followed by a mismatched ')' or similar error in the original file
                    table_rows.append(row)
                # ... rest of the code is fine
                df_table = pd.DataFrame(table_rows).sort_values(by='Confidence', ascending=False)
                st.dataframe(df_table.style.format({
                    'Confidence': '{:.1f}',
                    'Target': '{:.2f}',
                    'Stop Loss': '{:.2f}',
                    'Suggested Pos. (₹)': '{:,.2f}'
                }))

            # ML: prepare dataset (either uploaded labels or auto-label)
            st.subheader('🧠 ML Pattern Classifier')
            model = None
            model_metrics = None
            training_df = None
            training_labels = None

            if upload_labels is not None:
                try:
                    uploaded = pd.read_csv(upload_labels)
                    st.write('Uploaded labels:', uploaded.head())
                    if {'label', 'pattern_name'}.issubset(set(uploaded.columns)):
                        pattern_df, _ = build_dataset_from_patterns(patterns, hist)
                        merged = pattern_df.merge(uploaded[['pattern_name', 'label']], left_on='pattern_name', right_on='pattern_name', how='left')
                        training_df = merged.drop(columns=['label'])
                        training_labels = merged['label'].fillna(0).astype(int).tolist()
                    else:
                        st.error('Uploaded CSV must contain at minimum columns: pattern_name,label OR full pre-computed features matching our extractor.')
                except Exception as e:
                    st.error(f'Failed to parse uploaded CSV: {e}')

            if training_df is None and auto_label:
                with st.spinner('Auto-labeling patterns using forward outcome heuristic...'):
                    labels = auto_label_patterns_by_outcome(patterns, hist, outcome_window_days=30, threshold_pct=0.03)
                    df_features, _ = build_dataset_from_patterns(patterns, hist)
                    training_df = df_features
                    training_labels = labels
                    st.write('Auto-labeled dataset size:', len(training_df), 'positive labels:', sum(training_labels))

            if training_df is not None and training_labels is not None and len(training_df) > 10:
                if _SKLEARN_IMPORT_ERROR is not None:
                    st.error('scikit-learn not available; install scikit-learn to train the model.')
                else:
                    if st.button('Train Classifier'):
                        with st.spinner('Training classifier — this may take a while...'):
                            try:
                                pipeline, metrics = train_classifier_from_dataframe(training_df, training_labels, n_estimators=int(n_estimators))
                                model = pipeline
                                model_metrics = metrics
                                st.success('Training complete')
                                st.write('Metrics:', metrics)
                                model_bytes = pickle.dumps(pipeline)
                                st.download_button('Download trained model (pickle)', data=model_bytes, file_name='pattern_classifier.pkl')
                            except Exception as e:
                                st.error(f'Training failed: {e}')

            # If a model exists (either trained here), annotate detected patterns with predictions
            if model is not None:
                with st.spinner('Applying model to detected patterns...'):
                    preds = predict_patterns_with_model(model, patterns, hist)
                    rows = []
                    for p, prob, pred in preds:
                        rows.append({'pattern': p.name, 'probability': prob, 'prediction': pred, 'explanation': p.explanation})
                    st.table(pd.DataFrame(rows))

            # --- Main Price Chart with ML Predictions and Forecast ---
            st.header("Price, ML Prediction, and Forecast")

            # NEW: Generate Forecast
            forecast_series = None
            if show_forecast:
                st.info("The forecast shows a simple linear trend projected from the last 60 periods of data. It is for illustrative purposes only.", icon="💡")
                forecast_periods = 30 # For both daily and hourly, project 30 periods forward
                with st.spinner('Generating price forecast...'):
                    forecast_series = generate_forecast(hist_for_forecast, forecast_periods=forecast_periods)

            fig_main = go.Figure()

            # Price Line (primary Y-axis)
            fig_main.add_trace(go.Scatter(
                x=hist.index,
                y=hist["Close"],
                mode="lines",
                name="Close Price",
                yaxis="y"
            ))

            # ML Predicted Probability (secondary Y-axis)
            fig_main.add_trace(go.Scatter(
                x=hist.index,
                y=hist["Predicted_Prob_Up"],
                mode="lines",
                name="Uptrend Probability",
                line=dict(color="orange", dash="dot"),
                yaxis="y2"
            ))

            # NEW: Add Forecast Trace
            if forecast_series is not None:
                fig_main.add_trace(go.Scatter(
                    x=forecast_series.index,
                    y=forecast_series.values,
                    mode="lines",
                    name="Forecast Trend",
                    line=dict(color="#FF4B4B", dash="dash"), # A distinct red color
                    yaxis="y" # <--- THIS IS THE FIX
                ))

            fig_main.update_layout(
                title=f"{ticker} Price, ML Predicted Uptrend Probability, and Forecast",
                xaxis_title="Date",
                yaxis=dict(title="Price"),
                yaxis2=dict(
                    title="Up Probability",
                    overlaying="y",
                    side="right",
                    range=[0, 1]
                ),
                hovermode="x unified"
            )

            st.plotly_chart(fig_main, use_container_width=True)

        # --- Sidebar for Full Summary Refresh ---
        st.sidebar.markdown("---")
        st.sidebar.header("Summary Table Management")

        if st.sidebar.button("🔄 Refresh Full Summary (Max 1/hour)"):
            can_run, last_run_time = False, None
            if os.path.exists(TIMESTAMP_FILE):
                with open(TIMESTAMP_FILE, 'r') as f:
                    try:
                        last_run_timestamp = float(f.read())
                        last_run_time = datetime.fromtimestamp(last_run_timestamp)
                        if (datetime.now() - last_run_time).total_seconds() > 3600: # 1 hour
                            can_run = True
                    except (ValueError, FileNotFoundError):
                        can_run = True
            else:
                can_run = True

            if not can_run and last_run_time:
                next_run_time = last_run_time + timedelta(hours=1)
                st.warning(f"Full refresh is available once per hour. Please try again after {next_run_time.strftime('%H:%M:%S')}.")
            else:
                if os.path.exists(SUMMARY_FILE):
                    summary_df = pd.read_csv(SUMMARY_FILE)
                    tickers_to_update = summary_df['Ticker'].tolist()

                    if tickers_to_update:
                        st.info(f"Updating {len(tickers_to_update)} stocks by running a full analysis on each...")

                        # Loop through tickers and call the SAME function as the manual button
                        for ticker in tickers_to_update:
                            # We use daily ('1d') data for the summary refresh to be faster and more consistent
                            run_single_ticker_analysis(ticker, datetime.today() - timedelta(days=90), datetime.today(), '1d')
                            time.sleep(2) # Keep the polite delay between each stock

                        with open(TIMESTAMP_FILE, 'w') as f:
                            f.write(str(datetime.now().timestamp()))

                        st.success("✅ Full summary refresh complete!")
                    else:
                        st.warning("Summary table is empty.")
                else:
                    st.error("No summary file exists. Please analyze a stock or scan the Top 100 first.")

        # --- Main Page Display ---
        st.header("Top 100 Stocks Screener")
        if st.button('Scan Top 100 Stocks & Add to Summary'):
            with st.spinner("Scraping and validating tickers... This may take several minutes."):
                top_stocks = get_top_100_stocks()
                if top_stocks:
                    st.info("Adding/updating Top 100 stocks in the summary table...")
                    for ticker in top_stocks:
                        update_summary_table(ticker, 'Not yet checked', 'N/A')
                    st.success("Top 100 list updated. Click 'Refresh Full Summary' to run the analysis.")

        # This will always show the latest state of your summary table
        show_summary_table()

if __name__ == '__main__':
    main()
