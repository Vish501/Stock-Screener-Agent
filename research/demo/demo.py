from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from demo_db import StockVectorDB

# Initialize components
db_loc = "artifacts/embeddings/test_db"
db_name = "Test"

vector_db = StockVectorDB(persist_directory=db_loc, name=db_name)
llm = ChatOllama(model="qwen2.5:14b", temperature=0.1)

# Define the state
class StockAnalysisState(TypedDict):
    user_input: str
    identified_stocks: List[Dict[str, str]]
    fundamental_data: Dict
    sentiment_data: Dict
    technical_data: Dict
    comparison_data: Dict
    scores: Dict
    recommendations: List[Dict]
    explanation: str


def identify_stocks(state: StockAnalysisState) -> StockAnalysisState:
    """
    Node 1: Identify stocks from user input using vector search
    """
    user_input = state['user_input']
    
    # Use vector DB to find relevant stocks
    stocks = vector_db.search_stocks(
        query=user_input,
        n_results=10,
        min_similarity=0.4
    )
    
    print(f"Identified {len(stocks)} stocks from input")
    
    state['identified_stocks'] = stocks
    return state


def fetch_fundamental_data(state: StockAnalysisState) -> StockAnalysisState:
    """
    Node 2: Fetch fundamental data from yfinance
    """
    fundamental_data = {}
    
    for stock in state['identified_stocks']:
        ticker = stock['ticker']
        
        try:
            yf_ticker = yf.Ticker(ticker)
            info = yf_ticker.info
            
            fundamental_data[ticker] = {
                'pe_ratio': info.get('trailingPE', None),
                'pb_ratio': info.get('priceToBook', None),
                'ps_ratio': info.get('priceToSalesTrailing12Months', None),
                'roe': info.get('returnOnEquity', None),
                'eps': info.get('trailingEps', None),
                'book_value': info.get('bookValue', None),
                'ev_ebitda': info.get('enterpriseToEbitda', None),
                'market_cap': info.get('marketCap', None),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown')
            }
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            fundamental_data[ticker] = None
    
    state['fundamental_data'] = fundamental_data
    return state


def fetch_sentiment_data(state: StockAnalysisState) -> StockAnalysisState:
    """
    Node 3: Fetch sentiment from NewsAPI
    """
    # Placeholder - integrate your NewsAPI logic here
    sentiment_data = {}
    
    for stock in state['identified_stocks']:
        ticker = stock['ticker']
        
        # TODO: Implement NewsAPI integration
        sentiment_data[ticker] = {
            'sentiment_score': 0.5,  # -1 to 1
            'article_count': 10,
            'positive_articles': 6,
            'negative_articles': 2,
            'neutral_articles': 2
        }
    
    state['sentiment_data'] = sentiment_data
    return state


def fetch_technical_data(state: StockAnalysisState) -> StockAnalysisState:
    """
    Node 4: Fetch technical indicators and volatility (1 year data)
    """
    technical_data = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    for stock in state['identified_stocks']:
        ticker = stock['ticker']
        
        try:
            # Fetch historical data
            hist = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if len(hist) > 0:
                # Calculate technical indicators
                hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
                hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
                hist['Daily_Return'] = hist['Close'].pct_change()
                
                technical_data[ticker] = {
                    'current_price': float(hist['Close'].iloc[-1]),
                    'sma_50': float(hist['SMA_50'].iloc[-1]) if not pd.isna(hist['SMA_50'].iloc[-1]) else None,
                    'sma_200': float(hist['SMA_200'].iloc[-1]) if not pd.isna(hist['SMA_200'].iloc[-1]) else None,
                    'volatility': float(hist['Daily_Return'].std() * np.sqrt(252)),  # Annualized
                    'year_high': float(hist['High'].max()),
                    'year_low': float(hist['Low'].min()),
                    'ytd_return': float((hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100)
                }
            
        except Exception as e:
            print(f"Error fetching technical data for {ticker}: {e}")
            technical_data[ticker] = None
    
    state['technical_data'] = technical_data
    return state


def compare_with_indices(state: StockAnalysisState) -> StockAnalysisState:
    """
    Node 5: Compare with S&P 500 and sector indices
    """
    comparison_data = {}
    
    # Fetch S&P 500 data
    sp500 = yf.Ticker("SPY")
    sp500_pe = sp500.info.get('trailingPE', None)
    
    for stock in state['identified_stocks']:
        ticker = stock['ticker']
        fund_data = state['fundamental_data'].get(ticker, {})
        
        if fund_data:
            sector = fund_data.get('sector', 'Unknown')
            
            # Map sector to ETF
            sector_etf_map = {
                'Technology': 'XLK',
                'Healthcare': 'XLV',
                'Financials': 'XLF',
                'Energy': 'XLE',
                'Consumer Cyclical': 'XLY',
                'Consumer Defensive': 'XLP',
                'Industrials': 'XLI',
                'Utilities': 'XLU',
                'Real Estate': 'XLRE',
                'Materials': 'XLB',
                'Communication Services': 'XLC'
            }
            
            sector_etf = sector_etf_map.get(sector, 'SPY')
            
            try:
                sector_ticker = yf.Ticker(sector_etf)
                sector_pe = sector_ticker.info.get('trailingPE', None)
                
                comparison_data[ticker] = {
                    'sector_etf': sector_etf,
                    'sector_pe': sector_pe,
                    'sp500_pe': sp500_pe,
                    'pe_vs_sector': fund_data.get('pe_ratio', 0) / sector_pe if sector_pe else None,
                    'pe_vs_sp500': fund_data.get('pe_ratio', 0) / sp500_pe if sp500_pe else None
                }
                
            except Exception as e:
                print(f"Error comparing {ticker}: {e}")
                comparison_data[ticker] = None
    
    state['comparison_data'] = comparison_data
    return state


def score_stocks(state: StockAnalysisState) -> StockAnalysisState:
    """
    Node 6: Feature engineering and scoring
    """
    scores = {}
    
    for stock in state['identified_stocks']:
        ticker = stock['ticker']
        
        fund = state['fundamental_data'].get(ticker, {})
        sent = state['sentiment_data'].get(ticker, {})
        tech = state['technical_data'].get(ticker, {})
        comp = state['comparison_data'].get(ticker, {})
        
        if not (fund and tech):
            scores[ticker] = None
            continue
        
        # Scoring logic (normalize to 0-100)
        score_components = {
            'valuation_score': 0,
            'growth_score': 0,
            'sentiment_score': 0,
            'technical_score': 0,
            'comparison_score': 0
        }
        
        # Valuation (lower is better for PE, PB)
        if fund.get('pe_ratio'):
            score_components['valuation_score'] += 20 if fund['pe_ratio'] < 20 else 10
        if fund.get('pb_ratio'):
            score_components['valuation_score'] += 20 if fund['pb_ratio'] < 3 else 10
        
        # Growth
        if fund.get('roe'):
            score_components['growth_score'] += 30 if fund['roe'] > 0.15 else 15
        if tech.get('ytd_return'):
            score_components['growth_score'] += 20 if tech['ytd_return'] > 0 else 5
        
        # Sentiment
        if sent:
            score_components['sentiment_score'] = sent['sentiment_score'] * 50 + 50
        
        # Technical
        if tech.get('sma_50') and tech.get('current_price'):
            score_components['technical_score'] += 30 if tech['current_price'] > tech['sma_50'] else 10
        if tech.get('volatility'):
            score_components['technical_score'] += 20 if tech['volatility'] < 0.3 else 10
        
        # Comparison
        if comp and comp.get('pe_vs_sector'):
            score_components['comparison_score'] += 30 if comp['pe_vs_sector'] < 1 else 15
        
        # Calculate total score
        total_score = sum(score_components.values()) / 2  # Normalize to 100
        
        scores[ticker] = {
            'total_score': round(total_score, 2),
            'components': score_components
        }
    
    state['scores'] = scores
    return state


def generate_recommendations(state: StockAnalysisState) -> StockAnalysisState:
    """
    Node 7: Use LLM to generate BUY/HOLD/SELL recommendations
    """
    recommendations = []
    
    for stock in state['identified_stocks']:
        ticker = stock['ticker']
        company_name = stock['company_name']
        
        # Prepare data for LLM
        fund = state['fundamental_data'].get(ticker, {})
        tech = state['technical_data'].get(ticker, {})
        sent = state['sentiment_data'].get(ticker, {})
        comp = state['comparison_data'].get(ticker, {})
        score = state['scores'].get(ticker, {})
        
        # Create prompt for LLM
        prompt = f"""
        Analyze the following stock and provide a recommendation (BUY/HOLD/SELL):
        
        Company: {company_name} ({ticker})
        
        Fundamentals:
        - P/E Ratio: {fund.get('pe_ratio', 'N/A')}
        - P/B Ratio: {fund.get('pb_ratio', 'N/A')}
        - ROE: {fund.get('roe', 'N/A')}
        - EPS: {fund.get('eps', 'N/A')}
        
        Technical:
        - Current Price: {tech.get('current_price', 'N/A')}
        - YTD Return: {tech.get('ytd_return', 'N/A')}%
        - Volatility: {tech.get('volatility', 'N/A')}
        
        Sentiment:
        - Sentiment Score: {sent.get('sentiment_score', 'N/A')}
        
        Comparison:
        - P/E vs Sector: {comp.get('pe_vs_sector', 'N/A')}
        - P/E vs S&P500: {comp.get('pe_vs_sp500', 'N/A')}
        
        Overall Score: {score.get('total_score', 'N/A')}/100
        
        Provide:
        1. Recommendation: BUY/HOLD/SELL
        2. Brief explanation (2-3 sentences)
        
        Format: RECOMMENDATION: [BUY/HOLD/SELL] | REASON: [explanation]
        """
        
        # Get LLM response
        response = llm.invoke(prompt)
        
        recommendations.append({
            'ticker': ticker,
            'company_name': company_name,
            'recommendation': response.content,
            'score': score.get('total_score', 0)
        })
    
    state['recommendations'] = recommendations
    return state


def format_output(state: StockAnalysisState) -> StockAnalysisState:
    """
    Node 8: Format output for Streamlit display
    """
    explanation = "# Stock Analysis Results\n\n"
    
    for rec in state['recommendations']:
        explanation += f"## {rec['company_name']} ({rec['ticker']})\n"
        explanation += f"**Score:** {rec['score']}/100\n\n"
        explanation += f"{rec['recommendation']}\n\n"
        explanation += "---\n\n"
    
    state['explanation'] = explanation
    return state


# Build the graph
def create_stock_analysis_graph():
    workflow = StateGraph(StockAnalysisState)
    
    # Add nodes
    workflow.add_node("identify_stocks", identify_stocks)
    workflow.add_node("fetch_fundamentals", fetch_fundamental_data)
    workflow.add_node("fetch_sentiment", fetch_sentiment_data)
    workflow.add_node("fetch_technical", fetch_technical_data)
    workflow.add_node("compare_indices", compare_with_indices)
    workflow.add_node("score_stocks", score_stocks)
    workflow.add_node("generate_recommendations", generate_recommendations)
    workflow.add_node("format_output", format_output)
    
    # Define edges
    workflow.set_entry_point("identify_stocks")
    workflow.add_edge("identify_stocks", "fetch_fundamentals")
    workflow.add_edge("fetch_fundamentals", "fetch_sentiment")
    workflow.add_edge("fetch_sentiment", "fetch_technical")
    workflow.add_edge("fetch_technical", "compare_indices")
    workflow.add_edge("compare_indices", "score_stocks")
    workflow.add_edge("score_stocks", "generate_recommendations")
    workflow.add_edge("generate_recommendations", "format_output")
    workflow.add_edge("format_output", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = create_stock_analysis_graph()
    
    # Test input
    result = graph.invoke({
        "user_input": "Analyze Apple and Microsoft",
        "identified_stocks": [],
        "fundamental_data": {},
        "sentiment_data": {},
        "technical_data": {},
        "comparison_data": {},
        "scores": {},
        "recommendations": [],
        "explanation": ""
    })
    
    print(result['explanation'])
