# Stock Analyzer - Agentic AI Stock Screener

Multi-agent AI system for analyzing stocks and generating investment insights.

## Features
- Natural language query understanding
- Real-time market data aggregation
- Technical + fundamental + sentiment analysis
- Sector ETF benchmarking
- Explainable recommendations

## Architecture
[User → Entity Extraction → Data Gathering → Analysis → LLM → Output]

## Tech Stack
- LangGraph (multi-agent orchestration)
- Ollama (local LLM inference - Qwen 2.5 14B)
- ChromaDB (vector storage)
- SentenceTransformers (embeddings)
- yfinance, NewsAPI (data sources)
- Newspaper3k (data extraction)
- Streamlit (frontend)

## Example Queries
- "Should I buy Microsoft?"
- "Compare AAPL to tech sector"
- "Analysis of small cap tech stocks"

## Status
🚧 Active development

