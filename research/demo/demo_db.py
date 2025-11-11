import chromadb
import numpy as np

from pathlib import Path
from tqdm import tqdm
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional


class StockVectorDB:
    def __init__(self, persist_directory: Path, name: str, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the vector database for stocks.
        
        Args:
            persist_directory: Path to persist the database
        """
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create custom embedding function with normalization
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name
            )
            
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=name,
                embedding_function=self.embedding_function
            )
        except:
            self.collection = self.client.create_collection(
                name=name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )


    def populate_db(self, stocks: Dict[str, str]) -> None:
        """
        Populate the vector database with stock data.
        
        Args:
            stocks: Dictionary of {"Company Name": "Ticker"}
        """
        documents = [f"{name} {ticker}" for name, ticker in stocks.items()]
        metadatas = [{"company_name": name, "ticker": ticker} for name, ticker in stocks.items()]
        ids = [ticker for ticker in stocks.values()]

        # Batch insert instead of inside the loop
        with tqdm(total=len(stocks), desc="Populating vector DB") as pbar:
            self.collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            pbar.update(len(stocks))


    def search_stocks(self, query: str, n_results: int = 5, min_similarity: float = 0.3) -> List[Dict[str, str]]:
        """
        Search for stocks based on user query.
        
        Args:
            query: User's natural language query
            n_results: Maximum number of results to return
            min_similarity: Minimum similarity threshold (0-1)
            
        Returns:
            List with identified tickers
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

        stocks_found = []

        if results['ids'] and len(results['ids'][0]) > 0:
            for i, _ in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                
                # Convert distance to similarity (cosine distance -> similarity)
                similarity = 1 - distance
                
                if similarity >= min_similarity:
                    stocks_found.append({
                        "company_name": metadata['company_name'],
                        "ticker": metadata['ticker'],
                        "similarity": round(similarity, 3)
                    })
        
        return stocks_found
    

    def get_ticker(self, company_name: str) -> Optional[str]:
        """
        Get ticker for a specific company name.
        
        Args:
            company_name: Name of the company
            
        Returns:
            Ticker symbol or None
        """
        results = self.search_stocks(company_name, n_results=1, min_similarity=0.7)
        return results[0]["ticker"] if results else None


    def batch_search(self, queries: List[str], n_results: int = 3) -> Dict[str, List[Dict]]:
        """
        Search for multiple stock queries at once.
        
        Args:
            queries: List of company names or descriptions
            
        Returns:
            Dictionary mapping each query to found stocks
        """
        batch_results = {}
        
        for query in queries:
            batch_results[query] = self.search_stocks(query, n_results=n_results)
        
        return batch_results
    

    def get_embedding_stats(self) -> Dict:
        """Get statistics about embeddings"""
        # Sample some embeddings to check normalization
        sample_results = self.collection.get(limit=10, include=['embeddings'])
        
        if sample_results['embeddings']:
            embeddings = np.array(sample_results['embeddings'])
            norms = np.linalg.norm(embeddings, axis=1)
            
            return {
                'count': self.collection.count(),
                'dimension': len(sample_results['embeddings'][0]),
                'mean_norm': float(np.mean(norms)),
                'std_norm': float(np.std(norms)),
                'normalized': bool(np.allclose(norms, 1.0, rtol=1e-5))
            }
        
        return {'count': 0}



# Example usage for your LangGraph agent
if __name__ == "__main__":
    # Initialize database
    db_loc = "artifacts/embeddings/test_db"
    db_name = "Test"

    db = StockVectorDB(persist_directory=db_loc, name=db_name)
    
    # Sample stocks dictionary (you'd have a much larger list)
    stocks = {
        "Apple Inc.": "AAPL",
        "Microsoft Corporation": "MSFT",
        "Amazon.com Inc.": "AMZN",
        "Alphabet Inc.": "GOOGL",
        "Tesla Inc.": "TSLA",
        "NVIDIA Corporation": "NVDA",
        "Meta Platforms Inc.": "META",
        "Berkshire Hathaway Inc.": "BRK.B",
        "JPMorgan Chase & Co.": "JPM",
        "Visa Inc.": "V",
        "Walmart Inc.": "WMT",
        "Procter & Gamble Co.": "PG",
        "Johnson & Johnson": "JNJ",
        "Exxon Mobil Corporation": "XOM",
        "Chevron Corporation": "CVX",
        "Intel Corporation": "INTC",
        "Advanced Micro Devices Inc.": "AMD",
        "Netflix Inc.": "NFLX",
        "Adobe Inc.": "ADBE",
        "Salesforce Inc.": "CRM"
    }
    
    # Populate the database
    db.populate_db(stocks)
    
    # Test queries (simulating user input)
    test_queries = [
        "Apple",
        "tech companies like Microsoft and Google",
        "TSLA",
        "electric vehicle company",
        "streaming service",
        "semiconductor companies",
        "Warren Buffett company"
    ]
    
    print("\n" + "="*60)
    print("TESTING VECTOR SEARCH")
    print("="*60)
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        results = db.search_stocks(query, n_results=3)
        
        if results:
            for stock in results:
                print(f"  → {stock['ticker']}: {stock['company_name']} "
                      f"(similarity: {stock['similarity']})")
        else:
            print("  No matches found")
