#!/usr/bin/env python3

"""
Search and Request Tools Examples for EvoAgentX

This module provides comprehensive examples for:
- WikipediaSearchToolkit: Search Wikipedia for information
- GoogleSearchToolkit: Search Google using the official API
- GoogleFreeSearchToolkit: Search Google without requiring an API key
- DDGSSearchToolkit: Search using DuckDuckGo
- SerpAPIToolkit: Multi-engine search (Google, Bing, Baidu, Yahoo, DuckDuckGo)
- SerperAPIToolkit: Google search via SerperAPI
- ExaSearchToolkit: Search using Exa AI (requires EXA_API_KEY)
- RequestToolkit: Perform HTTP operations (GET, POST, PUT, DELETE)
- ArxivToolkit: Search for research papers
- RSSToolkit: Fetch and validate RSS feeds
- ResearchToolkit: Academic research tools for paper search, metadata
  fetching, and BibTeX lookup (ArxivPaperSearchTool works without API keys;
  PaperSearchTool, FetchPaperMetaDataTool, and BibReferenceTool require
  SERPAPI_KEY and optionally OPENROUTER_API_KEY)

The examples demonstrate various search capabilities and HTTP operations.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to import from evoagentx
sys.path.append(str(Path(__file__).parent.parent))

from evoagentx.tools import (
    WikipediaSearchToolkit,
    GoogleSearchToolkit,
    GoogleFreeSearchToolkit,
    DDGSSearchToolkit,
    SerpAPIToolkit,
    SerperAPIToolkit,
    ArxivToolkit,
    RSSToolkit,
    RequestToolkit,
    ResearchToolkit,
)
from evoagentx.tools.search_exa import ExaSearchToolkit
from evoagentx.tools.research_tools import (
    ArxivPaperSearchTool,
    PaperSearchTool,
    FetchPaperMetaDataTool,
    BibReferenceTool,
)


def run_search_examples():
    """
    Run examples using the search toolkits (Wikipedia, Google, Google Free, DDGS, SerpAPI, and SerperAPI).
    """
    print("\n===== SEARCH TOOLS EXAMPLES =====\n")
    
    # Initialize search toolkits
    wiki_toolkit = WikipediaSearchToolkit(max_summary_sentences=3)
    google_toolkit = GoogleSearchToolkit(num_search_pages=3, max_content_words=200)
    google_free_toolkit = GoogleFreeSearchToolkit()
    ddgs_toolkit = DDGSSearchToolkit(num_search_pages=3, max_content_words=200, backend="auto", region="us-en")
    
    # Initialize SerpAPI toolkit (will check for API key)
    serpapi_toolkit = SerpAPIToolkit(
        num_search_pages=3, 
        max_content_words=300,
        enable_content_scraping=True
    )
    
    # Initialize SerperAPI toolkit (will check for API key)
    serperapi_toolkit = SerperAPIToolkit(
        num_search_pages=3,
        max_content_words=300,
        enable_content_scraping=True
    )
    
    # Get the individual tools from toolkits
    wiki_tool = wiki_toolkit.get_tool("wikipedia_search")
    google_tool = google_toolkit.get_tool("google_search")
    google_free_tool = google_free_toolkit.get_tool("google_free_search")
    ddgs_tool = ddgs_toolkit.get_tool("ddgs_search")
    serpapi_tool = serpapi_toolkit.get_tool("serpapi_search")
    serperapi_tool = serperapi_toolkit.get_tool("serperapi_search")
    
    # Example search query
    query = "artificial intelligence agent architecture"
    
    # Run Wikipedia search example
    try:
        print("\nWikipedia Search Example:")
        print("-" * 50)
        wiki_results = wiki_tool(query=query, num_search_pages=2)
        
        if wiki_results.get("error"):
            print(f"Error: {wiki_results['error']}")
        else:
            for i, result in enumerate(wiki_results.get("results", [])):
                print(f"Result {i+1}: {result['title']}")
                print(f"Summary: {result['summary'][:150]}...")
                print(f"URL: {result['url']}")
                print("-" * 30)
    except Exception as e:
        print(f"Error running Wikipedia search: {str(e)}")
    
    # Run Google search example (requires API key)
    try:
        print("\nGoogle Search Example (requires API key):")
        print("-" * 50)
        google_results = google_tool(query=query)
        
        if google_results.get("error"):
            print(f"Error: {google_results['error']}")
        else:
            for i, result in enumerate(google_results.get("results", [])):
                print(f"Result {i+1}: {result['title']}")
                print(f"URL: {result['url']}")
                print("-" * 30)
    except Exception as e:
        print(f"Error running Google search: {str(e)}")
    
    # Run Google Free search example
    try:
        print("\nGoogle Free Search Example:")
        print("-" * 50)
        free_results = google_free_tool(query=query, num_search_pages=2)
        
        if free_results.get("error"):
            print(f"Error: {free_results['error']}")
        else:
            for i, result in enumerate(free_results.get("results", [])):
                print(f"Result {i+1}: {result['title']}")
                print(f"URL: {result['url']}")
                print("-" * 30)
    except Exception as e:
        print(f"Error running free Google search: {str(e)}")
    
    # Run DDGS search example
    try:
        print("\nDDGS Search Example:")
        print("-" * 50)
        ddgs_results = ddgs_tool(query=query, num_search_pages=2, backend="duckduckgo")
        
        if ddgs_results.get("error"):
            print(f"Error: {ddgs_results['error']}")
        else:
            for i, result in enumerate(ddgs_results.get("results", [])):
                print(f"Result {i+1}: {result['title']}")
                print(f"Result full: \n{result}")
                print(f"URL: {result['url']}")
                print("-" * 30)
    except Exception as e:
        print(f"Error running DDGS search: {str(e)}")
    
    # Run SerpAPI search example (requires API key)
    serpapi_api_key = os.getenv("SERPAPI_KEY")
    if serpapi_api_key:
        try:
            print("\nSerpAPI Search Example (with content scraping):")
            print("-" * 50)
            print(f"✓ Using SerpAPI key: {serpapi_api_key[:8]}...")
            
            serpapi_results = serpapi_tool(
                query=query, 
                num_search_pages=3,
                max_content_words=300,
                engine="google",
                location="United States",
                language="en"
            )
            
            if serpapi_results.get("error"):
                print(f"Error: {serpapi_results['error']}")
            else:
                # Display processed results
                print(f"SerpAPI results: {serpapi_results}")
                
        except Exception as e:
            print(f"Error running SerpAPI search: {str(e)}")
    else:
        print("\nSerpAPI Search Example:")
        print("-" * 50)
        print("❌ SERPAPI_KEY not found in environment variables")
        print("To test SerpAPI search, set your API key:")
        print("export SERPAPI_KEY='your-serpapi-key-here'")
        print("Get your key from: https://serpapi.com/")
        print("✓ SerpAPI toolkit initialized successfully (API key required for search)")
    
    # Run SerperAPI search example (requires API key)
    serperapi_api_key = os.getenv("SERPERAPI_KEY")
    if serperapi_api_key:
        try:
            print("\nSerperAPI Search Example (with content scraping):")
            print("-" * 50)
            print(f"✓ Using SerperAPI key: {serperapi_api_key[:8]}...")
            
            serperapi_results = serperapi_tool(
                query=query,
                num_search_pages=3,
                max_content_words=300,
                location="United States",
                language="en"
            )
            
            if serperapi_results.get("error"):
                print(f"Error: {serperapi_results['error']}")
            else:
                print(f"SerperAPI results: {serperapi_results}")
                
        except Exception as e:
            print(f"Error running SerperAPI search: {str(e)}")
    else:
        print("\nSerperAPI Search Example:")
        print("-" * 50)
        print("❌ SERPERAPI_KEY not found in environment variables")
        print("To test SerperAPI search, set your API key:")
        print("export SERPERAPI_KEY='your-serperapi-key-here'")
        print("Get your key from: https://serper.dev/")
        print("✓ SerperAPI toolkit initialized successfully (API key required for search)")


def run_arxiv_tool_example():
    """Simple example using ArxivToolkit to search for papers."""
    print("\n===== ARXIV TOOL EXAMPLE =====\n")
    
    try:
        # Initialize the arXiv toolkit
        arxiv_toolkit = ArxivToolkit()
        search_tool = arxiv_toolkit.get_tool("arxiv_search")
        
        print("✓ ArxivToolkit initialized")
        
        # Search for machine learning papers
        print("Searching for 'machine learning' papers...")
        result = search_tool(
            search_query="all:machine learning",
            max_results=3
        )
        
        if result.get('success'):
            papers = result.get('papers', [])
            print(f"✓ Found {len(papers)} papers")
            
            for i, paper in enumerate(papers):
                print(f"\nPaper {i+1}: {paper.get('title', 'No title')}")
                print(f"  Authors: {', '.join(paper.get('authors', ['Unknown']))}")
                print(f"  arXiv ID: {paper.get('arxiv_id', 'Unknown')}")
                print(f"  URL: {paper.get('url', 'No URL')}")
        else:
            print(f"❌ Search failed: {result.get('error', 'Unknown error')}")
        
        print("\n✓ ArxivToolkit test completed")
        
    except Exception as e:
        print(f"Error: {str(e)}")


def run_rss_tool_example():
    """Powerful example using RSSToolkit for RSS feed operations."""
    print("\n===== RSS TOOL EXAMPLE =====\n")
    
    try:
        # Initialize RSS toolkit
        toolkit = RSSToolkit(name="DemoRSSToolkit")
        
        print("✓ RSSToolkit initialized")
        
        # Get tools
        fetch_tool = toolkit.get_tool("rss_fetch")
        validate_tool = toolkit.get_tool("rss_validate")
        
        # Test RSS feed URLs
        test_feeds = [
            "https://feeds.bbci.co.uk/news/rss.xml",  # BBC News
            "https://rss.cnn.com/rss/edition.rss",    # CNN
            "https://feeds.feedburner.com/TechCrunch" # TechCrunch
        ]
        
        for feed_url in test_feeds:
            print(f"\n--- Testing RSS Feed: {feed_url} ---")
            
            # Validate the feed
            print("1. Validating RSS feed...")
            validate_result = validate_tool(url=feed_url)
            
            if validate_result.get("success") and validate_result.get("is_valid"):
                print(f"✓ Valid {validate_result.get('feed_type')} feed: {validate_result.get('title', 'Unknown')}")
                
                # Fetch the feed
                print("2. Fetching RSS feed...")
                fetch_result = fetch_tool(feed_url=feed_url, max_entries=3)
                
                if fetch_result.get("success"):
                    entries = fetch_result.get("entries", [])
                    print(f"✓ Fetched {len(entries)} entries from '{fetch_result.get('title')}'")
                    
                    # Display first few entries
                    for i, entry in enumerate(entries[:2], 1):
                        print(f"  Entry {i}: {entry.get('title', 'No title')}")
                        print(f"    Published: {entry.get('published', 'Unknown')}")
                        print(f"    Link: {entry.get('link', 'No link')}")
                        print(f"    Author: {entry.get('author', 'Unknown')}")
                        print()
                
                # Test monitoring for recent entries
                print("3. Testing feed monitoring...")
                
            else:
                print(f"❌ Invalid or inaccessible feed: {validate_result.get('error', 'Unknown error')}")
        
        print("\n✓ RSSToolkit test completed")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Note: RSS feed availability may vary. Some feeds may be temporarily unavailable.")


def run_exa_search_example():
    """Example using ExaSearchToolkit (requires EXA_API_KEY)."""
    print("\n===== EXA SEARCH TOOL EXAMPLE =====\n")

    exa_api_key = os.getenv("EXA_API_KEY")
    if not exa_api_key:
        print("❌ EXA_API_KEY not found in environment variables")
        print("To test Exa search, set your API key:")
        print("export EXA_API_KEY='your-exa-api-key-here'")
        print("Get your key from: https://exa.ai/")
        print("✓ ExaSearchToolkit initialized successfully (API key required for search)")
        return

    try:
        toolkit = ExaSearchToolkit(
            num_search_pages=3,
            max_content_words=120,
            search_type="auto",
            content_mode="highlights",
        )
        exa_search = toolkit.get_tool("exa_search")
        print("✓ ExaSearchToolkit initialized")
        print(f"✓ Using Exa API key: {exa_api_key[:8]}...")

        query = "artificial intelligence agent architecture"
        print(f"Searching for '{query}'...")

        result = exa_search(
            query=query,
            num_search_pages=3,
            max_content_words=120,
            search_type="auto",
            content_mode="highlights",
            user_location="US",
        )

        if result.get("error"):
            print(f"❌ Error: {result['error']}")
        else:
            results = result.get("results", [])
            print(f"✓ Returned {len(results)} results")
            for i, item in enumerate(results, start=1):
                print(f"\nResult {i}: {item.get('title', 'No Title')}")
                print(f"  URL: {item.get('url', '')}")
                content = item.get("content")
                if content:
                    print(f"  Content: {content[:200]}...")
                print("-" * 30)

        print("\n✓ ExaSearchToolkit test completed")

    except Exception as e:
        print(f"Error running Exa search: {str(e)}")


def run_request_tool_example():
    """Simple example using RequestToolkit for HTTP operations."""
    print("\n===== REQUEST TOOL EXAMPLE =====\n")
    
    try:
        # Initialize the request toolkit
        request_toolkit = RequestToolkit(name="DemoRequestToolkit")
        http_tool = request_toolkit.get_tool("http_request")
        
        print("✓ RequestToolkit initialized")
        
        # Test GET request
        print("1. Testing GET request...")
        get_result = http_tool(
            url="https://httpbin.org/get",
            method="GET",
            params={"test": "param", "example": "value"}
        )
        
        if get_result.get("success"):
            print("✓ GET request successful")
            print(f"Status: {get_result.get('status_code')}")
            print(f"Response size: {len(str(get_result.get('content', '')))} characters")
        else:
            print(f"❌ GET request failed: {get_result.get('error', 'Unknown error')}")
        
        # Test POST request with JSON data
        print("\n2. Testing POST request with JSON...")
        post_result = http_tool(
            url="https://httpbin.org/post",
            method="POST",
            json_data={"name": "Test User", "email": "test@example.com"},
            headers={"Content-Type": "application/json"}
        )
        
        if post_result.get("success"):
            print("✓ POST request successful")
            print(f"Status: {post_result.get('status_code')}")
            content = post_result.get('content', '')
            if isinstance(content, dict) and 'json' in content:
                print(f"✓ JSON data received: {content['json']}")
        else:
            print(f"❌ POST request failed: {post_result.get('error', 'Unknown error')}")
        
        # Test PUT request
        print("\n3. Testing PUT request...")
        put_result = http_tool(
            url="https://httpbin.org/put",
            method="PUT",
            data={"update": "new value", "timestamp": "2024-01-01"}
        )
        
        if put_result.get("success"):
            print("✓ PUT request successful")
            print(f"Status: {put_result.get('status_code')}")
        else:
            print(f"❌ PUT request failed: {put_result.get('error', 'Unknown error')}")
        
        # Test DELETE request
        print("\n4. Testing DELETE request...")
        delete_result = http_tool(
            url="https://httpbin.org/delete",
            method="DELETE"
        )
        
        if delete_result.get("success"):
            print("✓ DELETE request successful")
            print(f"Status: {delete_result.get('status_code')}")
        else:
            print(f"❌ DELETE request failed: {delete_result.get('error', 'Unknown error')}")
        
        # Test error handling with invalid URL
        print("\n5. Testing error handling...")
        error_result = http_tool(
            url="https://invalid-domain-that-does-not-exist-12345.com",
            method="GET"
        )
        
        if not error_result.get("success"):
            print("✓ Error handling working correctly")
            print(f"Error: {error_result.get('error', 'Unknown error')}")
        else:
            print("⚠ Error handling may not be working as expected")
        
        print("\n✓ RequestToolkit test completed")
        
    except Exception as e:
        print(f"Error: {str(e)}")


def run_research_tools_example():
    """
    Showcase for the ResearchToolkit and its individual tools.

    Tools demonstrated:
    - ResearchToolkit (unified toolkit, auto-registers tools based on available API keys)
    - ArxivPaperSearchTool (no API key required)
    - PaperSearchTool (requires SERPAPI_KEY)
    - FetchPaperMetaDataTool (requires SERPAPI_KEY)
    - BibReferenceTool (requires SERPAPI_KEY)
    """
    print("\n===== RESEARCH TOOLS EXAMPLES =====\n")

    serpapi_key = os.getenv("SERPAPI_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    # ------------------------------------------------------------------
    # 1. ArxivPaperSearchTool — works without any API keys
    # ------------------------------------------------------------------
    try:
        print("1. ArxivPaperSearchTool (no API key needed)")
        print("-" * 50)

        arxiv_tool = ArxivPaperSearchTool()
        print("✓ ArxivPaperSearchTool initialized")

        query = "retrieval-augmented generation"
        print(f"Searching arXiv for '{query}' (top 3)...")

        result = arxiv_tool(query=query, topk=3)
        papers = result.result  # dict returned by Arxiv.search_arxiv

        if isinstance(papers, dict) and papers.get("papers"):
            for i, paper in enumerate(papers["papers"], start=1):
                print(f"\n  Paper {i}: {paper.get('title', 'N/A')}")
                print(f"    Authors : {', '.join(paper.get('authors', [])[:3])}")
                print(f"    Published: {paper.get('published_date', 'N/A')}")
                links = paper.get("links", {})
                print(f"    Link    : {links.get('html', links.get('pdf', 'N/A'))}")
                summary = paper.get("summary", "")
                if summary:
                    print(f"    Summary : {summary[:150]}...")
        elif isinstance(papers, list):
            for i, paper in enumerate(papers[:3], start=1):
                print(f"\n  Paper {i}: {paper.get('title', 'N/A')}")
        else:
            print(f"  Raw result: {str(papers)[:300]}")

        print("\n✓ ArxivPaperSearchTool test completed")

    except Exception as e:
        print(f"Error running ArxivPaperSearchTool: {e}")

    # ------------------------------------------------------------------
    # 2. ResearchToolkit — unified toolkit (auto-detects API keys)
    # ------------------------------------------------------------------
    try:
        print("\n2. ResearchToolkit (unified toolkit)")
        print("-" * 50)

        research_toolkit = ResearchToolkit()
        available_tools = [t.name for t in research_toolkit.tools]
        print(f"✓ ResearchToolkit initialized with tools: {available_tools}")

        # The toolkit always registers ArxivPaperSearchTool as 'search_papers'
        search_tool = research_toolkit.get_tool("search_papers")
        if search_tool:
            print("✓ 'search_papers' tool is available via the toolkit")

            result = search_tool(query="large language model agents", topk=2)
            papers = result.result
            if isinstance(papers, dict) and papers.get("papers"):
                print(f"  Found {len(papers['papers'])} papers")
                for p in papers["papers"][:2]:
                    print(f"    - {p.get('title', 'N/A')}")
            else:
                print(f"  Result: {str(papers)[:200]}")
        else:
            print("⚠ 'search_papers' tool not found in toolkit")

        # Check for full-featured tools
        if "fetch_paper_metadata" in available_tools:
            print("✓ 'fetch_paper_metadata' tool is available (SERPAPI_KEY detected)")
        if "search_bibtex" in available_tools:
            print("✓ 'search_bibtex' tool is available (SERPAPI_KEY detected)")

        print("\n✓ ResearchToolkit test completed")

    except Exception as e:
        print(f"Error running ResearchToolkit: {e}")

    # ------------------------------------------------------------------
    # 3. PaperSearchTool — full-featured multi-source search
    # ------------------------------------------------------------------
    if serpapi_key:
        try:
            print("\n3. PaperSearchTool (multi-source, requires SERPAPI_KEY)")
            print("-" * 50)
            print(f"✓ Using SERPAPI_KEY: {serpapi_key[:8]}...")

            from evoagentx.tools.storage_handler import LocalStorageHandler
            storage = LocalStorageHandler()

            paper_search = PaperSearchTool(
                storage_handler=storage,
                serpapi_key=serpapi_key,
                openrouter_key=openrouter_key,
            )
            print("✓ PaperSearchTool initialized")

            # 3a. General search
            print("\n  3a. General search mode:")
            result = paper_search(
                queries=["retrieval-augmented generation"],
                search_mode="general",
                topk=3,
            )
            query_papers = result.result.get("query_papers", {})
            for query_text, papers in query_papers.items():
                print(f"    Query: '{query_text}' → {len(papers)} papers")
                for p in papers[:2]:
                    print(f"      - {p.get('paper_title', 'N/A')} ({p.get('year', '?')})")
                    print(f"        Citations: {p.get('citation_count', 'N/A')}")

            # 3b. Venue-specific search (arXiv)
            print("\n  3b. Venue-specific search (arXiv):")
            result = paper_search(
                queries=["arxiv: multi-agent cooperation"],
                search_mode="venue_specific",
                topk=3,
                year_from=2024,
            )
            query_papers = result.result.get("query_papers", {})
            for query_text, papers in query_papers.items():
                print(f"    Query: '{query_text}' → {len(papers)} papers")
                for p in papers[:2]:
                    print(f"      - {p.get('paper_title', 'N/A')} ({p.get('year', '?')})")

            # 3c. Lookup a specific paper
            print("\n  3c. Lookup mode (find a specific paper):")
            result = paper_search(
                queries=["Attention is All You Need"],
                search_mode="lookup",
                topk=1,
            )
            query_papers = result.result.get("query_papers", {})
            for query_text, papers in query_papers.items():
                if papers:
                    p = papers[0]
                    print(f"    Found: {p.get('paper_title', 'N/A')}")
                    print(f"    Link : {p.get('paper_link', 'N/A')}")
                    print(f"    Year : {p.get('year', 'N/A')}")
                    print(f"    Cite : {p.get('citation_count', 'N/A')}")

            print("\n  ✓ PaperSearchTool test completed")

        except Exception as e:
            print(f"Error running PaperSearchTool: {e}")
    else:
        print("\n3. PaperSearchTool (multi-source)")
        print("-" * 50)
        print("❌ SERPAPI_KEY not found — skipping PaperSearchTool")
        print("   Set SERPAPI_KEY to enable: export SERPAPI_KEY='your-key'")
        print("   Get a key from: https://serpapi.com/")

    # ------------------------------------------------------------------
    # 4. FetchPaperMetaDataTool — detailed metadata retrieval
    # ------------------------------------------------------------------
    if serpapi_key:
        try:
            print("\n4. FetchPaperMetaDataTool (requires SERPAPI_KEY)")
            print("-" * 50)

            fetch_tool = FetchPaperMetaDataTool(
                serpapi_key=serpapi_key,
                openrouter_key=openrouter_key,
            )
            print("✓ FetchPaperMetaDataTool initialized")

            papers_to_fetch = [
                {"paper_title": "Attention is All You Need", "paper_link": "https://arxiv.org/abs/1706.03762"},
            ]
            print(f"  Fetching metadata for {len(papers_to_fetch)} paper(s)...")

            result = fetch_tool(papers=papers_to_fetch)
            metadata_list = result.result.get("paper_metadata", [])

            for meta in metadata_list:
                print(f"\n  Title   : {meta.get('paper_title', 'N/A')}")
                authors = meta.get("authors", [])
                if authors:
                    print(f"  Authors : {', '.join(authors[:5])}{'...' if len(authors) > 5 else ''}")
                print(f"  Year    : {meta.get('year', 'N/A')}")
                print(f"  Venue   : {meta.get('venue', 'N/A')}")
                print(f"  Citations: {meta.get('citation_count', 'N/A')}")
                print(f"  Link    : {meta.get('paper_link', 'N/A')}")
                print(f"  PDF     : {meta.get('paper_pdf_link', 'N/A')}")
                abstract = meta.get("abstract", "")
                if abstract:
                    print(f"  Abstract: {abstract[:200]}...")

            print("\n  ✓ FetchPaperMetaDataTool test completed")

        except Exception as e:
            print(f"Error running FetchPaperMetaDataTool: {e}")
    else:
        print("\n4. FetchPaperMetaDataTool")
        print("-" * 50)
        print("❌ SERPAPI_KEY not found — skipping FetchPaperMetaDataTool")

    # ------------------------------------------------------------------
    # 5. BibReferenceTool — BibTeX lookup
    # ------------------------------------------------------------------
    if serpapi_key:
        try:
            print("\n5. BibReferenceTool (requires SERPAPI_KEY)")
            print("-" * 50)

            bib_tool = BibReferenceTool(
                serpapi_key=serpapi_key,
                openrouter_key=openrouter_key,
            )
            print("✓ BibReferenceTool initialized")

            titles = [
                "Attention is All You Need",
                "BERT: Pre-training of Deep Bidirectional Transformers",
            ]
            print(f"  Searching BibTeX for {len(titles)} paper(s)...")

            result = bib_tool(titles_or_keywords=titles)
            bibtex_entries = result.result.get("bibtex_entries", {})

            for title, bib in bibtex_entries.items():
                print(f"\n  Title: {title}")
                if bib and "@" in bib:
                    # Show first 5 lines of BibTeX entry
                    bib_lines = bib.strip().splitlines()
                    for line in bib_lines[:5]:
                        print(f"    {line}")
                    if len(bib_lines) > 5:
                        print(f"    ... ({len(bib_lines) - 5} more lines)")
                else:
                    print(f"    BibTeX not found: {bib}")

            print("\n  ✓ BibReferenceTool test completed")

        except Exception as e:
            print(f"Error running BibReferenceTool: {e}")
    else:
        print("\n5. BibReferenceTool")
        print("-" * 50)
        print("❌ SERPAPI_KEY not found — skipping BibReferenceTool")

    print("\n✓ Research tools examples completed")


def main():
    """Main function to run all search and request examples"""
    print("===== SEARCH AND REQUEST TOOLS EXAMPLES =====")
    
    # # Run search tools examples
    # run_search_examples()

    # # Run arXiv tool example
    # run_arxiv_tool_example()

    # # Run RSS tool example
    # run_rss_tool_example()

    # # Run Exa search example (requires EXA_API_KEY)
    # run_exa_search_example()

    # # Run Request tool example
    # run_request_tool_example()

    # Run Research tools example (ArxivPaperSearchTool, PaperSearchTool,
    # FetchPaperMetaDataTool, BibReferenceTool via ResearchToolkit)
    run_research_tools_example()

    print("\n===== ALL SEARCH AND REQUEST EXAMPLES COMPLETED =====")


if __name__ == "__main__":
    main()
