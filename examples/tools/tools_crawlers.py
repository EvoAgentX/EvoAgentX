#!/usr/bin/env python3
"""
Minimal EvoAgentX crawler examples - optimized for low cost.

Demonstrates:
1. RequestCrawler - Basic HTTP crawling
2. Crawl4AI - Browser crawling (if available)
3. Raw response printing to minimize processing costs
"""

import os


def example_request_crawler():
    """Minimal RequestCrawler example with raw output."""
    print("\n📡 RequestCrawler (Raw Output)")
    print("-" * 30)
    
    try:
        from evoagentx.tools import RequestCrawlToolkit
        from evoagentx.tools.crawler_base import DisabledPageContentHandler
        from evoagentx.tools.crawler_request import RequestCrawler
        
        # Use raw handler to avoid processing costs
        crawler = RequestCrawler(
            page_content_handler=DisabledPageContentHandler(max_length=200)
        )
        
        result = crawler.crawl("https://httpbin.org/html")
        
        print(f"Success: {result['success']}")
        print(f"Raw content (first 200 chars):")
        print(result['content'])
        
    except Exception as e:
        print(f"Error: {e}")


def example_crawl4ai_minimal():
    """Minimal Crawl4AI example - basic settings only."""
    print("\n🚀 Crawl4AI (Minimal Settings)")
    print("-" * 30)
    
    try:
        from evoagentx.tools import Crawl4AICrawlToolkit
        
        toolkit = Crawl4AICrawlToolkit(headless=True)
        tool = toolkit.tools[0]
        
        # Minimal settings to reduce execution time
        result = tool(
            url="https://httpbin.org/html",
            output_format="text",
            word_count_threshold=5,
            include_images=False,
            include_links=False,
            scan_full_page=False
        )
        
        print(f"Success: {result['success']}")
        print(f"Content length: {len(result['content'])} chars")
        print(f"First 200 chars: {result['content'][:200]}")
        
    except ImportError:
        print("Crawl4AI not installed - skipping")
    except Exception as e:
        print(f"Error: {e}")


def example_toolkit_usage():
    """Show minimal toolkit usage."""
    print("\n📦 Toolkit Usage")
    print("-" * 30)
    
    try:
        from evoagentx.tools import RequestCrawlToolkit
        
        # Minimal toolkit configuration
        toolkit = RequestCrawlToolkit(max_content_words=50)
        tool = toolkit.tools[0]
        
        result = tool(url="https://httpbin.org/html")
        
        print(f"Tool name: {tool.name}")
        print(f"Success: {result['success']}")
        print(f"Content preview: {result['content'][:100]}...")
        
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Run minimal crawler examples."""
    print("🎭 EvoAgentX Crawlers - Minimal Examples")
    print("=" * 40)
    print("⚡ Optimized for low cost execution")
    
    # Quick environment check
    try:
        import crawl4ai
        print("✅ Crawl4AI available")
    except ImportError:
        print("ℹ️ Crawl4AI not installed")
    
    # Run minimal examples
    example_request_crawler()
    example_crawl4ai_minimal()
    example_toolkit_usage()
    
    print("\n✅ Examples complete - minimal cost execution!")


if __name__ == "__main__":
    main()