#!/usr/bin/env python3
"""
crawlir.py - Improved Amazon.eg Scraper

Improvements:
- Added missing imports (os)
- Better error handling with retry logic
- Rate limiting protection
- Connection pooling
- Timeout management
- Concurrent product fetching
"""

import os
import argparse
import time
import csv
import json
import re
from typing import Dict, Optional, List
from urllib.parse import urljoin, quote_plus

import requests
from bs4 import BeautifulSoup
import concurrent.futures
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DEFAULT_USER_AGENT = (
   "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
   "(KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
)
BASE_URL = "https://www.amazon.eg"


def make_session(
    user_agent: Optional[str] = None, 
    max_pool_connections: int = 20
) -> requests.Session:
    """
    Create a requests.Session with connection pooling and retry logic.
    
    Args:
        user_agent: Custom user agent string
        max_pool_connections: Maximum number of pooled connections
        
    Returns:
        Configured requests Session
    """
    s = requests.Session()
    s.headers.update({
        "User-Agent": user_agent or DEFAULT_USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
    })

    # Add retries for transient failures
    retries = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=["GET", "POST"]
    )
    
    adapter = HTTPAdapter(
        max_retries=retries,
        pool_connections=max_pool_connections,
        pool_maxsize=max_pool_connections
    )
    
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    
    return s


def fetch_with_retry(
    session: requests.Session,
    url: str,
    max_retries: int = 3,
    timeout: int = 15
) -> Optional[requests.Response]:
    """
    Fetch URL with exponential backoff on failures.
    
    Args:
        session: Requests session
        url: URL to fetch
        max_retries: Maximum retry attempts
        timeout: Request timeout in seconds
        
    Returns:
        Response object or None on failure
    """
    for attempt in range(max_retries):
        try:
            response = session.get(url, timeout=timeout)
            
            if response.status_code == 200:
                return response
            elif response.status_code == 429:  # Rate limited
                wait_time = (2 ** attempt) * 2
                print(f"⚠️ Rate limited. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"⚠️ HTTP {response.status_code} for {url}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(2 ** attempt)
                
        except requests.Timeout:
            print(f"⏱️ Timeout on attempt {attempt + 1} for {url}")
            if attempt == max_retries - 1:
                return None
            time.sleep(2 ** attempt)
            
        except requests.RequestException as e:
            print(f"❌ Request error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(2 ** attempt)
    
    return None


def _clean_text(text: Optional[str]) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    return " ".join(text.split()).strip()


def parse_search_results(html: str) -> List:
    """Parse search results page and extract product items."""
    soup = BeautifulSoup(html, "html.parser")
    items = soup.find_all("div", {"data-component-type": "s-search-result"})
    return items


def extract_from_result_item(item) -> Dict[str, str]:
    """Extract basic product info from search result item."""
    data = {
        "asin": item.get("data-asin", "") or "",
        "title": "",
        "price": "",
        "rating": "",
        "image_url": "",
        "product_link": "Link not available",
    }

    # Title
    title_tag = item.h2
    if title_tag:
        data["title"] = _clean_text(title_tag.get_text(" ", strip=True))

    # Price
    try:
        price_whole = item.select_one("span.a-price-whole")
        price_frac = item.select_one("span.a-price-fraction")
        price_sym = item.select_one("span.a-price-symbol")
        
        if price_whole:
            pw = _clean_text(price_whole.get_text())
            pf = _clean_text(price_frac.get_text()) if price_frac else ""
            ps = _clean_text(price_sym.get_text()) if price_sym else ""
            data["price"] = f"{pw}{pf} {ps}".strip()
        else:
            p = item.select_one("span.a-price")
            if p:
                data["price"] = _clean_text(p.get_text())
    except Exception as e:
        print(f"⚠️ Price extraction error: {e}")

    # Rating
    r = item.select_one("span.a-icon-alt")
    if r:
        data["rating"] = _clean_text(r.get_text())

    # Image
    img = item.select_one("img.s-image")
    if img:
        data["image_url"] = img.get("src") or img.get("data-src", "")

    # Product link
    link = item.select_one("h2 a, a.a-link-normal.s-no-outline")
    if link and link.get("href"):
        data["product_link"] = urljoin(BASE_URL, link.get("href"))

    return data


def extract_product_description(soup: BeautifulSoup) -> str:
    """Extract product description from product page."""
    # Try multiple selectors
    for sel in (
        "#productDescription_feature_div #productDescription",
        "#productDescription",
        "#feature-bullets",
    ):
        node = soup.select_one(sel)
        if node:
            # Remove script/style tags
            for bad in node(["script", "style"]):
                bad.decompose()
            
            # Try to extract bullet points
            lis = [
                li.get_text(" ", strip=True) 
                for li in node.select("ul li") 
                if li.get_text(strip=True)
            ]
            if lis:
                return _clean_text(" | ".join(lis))
            
            # Fall back to full text
            text = node.get_text(" ", strip=True)
            if text:
                return _clean_text(text)
    
    return ""


def extract_product_information(soup: BeautifulSoup) -> Dict[str, str]:
    """Extract product specifications from product page."""
    info: Dict[str, str] = {}

    def parse_table(table):
        for tr in table.find_all("tr"):
            th = tr.find("th")
            tds = tr.find_all("td")
            if th and tds:
                k = _clean_text(th.get_text(" ", strip=True)).rstrip(":")
                v = _clean_text(tds[0].get_text(" ", strip=True))
                if k:
                    info.setdefault(k, v)
            elif len(tds) >= 2:
                k = _clean_text(tds[0].get_text(" ", strip=True)).rstrip(":")
                v = _clean_text(tds[1].get_text(" ", strip=True))
                if k:
                    info.setdefault(k, v)

    # Try various product details sections
    for tid in (
        "#productDetails_techSpec_section_1",
        "#productDetails_techSpec_section_2",
        "#prodDetails",
    ):
        section = soup.select_one(tid)
        if section:
            table = section.find("table") or section
            parse_table(table)

    return info


def fetch_product_page(
    session: requests.Session,
    url: str,
    timeout: int = 15
) -> Optional[str]:
    """Fetch product detail page HTML."""
    response = fetch_with_retry(session, url, timeout=timeout)
    return response.text if response else None


def extract_from_product_page(html: str) -> Dict[str, str]:
    """Extract detailed information from product page."""
    soup = BeautifulSoup(html, "html.parser")
    
    # Title
    title_elem = soup.select_one("#productTitle, h1 span")
    title_full = _clean_text(title_elem.get_text()) if title_elem else ""

    # Price
    price_full = ""
    for selector in (
        "#priceblock_ourprice",
        "#priceblock_dealprice",
        ".a-price .a-offscreen"
    ):
        p = soup.select_one(selector)
        if p and p.get_text(strip=True):
            price_full = _clean_text(p.get_text())
            break

    # Rating
    rating_full = ""
    r = soup.select_one("span.a-icon-alt, #acrPopover .a-icon-alt")
    if r:
        rating_full = _clean_text(r.get_text())

    # Description
    description = extract_product_description(soup)
    
    # Product specifications
    info = extract_product_information(soup)

    # Primary image
    image_primary = ""
    img = soup.select_one(
        "img#landingImage, img.a-dynamic-image, meta[property='og:image']"
    )
    if img:
        if img.name == "meta":
            image_primary = img.get("content", "") or ""
        else:
            image_primary = (
                img.get("src") or 
                img.get("data-old-hires") or 
                img.get("data-a-dynamic-image", "")
            )

    return {
        "title_full": title_full,
        "price_full": price_full,
        "rating_full": rating_full,
        "description": description,
        "image_primary": image_primary,
        "product_info_json": json.dumps(info, ensure_ascii=False),
    }


def _fetch_and_merge(
    session: requests.Session,
    base_item: Dict[str, str],
    timeout: int = 15
) -> Dict[str, str]:
    """Fetch product details and merge with base item."""
    if base_item.get("product_link") == "Link not available":
        base_item.setdefault("description", "")
        base_item.setdefault("image_primary", base_item.get("image_url", ""))
        base_item.setdefault("product_info_json", "{}")
        return base_item

    html = fetch_product_page(session, base_item["product_link"], timeout=timeout)
    if not html:
        base_item.setdefault("description", "")
        base_item.setdefault("image_primary", base_item.get("image_url", ""))
        base_item.setdefault("product_info_json", "{}")
        return base_item

    details = extract_from_product_page(html)
    
    # Merge details
    base_item["title"] = details.get("title_full") or base_item.get("title", "")
    base_item["price"] = details.get("price_full") or base_item.get("price", "")
    base_item["rating"] = details.get("rating_full") or base_item.get("rating", "")
    base_item["description"] = details.get("description", "")
    base_item["image_primary"] = details.get(
        "image_primary", 
        base_item.get("image_url", "")
    )
    base_item["product_info_json"] = details.get("product_info_json", "{}")
    
    return base_item


def write_csv(
    path: str,
    rows: List[Dict],
    fieldnames: List[str],
    append: bool = False
):
    """
    Write product data to CSV file.
    
    Args:
        path: Output file path
        rows: List of product dictionaries
        fieldnames: CSV column names
        append: Whether to append to existing file
    """
    # Check if file exists for append mode
    file_exists = append and os.path.isfile(path)
    mode = "a" if append else "w"

    with open(path, mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Write header only if new file or overwrite mode
        if not file_exists:
            writer.writeheader()
        
        for row in rows:
            writer.writerow(row)


def crawl_amazon_to_csv(
    query: str,
    output_path: str = "amazon_products.csv",
    language: str = "en",
    pages: int = 2,
    delay: float = 1.5,
    detailed: bool = False,
    max_products: int = 0,
    concurrency: int = 15,
    append: bool = False,
):
    """
    Main crawling function - searches Amazon and saves results to CSV.
    
    Args:
        query: Search query string
        output_path: Path to output CSV file
        language: Search language ("en" or "ar")
        pages: Number of pages to crawl
        delay: Delay between page requests
        detailed: Whether to fetch detailed product info
        max_products: Maximum products to collect (0 = no limit)
        concurrency: Number of concurrent detail page fetches
        append: Whether to append to existing CSV
        
    Returns:
        int: Number of products collected
    """
    if pages < 0:
        raise ValueError("pages must be >= 0")

    print(f"🔍 Starting Amazon search for: '{query}'")
    print(f"📄 Pages to crawl: {pages}")
    print(f"🎯 Max products: {max_products if max_products > 0 else 'unlimited'}")
    
    # Create session with connection pooling
    session = make_session(max_pool_connections=max(20, concurrency + 5))
    query_quoted = quote_plus(query)

    collected = []
    page_number = 1
    fetched_products = 0

    # Main crawling loop
    while True:
        if pages and page_number > pages:
            break

        search_url = (
            f"{BASE_URL}/s?k={query_quoted}"
            f"&language={language}&page={page_number}"
        )
        
        print(f"📖 Fetching page {page_number}...")
        
        response = fetch_with_retry(session, search_url, timeout=20)
        if not response:
            print(f"❌ Failed to fetch page {page_number}. Stopping.")
            break

        # Parse search results
        items = parse_search_results(response.text)
        if not items:
            print("ℹ️ No more items found. Stopping pagination.")
            break

        print(f"✅ Found {len(items)} items on page {page_number}")
        
        # Extract basic info from search results
        page_bases = [extract_from_result_item(item) for item in items]

        # Fetch detailed info if requested
        if detailed:
            print(f"   📥 Fetching details (concurrency={concurrency})...")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    executor.submit(_fetch_and_merge, session, base)
                    for base in page_bases
                ]
                
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        merged = fut.result()
                        collected.append({
                            "title": merged.get("title", ""),
                            "price": merged.get("price", ""),
                            "rating": merged.get("rating", ""),
                            "image": merged.get("image_primary", merged.get("image_url", "")),
                            "product_link": merged.get("product_link", ""),
                            "description": merged.get("description", ""),
                            "product_info_json": merged.get("product_info_json", "{}"),
                            "search_query": query,
                        })
                        fetched_products += 1
                        
                        if max_products and fetched_products >= max_products:
                            break
                    except Exception as exc:
                        print(f"   ⚠️ Detail fetch error: {exc}")
                
                if max_products and fetched_products >= max_products:
                    break
        else:
            # No detailed fetching
            for base in page_bases:
                collected.append({
                    "title": base.get("title", ""),
                    "price": base.get("price", ""),
                    "rating": base.get("rating", ""),
                    "image": base.get("image_url", ""),
                    "product_link": base.get("product_link", ""),
                    "description": "",
                    "product_info_json": "{}",
                    "search_query": query,
                })
                fetched_products += 1
                
                if max_products and fetched_products >= max_products:
                    break

        # Check if we've hit the limit
        if max_products and fetched_products >= max_products:
            print(f"🎯 Reached max-products limit ({max_products})")
            break

        page_number += 1
        time.sleep(delay / 3.0)

    # Save results to CSV
    if collected:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        fieldnames = [
            "title", "price", "rating",
            "image", "product_link",
            "description", "product_info_json",
            "search_query",
        ]
        
        write_csv(output_path, collected, fieldnames, append=append)
        print(f"💾 Saved {len(collected)} products to {output_path}")
    else:
        print("⚠️ No products collected")

    return len(collected)


def main():
    """Command-line interface for the crawler."""
    parser = argparse.ArgumentParser(
        description="Search Amazon Egypt and collect product data"
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        required=True,
        help="Product search query"
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        default="en",
        help="Language for search results (en or ar)"
    )
    parser.add_argument(
        "--pages", "-p",
        type=int,
        default=2,
        help="Number of pages to fetch (0 = unlimited)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="amazon_products.csv",
        help="CSV output path"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between page fetches (seconds)"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Fetch detailed product information"
    )
    parser.add_argument(
        "--max-products",
        type=int,
        default=0,
        help="Maximum products to collect (0 = no limit)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=15,
        help="Number of concurrent detail page fetches"
    )
    parser.add_argument(
        "--no-append",
        action="store_true",
        help="Overwrite output file instead of appending"
    )
    
    args = parser.parse_args()

    crawl_amazon_to_csv(
        query=args.query,
        output_path=args.output,
        language=args.language,
        pages=args.pages,
        delay=args.delay,
        detailed=args.detailed,
        max_products=args.max_products,
        concurrency=args.concurrency,
        append=not args.no_append,
    )


if __name__ == "__main__":
    main()