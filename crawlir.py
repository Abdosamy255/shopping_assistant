#!/usr/bin/env python3
"""
crawlir.py

Faster Amazon.eg scraper (concurrent product-page fetching).

- Adds concurrent fetching of product detail pages using ThreadPoolExecutor.
- Adds connection pooling and retries to the requests Session.
- Adds --concurrency to control number of worker threads.
- Keeps previous robust parsing logic for descriptions & product info.

Usage:
  python crawlir.py -q "laptop" --detailed --pages 8 --concurrency 8 --output amazon_products.csv

Notes:
- Still respect robots.txt and Amazon TOS.
- Concurrent requests increase throughput but can increase the chance of being blocked.
- If pages require JS to render product info, concurrency won't help — use Playwright/Selenium.
"""
from typing import Dict, Optional
import argparse
import time
import csv
import sys
from urllib.parse import urljoin, quote_plus
import json
import requests
from bs4 import BeautifulSoup
import concurrent.futures
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_USER_AGENT = (
   "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
)
BASE_URL = "https://www.amazon.eg"


def make_session(user_agent: Optional[str] = None, max_pool_connections: int = 20) -> requests.Session:
    """
    Create a requests.Session with connection pooling and sensible retries.
    This session is safe to use concurrently for GET requests.
    """
    s = requests.Session()
    s.headers.update({
        "User-Agent": user_agent or DEFAULT_USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Connection": "keep-alive",
    })

    # Add retries for transient failures
    retries = Retry(total=3, backoff_factor=0.3, status_forcelist=(429, 500, 502, 503, 504))
    adapter = HTTPAdapter(max_retries=retries, pool_connections=max_pool_connections, pool_maxsize=max_pool_connections)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def _clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return " ".join(text.split()).strip()


# ---- Search result parsing ----
def parse_search_results(html: str):
    soup = BeautifulSoup(html, "html.parser")
    items = soup.find_all("div", {"data-component-type": "s-search-result"})
    return items


def extract_from_result_item(item) -> Dict[str, str]:
    data = {
        "asin": item.get("data-asin", "") or "",
        "title": "",
        "price": "",
        "rating": "",
        "image_url": "",
        "product_link": "Link not available",
    }

    title_tag = item.h2
    if title_tag:
        data["title"] = _clean_text(title_tag.get_text(" ", strip=True))

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
    except Exception:
        pass

    r = item.select_one("span.a-icon-alt")
    if r:
        data["rating"] = _clean_text(r.get_text())

    img = item.select_one("img.s-image")
    if img and img.get("src"):
        data["image_url"] = img.get("src")
    elif img and img.get("data-src"):
        data["image_url"] = img.get("data-src")

    link = item.select_one("h2 a, a.a-link-normal.s-no-outline")
    if link and link.get("href"):
        data["product_link"] = urljoin(BASE_URL, link.get("href"))

    return data


# ---- Product page parsing (same robust helpers) ----
def extract_product_description(soup: BeautifulSoup) -> str:
    for sel in (
        "#productDescription_feature_div #productDescription",
        "#productDescription",
        "#productDescription_feature_div",
    ):
        node = soup.select_one(sel)
        if node:
            for bad in node(["script", "style"]):
                bad.decompose()
            lis = [li.get_text(" ", strip=True) for li in node.select("ul li") if li.get_text(strip=True)]
            if lis:
                return _clean_text(" | ".join(lis))
            text = node.get_text(" ", strip=True)
            if text:
                return _clean_text(text)

    bullets_nodes = soup.select("#feature-bullets ul li, #feature-bullets .a-list-item")
    bullets = [li.get_text(" ", strip=True) for li in bullets_nodes if li.get_text(strip=True)]
    if bullets:
        return _clean_text(" | ".join(bullets))

    detail_bullets = soup.select("#detailBullets_feature_div li, #detailBullets_feature_div .a-list-item")
    items = [li.get_text(" ", strip=True) for li in detail_bullets if li.get_text(strip=True)]
    if items:
        return _clean_text(" | ".join(items))

    meta = soup.find("meta", {"name": "description"})
    if meta and meta.get("content"):
        return _clean_text(meta["content"])

    return ""


def extract_product_information(soup: BeautifulSoup) -> Dict[str, str]:
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

    def parse_rows_container(container):
        for li in container.select("li, tr, div"):
            bold = li.find(["b", "th", "span"], class_="a-text-bold")
            if bold:
                k = _clean_text(bold.get_text(" ", strip=True)).rstrip(":")
                clone = BeautifulSoup(str(li), "html.parser")
                for b in clone.find_all(["b", "span"], class_="a-text-bold"):
                    b.decompose()
                v = _clean_text(clone.get_text(" ", strip=True))
                if k:
                    info.setdefault(k, v)
                    continue
            text = _clean_text(li.get_text(" ", strip=True))
            if ":" in text:
                parts = text.split(":", 1)
                k = parts[0].strip().rstrip(":")
                v = parts[1].strip()
                if k:
                    info.setdefault(k, v)

    for tid in ("#productDetails_techSpec_section_1", "#productDetails_techSpec_section_2"):
        table = soup.select_one(tid)
        if table:
            tbl = table.find("table") or table
            parse_table(tbl)

    sec = soup.select_one("#productDetails_detailBullets_sections1")
    if sec:
        tbl = sec.find("table")
        if tbl:
            parse_table(tbl)
        else:
            parse_rows_container(sec)

    prod = soup.select_one("#prodDetails, #productDetails_feature_div, #productDetails_techSpec_section_1")
    if prod:
        tbl = prod.find("table")
        if tbl:
            parse_table(tbl)
        else:
            parse_rows_container(prod)

    detail_nodes = soup.select("#detailBullets_feature_div li, #detailBullets_feature_div .a-list-item")
    for li in detail_nodes:
        text = _clean_text(li.get_text(" ", strip=True))
        if ":" in text:
            parts = text.split(":", 1)
            k = parts[0].strip().rstrip(":")
            v = parts[1].strip()
            if k:
                info.setdefault(k, v)
        else:
            lines = [l for l in text.splitlines() if l.strip()]
            if len(lines) >= 2:
                k = lines[0].rstrip(":").strip()
                v = " ".join(lines[1:]).strip()
                info.setdefault(k, v)

    return info


def fetch_product_page(session: requests.Session, url: str, timeout: int = 15) -> Optional[str]:
    try:
        resp = session.get(url, timeout=timeout)
        if resp.status_code == 200:
            return resp.text
        else:
            print(f"Warning: product page returned {resp.status_code} for {url}")
    except requests.RequestException as exc:
        print(f"Warning: exception fetching product page {url}: {exc}")
    return None


def extract_from_product_page(html: str) -> Dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title_elem = soup.select_one("#productTitle, h1 span")
    title_full = _clean_text(title_elem.get_text()) if title_elem else ""

    price_full = ""
    for pid in ("#priceblock_ourprice", "#priceblock_dealprice", ".a-price .a-offscreen"):
        p = soup.select_one(pid)
        if p and p.get_text(strip=True):
            price_full = _clean_text(p.get_text())
            break

    rating_full = ""
    r = soup.select_one("span.a-icon-alt, #acrPopover .a-icon-alt")
    if r:
        rating_full = _clean_text(r.get_text())

    description = extract_product_description(soup)
    info = extract_product_information(soup)

    image_primary = ""
    img = soup.select_one("img#landingImage, img.a-dynamic-image, meta[property='og:image']")
    if img:
        if img.name == "meta":
            image_primary = img.get("content", "") or ""
        else:
            image_primary = img.get("src") or img.get("data-old-hires") or img.get("data-a-dynamic-image", "")

    return {
        "title_full": title_full,
        "price_full": price_full,
        "rating_full": rating_full,
        "description": description,
        "image_primary": image_primary,
        "product_info_json": json.dumps(info, ensure_ascii=False),
    }


# ---- Concurrency helpers ----
def _fetch_and_merge(session: requests.Session, base_item: Dict[str, str], timeout: int = 15) -> Dict[str, str]:
    """Fetch product page for base_item and merge details in; return merged dict ready for CSV."""
    if base_item.get("product_link") == "Link not available":
        # nothing to fetch
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
    base_item["title"] = details.get("title_full") or base_item.get("title", "")
    base_item["price"] = details.get("price_full") or base_item.get("price", "")
    base_item["rating"] = details.get("rating_full") or base_item.get("rating", "")
    base_item["description"] = details.get("description", "")
    base_item["image_primary"] = details.get("image_primary", base_item.get("image_url", ""))
    base_item["product_info_json"] = details.get("product_info_json", "{}")
    return base_item

def write_csv(path: str, rows: list, fieldnames: list, append: bool = False):
    """
    لو append=True هيزوّد على الملف لو موجود، ومش هيكتب الهيدر تاني.
    لو append=False هيعمل overwrite عادي.
    """
    file_exists = append and os.path.isfile(path)
    mode = "a" if append else "w"

    with open(path, mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)
 #############################           
def crawl_amazon_to_csv(
    query: str,
    output_path: str = "amazon_products.csv",
    language: str = "en",
    pages: int = 2,
    delay: float = 1.5,
    detailed: bool = False,
    max_products: int = 0,
    concurrency: int = 15,
    append: bool = True,
):
    ######################################
    """
    تستخدم جوّا الكود (مثلاً من Streamlit):
    - تاخد query كنص
    - تبحث في Amazon
    - تكتب / تزوّد في CSV
    - وترجع عدد المنتجات اللي اتحفظت
    """
    if pages < 0:
        raise ValueError("pages must be >= 0")

    session = make_session(max_pool_connections=max(20, concurrency + 5))
    query_quoted = quote_plus(query)

    collected = []
    page_number = 1
    fetched_products = 0

    while True:
        if pages and page_number > pages:
            break

        search_url = f"{BASE_URL}/s?k={query_quoted}&language={language}&page={page_number}"
        print(f"[page {page_number}] Fetching search page: {search_url}")
        try:
            resp = session.get(search_url, timeout=20)
        except requests.RequestException as exc:
            print(f"Failed to retrieve page {page_number}: {exc}")
            break

        if resp.status_code != 200:
            print(f"Failed to retrieve page {page_number} (status {resp.status_code}). Stopping.")
            break

        items = parse_search_results(resp.text)
        if not items:
            print("No items found on this page. Stopping pagination.")
            break

        page_bases = [extract_from_result_item(item) for item in items]

        if detailed:
            to_fetch = page_bases
            print(f"  -> Fetching details concurrently for {len(to_fetch)} items (concurrency={concurrency})")
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(_fetch_and_merge, session, base) for base in to_fetch]
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
                            "search_query": query,  # 🔥 نضيف الكويري
                        })
                        fetched_products += 1
                        if max_products and fetched_products >= max_products:
                            break
                    except Exception as exc:
                        print(f"  -> Detail fetch error: {exc}")
                if max_products and fetched_products >= max_products:
                    pass
        else:
            for base in page_bases:
                collected.append({
                    "title": base.get("title", ""),
                    "price": base.get("price", ""),
                    "rating": base.get("rating", ""),
                    "image": base.get("image_url", ""),
                    "product_link": base.get("product_link", ""),
                    "description": "",
                    "product_info_json": "{}",
                    "search_query": query,  # 🔥 نضيف الكويري
                })
                fetched_products += 1
                if max_products and fetched_products >= max_products:
                    break

        if max_products and fetched_products >= max_products:
            print("Reached max-products limit; stopping.")
            break

        page_number += 1
        time.sleep(delay / 3.0)

    if collected:
        fieldnames = [
            "title", "price", "rating",
            "image", "product_link",
            "description", "product_info_json",
            "search_query",
        ]
        write_csv(output_path, collected, fieldnames, append=append)
        print(f"Saved {len(collected)} products to {output_path}")
    else:
        print("No products collected.")

    return len(collected)

def main():
    parser = argparse.ArgumentParser(description="Search for a product on Amazon Egypt and collect results.")
    parser.add_argument("--query", "-q", type=str, required=True, help="Product search query")
    parser.add_argument("--language", "-l", type=str, default="en", help="Language for the search results (en or ar)")
    parser.add_argument("--pages", "-p", type=int, default=2, help="How many search result pages to fetch (set to 0 for unlimited until no items)")
    parser.add_argument("--output", "-o", type=str, default="amazon_products.csv", help="CSV output path")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay in seconds between page fetches")
    parser.add_argument("--detailed", action="store_true", help="Visit each product page to extract extra fields (slower without concurrency)")
    parser.add_argument("--max-products", type=int, default=0, help="Stop after collecting this many products (0 = no limit)")
    parser.add_argument("--concurrency", type=int, default=15, help="Number of concurrent product-page fetches")
    parser.add_argument("--no-append", action="store_true", help="Overwrite output file instead of appending")
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

def crawl_amazon(query: str, pages: int = 2, detailed: bool = False, max_products: int = 30):
    """
    Live Amazon search for integration inside Streamlit.
    Returns a list of product dicts (not CSV).
    """
    results = []

    # prepare request
    q = quote_plus(query)
    session = make_session(max_pool_connections=20)

    page_number = 1
    fetched = 0

    while page_number <= pages:
        search_html = search_url(q, "en", page_number, session)
        if not search_html:
            break

        items = parse_search_results(search_html)
        if not items:
            break

        base_items = [extract_from_result_item(item) for item in items]

        for base in base_items:
            results.append({
                "title": base.get("title", ""),
                "price": base.get("price", ""),
                "rating": base.get("rating", ""),
                "image": base.get("image_url", ""),
                "product_link": base.get("product_link", "")
            })
            fetched += 1
            if fetched >= max_products:
                return results

        page_number += 1

    return results

if __name__ == "__main__":
    main()
