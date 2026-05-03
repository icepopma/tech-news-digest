#!/usr/bin/env python3
"""
Fetch articles from websites via Jina Reader API (for sites without RSS feeds).

Reads sources.json, filters for type=="jina", fetches each URL through
https://r.jina.ai/{url} to get markdown content, extracts article links
and titles, and outputs structured JSON compatible with merge-sources.py.

Usage:
    python3 fetch-jina.py [--defaults DIR] [--config DIR] [--hours 48] [--output FILE] [--verbose]
"""

import json
import re
import sys
import os
import argparse
import logging
import time
import tempfile
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from urllib.parse import urljoin, urlparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

TIMEOUT = 30
MAX_WORKERS = 3  # Jina rate limits
MAX_ARTICLES_PER_SOURCE = 15
RETRY_COUNT = 1
RETRY_DELAY = 3.0
JINA_BASE = "https://r.jina.ai/"

# Navigation boilerplate patterns to exclude
NAV_PATTERNS = [
    r'^skip to', r'^go to', r'^jump to',
    r'^(about|contact|login|sign.?in|sign.?up|subscribe|newsletter)',
    r'^(privacy|terms|cookie|accessibility|copyright)',
    r'^(home|back|menu|navigation|search)',
    r'^(follow us|share|social|email us)',
    r'^(read more|learn more|find out more|view all|see all)',
    r'^(book a demo|try it free|get started|start here)',
    r'^(upgrade to pro|pricing|enterprise)',
    r'^(developers|research index|company|business)',
    r'^\d+k (reads|followers|subscribers|stars)',
    r'^©\s*\d{4}',
    r'^my favourites',
    r'^prototypes?$',
    r'^writing$',
    r'^speaking$',
    r'^about$',
    r'^more than \d',
    r'^(premium|free trial|save \d+%)',
    r'^not boring$',
    r'^deep dives?$',
    r'^vertical integrators?$',
    r'^ai job board',
    r'^fellowship programs?$',
    r'^student affinity',
    r'^subscribe to email',
    r'^(pause|play) media$',
]

# URL patterns to exclude (navigation/footer links)
URL_SKIP_PATTERNS = [
    '/cdn-cgi', '/privacy', '/terms', '/login', '/signup', '/signin',
    '/about', '/contact', '/subscribe', '/search',
    '#', 'javascript:', 'mailto:', '/feed', '/rss',
    '.xml', '.json', '/sitemap', '/cdn/', '/static/',
    'twitter.com', 'x.com', 'facebook.com', 'linkedin.com',
    'github.com/login', 'github.com/feedback',
    'chromewebstore.google.com',
    '/start-here', '/prototyping', '/speaking',
]

# File path patterns that look like articles
ARTICLE_URL_PATTERNS = [
    r'/blog/', r'/engineering/', r'/research/', r'/news/', r'/post/',
    r'/article/', r'/publish/', r'/paper/', r'/report/', r'/insight/',
    r'/updates?/', r'/announcements?/', r'/notes?/',
    r'/writing/', r'/essays?/', r'/tutorials?',
    r'/\d{4}/',  # Year in path like /2026/...
    r'/\d{4}-\d{2}-\d{2}',  # Date in path
    r'[a-z]-[a-z].*/\w{3,}',  # slug-style paths like /managed-agents
]


def setup_logging(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


def is_nav_title(title: str) -> bool:
    """Check if a title looks like navigation boilerplate."""
    title_lower = title.lower().strip()
    if len(title_lower) < 8:
        return True
    for pattern in NAV_PATTERNS:
        if re.match(pattern, title_lower):
            return True
    return False


def is_article_url(url: str) -> bool:
    """Check if a URL looks like an article (has a meaningful path)."""
    if not url:
        return False
    
    # Skip known non-article URLs
    url_lower = url.lower()
    for pattern in URL_SKIP_PATTERNS:
        if pattern in url_lower:
            return False
    
    parsed = urlparse(url)
    path = parsed.path.rstrip('/')
    
    # Must have a meaningful path (not just / or empty)
    if not path or path == '/':
        return False
    
    # Check for article-like path patterns
    for pattern in ARTICLE_URL_PATTERNS:
        if re.search(pattern, path):
            return True
    
    # Path with 2+ segments that looks like a slug
    parts = [p for p in path.split('/') if p]
    if len(parts) >= 2:
        # Like /engineering/managed-agents or /blog/post-slug
        last_part = parts[-1]
        if len(last_part) >= 5 and '-' in last_part:
            return True
    
    # Single-segment paths that are slug-like (e.g., /contextual-retrieval)
    if len(parts) == 1 and len(parts[0]) >= 8 and '-' in parts[0]:
        return True
    
    return False


def parse_jina_markdown(content: str, source_url: str) -> List[Dict[str, Any]]:
    """Extract articles from Jina Reader markdown output.
    
    Strategy:
    1. Find all markdown links [Title](URL) 
    2. Filter by: title quality (not nav boilerplate) + URL quality (looks like article)
    3. Dedup by URL
    """
    articles = []
    seen_urls: Set[str] = set()
    
    base_domain = urlparse(source_url).netloc.lower().replace("www.", "")
    
    # Find all markdown links
    # Pattern matches [Title](URL) including multi-line titles
    link_pattern = re.compile(r'\[([^\]]+)\]\((https?://[^\s\)]+)\)')
    
    for match in link_pattern.finditer(content):
        raw_title = match.group(1).strip()
        url = match.group(2).strip()
        
        # Clean title
        # Remove image markers: ![Image N: text]
        title = re.sub(r'!?#{1,6}\s*', '', raw_title)
        title = re.sub(r'Image \d+:?\s*', '', title)
        title = re.sub(r'#{1,4}\s+', ' ', title)
        title = re.sub(r'\s+', ' ', title).strip()
        
        # Remove trailing dates from title (keep them for article though)
        date_match = re.search(r'(\w{3}\s+\d{1,2},?\s+\d{4})$', title)
        article_date = None
        if date_match:
            article_date = date_match.group(1)
            title = title[:date_match.start()].strip()
        
        # Filter by title quality
        if is_nav_title(title):
            continue
        
        # Filter by URL quality - must look like an article
        if not is_article_url(url):
            continue
        
        # Domain check - prefer same domain
        link_domain = urlparse(url).netloc.lower().replace("www.", "")
        if link_domain and base_domain:
            # Allow same domain or parent/subdomain
            if not (link_domain == base_domain or 
                    link_domain.endswith('.' + base_domain) or
                    base_domain.endswith('.' + link_domain)):
                # Allow known content CDNs
                if 'cdn' not in link_domain:
                    continue
        
        # Dedup
        url_key = url.rstrip('/')
        if url_key in seen_urls:
            continue
        seen_urls.add(url_key)
        
        # Parse date
        date_str = None
        if article_date:
            try:
                dt = datetime.strptime(article_date.replace(',', ''), '%b %d %Y')
                date_str = dt.replace(tzinfo=timezone.utc).isoformat()
            except ValueError:
                pass
        
        articles.append({
            "title": title[:200],
            "link": url,
            "date": date_str or datetime.now(timezone.utc).isoformat(),
        })
        
        if len(articles) >= MAX_ARTICLES_PER_SOURCE:
            break
    
    return articles


def fetch_source(source: Dict[str, Any], cutoff: datetime) -> Dict[str, Any]:
    """Fetch a single Jina source with retry."""
    source_id = source["id"]
    name = source["name"]
    url = source["url"]
    priority = source.get("priority", False)
    topics = source.get("topics", [])
    
    for attempt in range(RETRY_COUNT + 1):
        try:
            jina_url = f"{JINA_BASE}{url}"
            headers = {
                "Accept": "text/plain",
                "User-Agent": "TechDigest/2.0",
            }
            req = Request(jina_url, headers=headers)
            
            with urlopen(req, timeout=TIMEOUT) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            
            # Check for error indicators in Jina response
            if "returned error 404" in raw[:500].lower():
                return {
                    "source_id": source_id, "source_type": "jina",
                    "name": name, "url": url, "priority": priority,
                    "topics": topics, "status": "error",
                    "attempts": attempt + 1, "error": "Target returned 404",
                    "count": 0, "articles": [],
                }
            
            articles = parse_jina_markdown(raw, url)
            
            # Filter by cutoff date
            recent_articles = []
            for a in articles:
                try:
                    article_date = datetime.fromisoformat(a["date"].replace("Z", "+00:00"))
                    if article_date >= cutoff:
                        recent_articles.append(a)
                except (ValueError, AttributeError):
                    recent_articles.append(a)
            
            # Tag with topics
            for a in recent_articles:
                a["topics"] = topics[:]
            
            return {
                "source_id": source_id, "source_type": "jina",
                "name": name, "url": url, "priority": priority,
                "topics": topics, "status": "ok",
                "attempts": attempt + 1, "count": len(recent_articles),
                "articles": recent_articles,
            }
            
        except HTTPError as e:
            if e.code == 429:
                # Rate limited - wait longer and retry
                if attempt < RETRY_COUNT:
                    time.sleep(5)
                    continue
            error_msg = f"HTTP {e.code}: {e.reason}"
            logging.debug(f"Attempt {attempt + 1} failed for {name}: {error_msg}")
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_DELAY * (2 ** attempt))
                continue
            return {
                "source_id": source_id, "source_type": "jina",
                "name": name, "url": url, "priority": priority,
                "topics": topics, "status": "error",
                "attempts": attempt + 1, "error": error_msg,
                "count": 0, "articles": [],
            }
        except Exception as e:
            error_msg = str(e)[:100]
            logging.debug(f"Attempt {attempt + 1} failed for {name}: {error_msg}")
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_DELAY * (2 ** attempt))
                continue
            return {
                "source_id": source_id, "source_type": "jina",
                "name": name, "url": url, "priority": priority,
                "topics": topics, "status": "error",
                "attempts": attempt + 1, "error": error_msg,
                "count": 0, "articles": [],
            }


def load_sources(defaults_dir: Path, config_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load Jina sources from unified configuration."""
    try:
        from config_loader import load_merged_sources
    except ImportError:
        import sys
        sys.path.append(str(Path(__file__).parent))
        from config_loader import load_merged_sources
    
    all_sources = load_merged_sources(defaults_dir, config_dir)
    
    jina_sources = []
    for source in all_sources:
        if source.get("type") == "jina" and source.get("enabled", True):
            jina_sources.append(source)
    
    logging.info(f"Loaded {len(jina_sources)} enabled Jina sources")
    return jina_sources


def main():
    parser = argparse.ArgumentParser(
        description="Fetch articles from websites via Jina Reader API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    _script_dir = Path(__file__).resolve().parent
    _default_defaults = _script_dir.parent / "config" / "defaults"
    
    parser.add_argument("--defaults", type=Path, default=_default_defaults)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--hours", type=int, default=48)
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--force", action="store_true")
    
    args = parser.parse_args()
    logger = setup_logging(args.verbose)
    
    if not args.output:
        fd, temp_path = tempfile.mkstemp(prefix="tech-news-digest-jina-", suffix=".json")
        os.close(fd)
        args.output = Path(temp_path)
    
    # Resume support
    if args.output.exists() and not args.force:
        try:
            age_seconds = time.time() - args.output.stat().st_mtime
            if age_seconds < 3600:
                with open(args.output) as f:
                    json.load(f)
                logger.info(f"Skipping (cached output exists): {args.output}")
                return 0
        except (json.JSONDecodeError, OSError):
            pass
    
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=args.hours)
        sources = load_sources(args.defaults, args.config)
        
        if not sources:
            logger.warning("No Jina sources found")
        
        logger.info(f"Fetching {len(sources)} Jina sources (window: {args.hours}h)")
        
        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(fetch_source, source, cutoff): source for source in sources}
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                
                if result["status"] == "ok":
                    logger.debug(f"✅ {result['name']}: {result['count']} articles")
                else:
                    logger.debug(f"❌ {result['name']}: {result.get('error', 'unknown')}")
        
        results.sort(key=lambda x: (not x.get("priority", False), -x.get("count", 0)))
        
        ok_count = sum(1 for r in results if r["status"] == "ok")
        total_articles = sum(r.get("count", 0) for r in results)
        
        output = {
            "generated": datetime.now(timezone.utc).isoformat(),
            "source_type": "jina",
            "defaults_dir": str(args.defaults),
            "config_dir": str(args.config) if args.config else None,
            "hours": args.hours,
            "sources_total": len(results),
            "sources_ok": ok_count,
            "total_articles": total_articles,
            "sources": results,
        }
        
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Done: {ok_count}/{len(results)} sources ok, "
                    f"{total_articles} articles → {args.output}")
        return 0
        
    except Exception as e:
        logger.error(f"💥 Jina fetch failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
