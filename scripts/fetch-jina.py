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

# Navigation / category / boilerplate title patterns to reject
NAV_TITLE_KEYWORDS = {
    # Site chrome
    'skip to', 'go to', 'jump to', 'scroll to',
    'home', 'back', 'menu', 'navigation', 'sitemap',
    'login', 'log in', 'sign in', 'sign up', 'register',
    'subscribe', 'newsletter', 'email us', 'contact us',
    'privacy', 'terms of', 'cookie', 'accessibility', 'copyright',
    # CTAs
    'book a demo', 'try it free', 'get started', 'start here',
    'upgrade to', 'pricing', 'free trial', 'save ', 'subscribe now',
    'read more', 'learn more', 'find out', 'view all', 'see all',
    'join ieee',
    # Generic sections
    'about', 'company', 'business', 'developers', 'enterprise',
    'research index', 'research overview', 'view research',
    'latest crypto news', 'latest news',
    'follow us', 'share this', 'social media',
    'pause media', 'play media',
    # Category/topic pages (not articles)
    'centers & labs', 'research publications', 'research partners',
    'software progress', 'open models', 'leading companies',
    'climate tech', 'consumer electronics', 'biomedical',
    'scale data engine', 'scale genai platform',
    'fellowship program', 'student affinity',
    # Substack/newsletter boilerplate
    'not boring by', 'deep dive', 'deep dives', 'vertical integrator',
    'ai job board', 'view blog',
    # Personal site nav
    'my favourite', 'prototypes', 'speaking', 'writing',
    'more than ', 'more than 50k',
}
# Short exact-match titles to reject (case-insensitive)
NAV_TITLE_EXACT = {
    'research', 'about', 'enterprise', 'newsletter', 'resources',
    'premium', 'explore', 'blog', 'careers', 'team', 'press',
    'products', 'solutions', 'services', 'support', 'help',
    'community', 'events', 'webinars', 'podcast', 'docs',
    'api', 'status', 'security', 'partners',
    # Epoch AI nav pages
    'data insights', 'frontier data centers', 'chip owners',
    'chip sales', 'frontiermath: open problems', 'frontiermath: tiers 1-4',
    # IEEE nav
    'engineering resources', 'special reports', 'top programming languages',
    'current issue', 'the institute', 'the institute archive',
    # Stanford HAI nav
    'executive and professional education', 'government and policymakers',
    'stanford students', 'student opportunities', 'ai index report',
    'global vibrancy tool', 'policymaker education',
    # Scale AI nav
    'us public sector', 'global public sector', 'modern slavery statement',
    'data labeling', 'ml model training', 'diffusion models',
    'guide to ai for ecommerce', 'computer vision applications',
    'large language models',
    # Every nav/columns
    'context window', 'source code',
    # Epoch AI nav
    'see more insights',
    # The Block nav
    'press release',
}

# URL patterns that indicate NON-article links
URL_SKIP_SUBSTRINGS = [
    '/cdn-cgi', '/privacy', '/terms', '/login', '/signup', '/signin',
    '/about', '/contact', '/subscribe', '/search',
    '#', 'javascript:', 'mailto:', '/feed', '/rss',
    '.xml', '.json', '/sitemap', '/cdn/', '/static/',
    'twitter.com', 'x.com', 'facebook.com', 'linkedin.com',
    'github.com/login', 'github.com/feedback',
    'chromewebstore.google.com',
    '/start-here', '/prototyping', '/speaking',
    '/membership', '/join', '/donate',
]
# URL file extensions that are NOT articles
URL_SKIP_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.ico',
    '.mp4', '.mp3', '.wav', '.avi', '.mov',
    '.pdf', '.zip', '.tar', '.gz',
    '.css', '.js', '.woff', '.ttf', '.eot',
}
# URL path segments that are category/topic pages, not articles
URL_CATEGORY_SEGMENTS = [
    '/topic/', '/topics/', '/category/', '/categories/', '/tag/', '/tags/',
    '/centers-', '/centers/', '/type/', '/filter/',
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
    """Check if a title looks like navigation boilerplate or a category page."""
    title_stripped = title.strip()
    title_lower = title_stripped.lower()
    
    # Reject titles with image markers that survived cleaning
    if title_stripped.startswith('![') or title_stripped.startswith('['):
        return True
    
    # Reject very short titles (< 10 chars after trimming)
    if len(title_stripped) < 10:
        return True
    
    # Reject single-person-name titles: 2-3 words, all capitalized, no verbs
    # Pattern: "Firstname Lastname" or "Firstname Middlename Lastname,"
    # Heuristic: if it's ≤3 words, starts with capital, ends with optional comma
    # This catches "Shana Lynch", "Curtis Langlotz,", "Mike Taylor"
    words = title_stripped.replace(',', '').split()
    if 1 <= len(words) <= 3 and all(w[0].isupper() for w in words if w):
        return True
    
    # Exact match against short nav words
    if title_lower in NAV_TITLE_EXACT:
        return True
    
    # Keyword substring match
    for kw in NAV_TITLE_KEYWORDS:
        if kw in title_lower:
            return True
    
    return False


def is_article_url(url: str) -> bool:
    """Check if a URL looks like an actual article page (not nav/image/category)."""
    if not url:
        return False
    
    url_lower = url.lower()
    
    # 1. Reject by substring patterns
    for pattern in URL_SKIP_SUBSTRINGS:
        if pattern in url_lower:
            return False
    
    # 2. Reject by file extension (image, media, binary files)
    path_part = urlparse(url_lower).path.rstrip('/')
    for ext in URL_SKIP_EXTENSIONS:
        if path_part.endswith(ext):
            return False
    
    # 3. Reject CDN image URLs (common pattern: cdn.somesite.com/images/...)
    parsed = urlparse(url)
    host_lower = parsed.netloc.lower()
    if any(seg in host_lower for seg in ('cdn.', 'images.', 'image.', 'img.', 'static.', 'assets.', 'media.')):
        return False
    
    # 4. Reject category/topic listing pages
    for cat_seg in URL_CATEGORY_SEGMENTS:
        if cat_seg in url_lower:
            return False
    
    path = parsed.path.rstrip('/')
    
    # Must have a meaningful path
    if not path or path == '/':
        return False
    
    parts = [p for p in path.split('/') if p]
    
    # Path with 2+ segments, last segment looks like a slug
    if len(parts) >= 2:
        last_part = parts[-1]
        if len(last_part) >= 5 and ('-' in last_part or '_' in last_part):
            return True
        # Date-based paths like /2026/05/01/slug or /blog/1234
        if re.match(r'^\d{4}$', parts[-2] if len(parts) >= 2 else ''):
            return True
    
    # Single-segment slug paths (e.g., /contextual-retrieval)
    if len(parts) == 1 and len(parts[0]) >= 8 and ('-' in parts[0] or '_' in parts[0]):
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
        
        # Clean title — aggressive stripping of image/markdown artifacts
        title = raw_title
        # Remove image markers: ![Alt text] → empty
        title = re.sub(r'!\[.*?\]', '', title)
        # Remove standalone markdown image references
        title = re.sub(r'^\s*\[Image\s*\d*.*?\]', '', title)
        # Remove heading markers (#### etc)
        title = re.sub(r'#{1,6}\s+', ' ', title)
        # Remove bullet markers (• or -)
        title = re.sub(r'^\s*[•\-]\s*', '', title)
        # Remove leading bracket artifacts like [icon]
        if re.match(r'^\[[^\]]*\]$', title.strip()):
            continue
        # Remove trailing "Release DATE X min read" pattern (OpenAI)
        title = re.sub(r'\s+Release\s+\w{3}\s+\d{1,2},?\s+\d{4}\s+\d+\s+min\s+read$', '', title)
        # Remove trailing date patterns like "Apr 23, 2026 12 min read"
        title = re.sub(r'\s+\w{3}\s+\d{1,2},?\s+\d{4}\s+\d+\s+min\s+read$', '', title)
        # Remove "Featured ##" prefix
        title = re.sub(r'^Featured\s+##\s*', '', title)
        # Remove trailing " — TAG, TAG" patterns (Philipp Schmid)
        title = re.sub(r'\s+—\s+\w.*$', '', title)
        # Remove date prefix like "Feb 26, 2026 " (Cursor)
        title = re.sub(r'^\w{3}\s+\d{1,2},?\s+\d{4}\s+', '', title)
        title = re.sub(r'\s+', ' ', title).strip()
        # Fix weird intra-word spaces from Jina parsing: "C ash" → "Cash", "B eyond" → "Beyond"
        # Pattern: single uppercase letter + space + lowercase word, but NOT "A" (article)
        title = re.sub(r'\b(?!A\b)([A-Z])\s+([a-z]{2,})\b', r'\1\2', title)
        
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
        
        # Domain check — reject cross-domain links (not same site)
        link_domain = urlparse(url).netloc.lower().replace("www.", "")
        if link_domain and base_domain:
            if not (link_domain == base_domain or
                    link_domain.endswith('.' + base_domain) or
                    base_domain.endswith('.' + link_domain)):
                # NOT same domain → reject (handles CDN images too)
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
