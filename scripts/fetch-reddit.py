#!/usr/bin/env python3
"""
Fetch Reddit posts from unified sources configuration.

Reads sources.json, filters Reddit sources, fetches posts via Reddit JSON API,
and outputs structured JSON with posts tagged by topics.

Usage:
    python3 fetch-reddit.py [--defaults DEFAULTS_DIR] [--config CONFIG_DIR] [--hours 48] [--output FILE] [--verbose] [--force] [--no-cache]

Environment:
    No API key required. Uses Reddit's public JSON API.
"""

import json
import sys
import os
import re
import argparse
import logging
import ssl
import time
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.request import Request, urlopen

_SSL_CTX = ssl.create_default_context()
from urllib.error import HTTPError, URLError

# Constants
MAX_WORKERS = 4
TIMEOUT = 30
RETRY_COUNT = 2
RETRY_DELAY = 3
USER_AGENT = "TechDigest/2.8 (bot; +https://github.com/draco-agent/tech-news-digest)"
RESUME_MAX_AGE_SECONDS = 3600  # 1 hour
JINA_BASE = "https://r.jina.ai/"


# ---------------------------------------------------------------------------
# Rate limiter (for Jina backend)
# ---------------------------------------------------------------------------

class RateLimiter:
    """Simple token-bucket rate limiter."""
    def __init__(self, qps: float):
        self._lock = threading.Lock()
        self._min_interval = 1.0 / qps
        self._last = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            wait_time = self._min_interval - (now - self._last)
            if wait_time > 0:
                time.sleep(wait_time)
            self._last = time.monotonic()


def setup_logging(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)


def load_reddit_sources(defaults_dir: Optional[Path], config_dir: Optional[Path]) -> List[Dict[str, Any]]:
    """Load Reddit sources from config, with user overrides."""
    sys.path.insert(0, str(Path(__file__).parent))
    from config_loader import load_merged_sources as load_sources
    
    all_sources = load_sources(defaults_dir, config_dir)
    reddit_sources = []
    for s in all_sources:
        if s.get('type') != 'reddit':
            continue
        if not s.get('enabled', True):
            continue
        if not s.get('subreddit'):
            logging.warning(f"Reddit source {s.get('id')} missing subreddit, skipping")
            continue
        reddit_sources.append(s)
    
    return reddit_sources


def fetch_subreddit(source: Dict[str, Any], cutoff: datetime) -> Dict[str, Any]:
    """Fetch posts from a subreddit using Reddit's JSON API."""
    source_id = source['id']
    subreddit = source['subreddit']
    sort = source.get('sort', 'hot')
    limit = source.get('limit', 25)
    min_score = source.get('min_score', 0)
    priority = source.get('priority', False)
    topics = source.get('topics', [])
    name = source.get('name', f'r/{subreddit}')
    
    url = f"https://www.reddit.com/r/{subreddit}/{sort}.json?limit={limit}&raw_json=1"
    
    for attempt in range(RETRY_COUNT + 1):
        try:
            req = Request(url, headers={
                'User-Agent': USER_AGENT,
                'Accept': 'text/html,application/json',
                'Accept-Language': 'en-US,en;q=0.9',
            })
            
            with urlopen(req, timeout=TIMEOUT, context=_SSL_CTX) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            
            articles = []
            children = data.get('data', {}).get('children', [])
            
            for child in children:
                post = child.get('data', {})
                if not post:
                    continue
                
                # Parse timestamp
                created_utc = post.get('created_utc', 0)
                post_time = datetime.fromtimestamp(created_utc, tz=timezone.utc)
                
                # Filter by time
                if post_time < cutoff:
                    continue
                
                # Filter by score
                score = post.get('score', 0)
                if score < min_score:
                    continue
                
                # Skip stickied/pinned posts
                if post.get('stickied', False):
                    continue
                
                # Get the external URL (if it's a link post) vs self post
                permalink = f"https://www.reddit.com{post.get('permalink', '')}"
                external_url = post.get('url', '')
                is_self = post.get('is_self', True)
                
                # If it's a self post or URL points to reddit, use permalink
                if is_self or 'reddit.com' in external_url or 'redd.it' in external_url:
                    link = permalink
                    external_url = None
                else:
                    link = external_url
                
                title = post.get('title', '').strip()
                if not title:
                    continue
                
                flair = post.get('link_flair_text', '')
                num_comments = post.get('num_comments', 0)
                upvote_ratio = post.get('upvote_ratio', 0)
                
                articles.append({
                    "title": title,
                    "link": link,
                    "reddit_url": permalink,
                    "external_url": external_url,
                    "date": post_time.isoformat(),
                    "score": score,
                    "num_comments": num_comments,
                    "flair": flair,
                    "is_self": is_self,
                    "topics": topics[:],
                    "metrics": {
                        "score": score,
                        "num_comments": num_comments,
                        "upvote_ratio": upvote_ratio
                    }
                })
            
            return {
                "source_id": source_id,
                "source_type": "reddit",
                "name": name,
                "subreddit": subreddit,
                "sort": sort,
                "priority": priority,
                "topics": topics,
                "status": "ok",
                "attempts": attempt + 1,
                "count": len(articles),
                "articles": articles,
            }
        
        except HTTPError as e:
            if e.code == 429:
                logging.warning(f"Rate limit for r/{subreddit}, attempt {attempt + 1}")
                if attempt < RETRY_COUNT:
                    time.sleep(10)
                    continue
            elif e.code == 403:
                logging.warning(f"r/{subreddit} is private or quarantined")
                return {
                    "source_id": source_id,
                    "source_type": "reddit",
                    "name": name,
                    "subreddit": subreddit,
                    "status": "error",
                    "error": f"HTTP {e.code}: Forbidden",
                    "count": 0,
                    "articles": [],
                }
            error_msg = f"HTTP {e.code}"
            logging.warning(f"Error fetching r/{subreddit}: {error_msg}")
        except (URLError, OSError) as e:
            error_msg = str(e)
            logging.warning(f"Network error for r/{subreddit}: {error_msg}")
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Unexpected error for r/{subreddit}: {error_msg}")
        
        if attempt < RETRY_COUNT:
            time.sleep(RETRY_DELAY)
    
    return {
        "source_id": source_id,
        "source_type": "reddit",
        "name": name,
        "subreddit": subreddit,
        "status": "error",
        "error": error_msg,
        "count": 0,
        "articles": [],
    }


def fetch_subreddit_jina(source: Dict[str, Any], cutoff: datetime, limiter: RateLimiter) -> Dict[str, Any]:
    """Fetch posts from a subreddit using Reddit's Atom RSS feed as a fallback.
    
    When the JSON API is blocked (403), Reddit's RSS (.rss) endpoint often still works.
    This fetches the Atom feed and parses it using xml.etree.ElementTree.
    """
    source_id = source['id']
    subreddit = source['subreddit']
    sort = source.get('sort', 'hot')
    min_score = source.get('min_score', 0)
    priority = source.get('priority', False)
    topics = source.get('topics', [])
    name = source.get('name', f'r/{subreddit}')
    
    url = f"https://www.reddit.com/r/{subreddit}/{sort}/.rss?limit=25"
    
    for attempt in range(RETRY_COUNT + 1):
        try:
            req = Request(url, headers={
                'User-Agent': USER_AGENT,
                'Accept': 'application/atom+xml,application/xml,text/xml',
            })
            with urlopen(req, timeout=TIMEOUT, context=_SSL_CTX) as resp:
                xml_data = resp.read().decode('utf-8', errors='replace')
            
            if not xml_data or len(xml_data) < 100:
                error_msg = "RSS feed returned empty/too-short content"
                if attempt < RETRY_COUNT:
                    time.sleep(RETRY_DELAY)
                    continue
                return {
                    "source_id": source_id,
                    "source_type": "reddit",
                    "name": name,
                    "subreddit": subreddit,
                    "sort": sort,
                    "priority": priority,
                    "topics": topics,
                    "status": "error",
                    "error": error_msg,
                    "attempts": attempt + 1,
                    "count": 0,
                    "articles": [],
                }
            
            articles = _parse_reddit_rss(xml_data, subreddit, topics, cutoff, min_score)
            
            return {
                "source_id": source_id,
                "source_type": "reddit",
                "name": name,
                "subreddit": subreddit,
                "sort": sort,
                "priority": priority,
                "topics": topics,
                "status": "ok",
                "attempts": attempt + 1,
                "count": len(articles),
                "articles": articles,
                "backend": "rss",
            }
        
        except HTTPError as e:
            error_msg = f"RSS HTTP {e.code}: {e.reason}"
            logging.warning(f"RSS error for r/{subreddit}: {error_msg}")
        except Exception as e:
            error_msg = str(e)[:100]
            logging.warning(f"RSS fetch failed for r/{subreddit}: {error_msg}")
        
        if attempt < RETRY_COUNT:
            time.sleep(RETRY_DELAY)
    
    return {
        "source_id": source_id,
        "source_type": "reddit",
        "name": name,
        "subreddit": subreddit,
        "sort": sort,
        "priority": priority,
        "topics": topics,
        "status": "error",
        "error": error_msg,
        "attempts": RETRY_COUNT + 1,
        "count": 0,
        "articles": [],
    }


def _parse_reddit_rss(xml_data: str, subreddit: str, topics: list,
                       cutoff: datetime, min_score: int = 0) -> list:
    """Parse Reddit Atom RSS feed into article dicts."""
    import xml.etree.ElementTree as ET
    
    articles = []
    
    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as e:
        logging.warning(f"Failed to parse RSS XML for r/{subreddit}: {e}")
        return articles
    
    # Atom namespace
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    
    for entry in root.findall('atom:entry', ns):
        # Get title
        title_el = entry.find('atom:title', ns)
        if title_el is None or not title_el.text:
            continue
        title = title_el.text.strip()
        
        # Get link
        link_el = entry.find('atom:link', ns)
        link = ''
        if link_el is not None:
            link = link_el.get('href', '')
        
        # Get published date
        published_el = entry.find('atom:published', ns)
        post_time = None
        if published_el is not None and published_el.text:
            try:
                # Atom dates are ISO 8601: 2026-04-30T17:37:23+00:00
                date_str = published_el.text
                if date_str.endswith('Z'):
                    date_str = date_str[:-1] + '+00:00'
                post_time = datetime.fromisoformat(date_str)
            except (ValueError, TypeError):
                pass
        
        # Filter by time cutoff
        if post_time and post_time < cutoff:
            continue
        
        # Get author
        author_el = entry.find('atom:author/atom:name', ns)
        author = author_el.text if author_el is not None else ''
        
        # Get entry ID (e.g. t3_1t038g7)
        id_el = entry.find('atom:id', ns)
        entry_id = id_el.text if id_el is not None else ''
        
        # Extract external URL from content HTML
        content_el = entry.find('atom:content', ns)
        external_url = None
        is_self = True
        if content_el is not None and content_el.text:
            content_html = content_el.text
            # Find [link] href — this is the external link
            # Pattern: <span><a href="URL">[link]</a></span>
            link_match = re.search(r'<a href="([^"]+)"\s*>\[link\]</a>', content_html)
            if link_match:
                ext = link_match.group(1)
                # If it's not a reddit link, it's an external URL
                if 'reddit.com' not in ext and 'redd.it' not in ext:
                    external_url = ext
                    is_self = False
        
        # Skip if no meaningful link
        if not link:
            continue
            
        permalink = link
        
        articles.append({
            "title": title,
            "link": external_url or permalink,
            "reddit_url": permalink,
            "external_url": external_url,
            "date": post_time.isoformat() if post_time else datetime.now(timezone.utc).isoformat(),
            "score": 0,  # RSS doesn't include score
            "num_comments": 0,
            "flair": "",
            "is_self": is_self,
            "topics": topics[:],
            "metrics": {
                "score": 0,
                "num_comments": 0,
                "upvote_ratio": 0,
            }
        })
    
    return articles


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch Reddit posts from configured subreddits.\n"
                    "Uses Reddit's public JSON API (no authentication required).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    python3 fetch-reddit.py --defaults config/defaults --output /tmp/td-reddit.json --verbose
    python3 fetch-reddit.py --defaults config/defaults --config ~/workspace/config --hours 48
    """
    )
    parser.add_argument('--defaults', type=Path, default=Path('config/defaults'),
                       help='Default config directory')
    parser.add_argument('--config', type=Path, default=None,
                       help='User config directory (overrides defaults)')
    parser.add_argument('--hours', type=int, default=48,
                       help='How many hours back to fetch (default: 48)')
    parser.add_argument('--output', type=Path, default=None,
                       help='Output JSON file path')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--force', action='store_true',
                       help='Force fetch even if cached output exists')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable all caching')
    
    args = parser.parse_args()
    logger = setup_logging(args.verbose)
    
    # Auto-generate output path if not specified
    if not args.output:
        fd, temp_path = tempfile.mkstemp(prefix="tech-news-digest-reddit-", suffix=".json")
        os.close(fd)
        args.output = Path(temp_path)
    
    # Resume support
    if not args.force and args.output.exists():
        try:
            age = time.time() - args.output.stat().st_mtime
            if age < RESUME_MAX_AGE_SECONDS:
                with open(args.output) as f:
                    existing = json.load(f)
                if existing.get('subreddits'):
                    logger.info(f"⏭️  Skipping fetch: {args.output} is {age:.0f}s old (< {RESUME_MAX_AGE_SECONDS}s). Use --force to override.")
                    print(f"Output (cached): {args.output}")
                    return 0
        except (json.JSONDecodeError, KeyError):
            pass
    
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=args.hours)
        
        # Load sources
        if args.config and args.defaults == Path("config/defaults") and not args.defaults.exists():
            sources = load_reddit_sources(args.config, None)
        else:
            sources = load_reddit_sources(args.defaults, args.config)
        
        if not sources:
            logger.warning("No Reddit sources found or all disabled")
            output = {
                "source": "reddit",
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "subreddits": [],
                "skipped_reason": "No Reddit sources configured"
            }
            with open(args.output, "w") as f:
                json.dump(output, f, indent=2)
            print(f"Output (empty): {args.output}")
            return 0
        
        logger.info(f"📡 Fetching {len(sources)} subreddits (cutoff: {cutoff.strftime('%Y-%m-%d %H:%M')} UTC)")
        
        results = []
        total_posts = 0
        
        # Phase 1: Try direct Reddit JSON API
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(fetch_subreddit, source, cutoff): source for source in sources}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                total_posts += result.get('count', 0)
        
        ok_count = sum(1 for r in results if r['status'] == 'ok')
        failed = [r for r in results if r['status'] != 'ok']
        
        # Phase 2: Fallback to RSS feed for failed subreddits
        if failed:
            logger.info(f"🔄 {len(failed)} subreddits failed via JSON API, trying RSS feed fallback...")
            rss_results = []
            for r in failed:
                # Find the matching source
                source = next((s for s in sources if s['id'] == r['source_id']), None)
                if not source:
                    continue
                rss_result = fetch_subreddit_jina(source, cutoff, limiter=None)
                if rss_result['status'] == 'ok' and rss_result['count'] > 0:
                    logger.info(f"  ✅ r/{source['subreddit']}: {rss_result['count']} posts via RSS")
                    rss_results.append(rss_result)
                else:
                    logger.warning(f"  ❌ r/{source['subreddit']}: RSS also failed ({rss_result.get('error', 'no results')})")
            
            # Replace failed results with successful RSS results
            if rss_results:
                failed_ids = {r['source_id'] for r in failed}
                results = [r for r in results if r['source_id'] not in failed_ids] + rss_results
                total_posts = sum(r.get('count', 0) for r in results)
        
        ok_count = sum(1 for r in results if r['status'] == 'ok')
        
        output = {
            "source": "reddit",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "defaults_dir": str(args.defaults),
            "config_dir": str(args.config) if args.config else None,
            "hours": args.hours,
            "cutoff": cutoff.isoformat(),
            "subreddits_total": len(results),
            "subreddits_ok": ok_count,
            "total_posts": total_posts,
            "subreddits": results
        }
        
        json_str = json.dumps(output, ensure_ascii=False, indent=2)
        with open(args.output, "w", encoding='utf-8') as f:
            f.write(json_str)
        
        logger.info(f"✅ Fetched {ok_count}/{len(results)} subreddits, {total_posts} posts")
        print(f"Output: {args.output}")
        return 0
    
    except Exception as e:
        logger.error(f"💥 Reddit fetch failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
