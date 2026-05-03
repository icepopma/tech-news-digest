#!/usr/bin/env python3
"""
Fetch Twitter/X posts from KOL accounts using X API.

Reads sources.json, filters Twitter sources, fetches recent posts using
either the official X API v2, twitterapi.io, GetXAPI, or Jina Reader, and outputs structured JSON.

Usage:
    python3 fetch-twitter.py [--config CONFIG_DIR] [--hours 48] [--output FILE] [--verbose]
    python3 fetch-twitter.py --backend twitterapiio  # force twitterapi.io backend
    python3 fetch-twitter.py --backend jina          # free Jina Reader backend (no key)

Environment:
    TWITTER_API_BACKEND - Backend selection: "auto" (default), "getxapi", "twitterapiio", "official", or "jina"
                        Auto priority: getxapi ($0.001/call) > twitterapi.io (~$5/mo) > official X API > jina (free)
    GETX_API_KEY        - GetXAPI API key (preferred backend, $0.001 per call)
    TWITTERAPI_IO_KEY   - twitterapi.io API key (alternative backend, ~$5/month)
    X_BEARER_TOKEN      - Twitter/X official API v2 bearer token (fallback)
"""

import json
import sys
import os
import argparse
import logging
import time
import tempfile
import re
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from urllib.parse import urlencode, quote
from pathlib import Path
from typing import Dict, List, Any, Optional

TIMEOUT = 30
MAX_WORKERS = 5  # Lower for API rate limits
RETRY_COUNT = 2
RETRY_DELAY = 2.0
MAX_TWEETS_PER_USER = 20
ID_CACHE_PATH = "/tmp/tech-news-digest-twitter-id-cache.json"
ID_CACHE_TTL_DAYS = 7

# Twitter API v2 endpoints
OFFICIAL_API_BASE = "https://api.x.com/2"
USER_LOOKUP_ENDPOINT = f"{OFFICIAL_API_BASE}/users/by"

# twitterapi.io endpoints
TWITTERAPIIO_BASE = "https://api.twitterapi.io"
GETXAPI_BASE = "https://api.getxapi.com"


def setup_logging(verbose: bool) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def clean_tweet_text(text: str) -> str:
    """Clean tweet text for better display."""
    # Remove markdown images: [![alt](url)](link) and ![alt](url)
    text = re.sub(r'\[!\[.*?\]\(.*?\)\]\(.*?\)', '', text)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # Remove markdown links but keep text: [text](url) → text
    text = re.sub(r'\[([^\]]*)\]\(.*?\)', r'\1', text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Truncate if too long
    if len(text) > 280:
        text = text[:277] + "..."
    return text


# ---------------------------------------------------------------------------
# Rate limiting
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


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------

class TwitterBackend(ABC):
    """Base class for Twitter API backends."""

    @staticmethod
    def _make_result(source, articles, attempt):
        return {
            "source_id": source["id"],
            "source_type": "twitter",
            "name": source["name"],
            "handle": source["handle"].lstrip('@'),
            "priority": source["priority"],
            "topics": source["topics"],
            "status": "ok",
            "attempts": attempt + 1,
            "count": len(articles),
            "articles": articles,
        }

    @staticmethod
    def _make_error(source, error_msg, attempt):
        return {
            "source_id": source["id"],
            "source_type": "twitter",
            "name": source["name"],
            "handle": source["handle"].lstrip('@'),
            "priority": source["priority"],
            "topics": source["topics"],
            "status": "error",
            "attempts": attempt + 1,
            "error": error_msg,
            "count": 0,
            "articles": [],
        }

    @abstractmethod
    def fetch_all(self, sources: List[Dict[str, Any]], cutoff: datetime) -> List[Dict[str, Any]]:
        """Fetch tweets for all sources. Returns list of source result dicts."""


class OfficialBackend(TwitterBackend):
    """Official X API v2 backend (existing logic)."""

    def __init__(self, bearer_token: str, no_cache: bool = False):
        self.bearer_token = bearer_token
        self.no_cache = no_cache

    # -- ID cache helpers --

    @staticmethod
    def _load_id_cache() -> Dict[str, Any]:
        try:
            with open(ID_CACHE_PATH, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _save_id_cache(cache: Dict[str, Any]) -> None:
        try:
            with open(ID_CACHE_PATH, 'w') as f:
                json.dump(cache, f)
        except Exception as e:
            logging.warning(f"Failed to save ID cache: {e}")

    def _batch_resolve_user_ids(self, handles: List[str]) -> Dict[str, str]:
        now = time.time()
        cache = {} if self.no_cache else self._load_id_cache()
        ttl_seconds = ID_CACHE_TTL_DAYS * 86400

        result: Dict[str, str] = {}
        to_resolve: List[str] = []
        for handle in handles:
            key = handle.lower()
            entry = cache.get(key)
            if entry and (now - entry.get("ts", 0)) < ttl_seconds:
                result[key] = entry["id"]
            else:
                to_resolve.append(handle)

        if to_resolve:
            logging.info(f"Batch resolving {len(to_resolve)} usernames (cached: {len(result)})")
            headers = {
                "Authorization": f"Bearer {self.bearer_token}",
                "User-Agent": "TechDigest/2.0"
            }
            for i in range(0, len(to_resolve), 100):
                batch = to_resolve[i:i+100]
                url = f"{USER_LOOKUP_ENDPOINT}?{urlencode({'usernames': ','.join(batch)})}"
                try:
                    req = Request(url, headers=headers)
                    with urlopen(req, timeout=TIMEOUT) as resp:
                        data = json.loads(resp.read().decode())

                    if 'data' in data:
                        for user in data['data']:
                            key = user['username'].lower()
                            result[key] = user['id']
                            cache[key] = {"id": user['id'], "ts": now}

                    if 'errors' in data:
                        for err in data['errors']:
                            logging.warning(f"User lookup error: {err.get('detail', err)}")

                except Exception as e:
                    logging.error(f"Batch user lookup failed: {e}")
                    for handle in batch:
                        try:
                            fallback_url = f"{USER_LOOKUP_ENDPOINT}?{urlencode({'usernames': handle})}"
                            req = Request(fallback_url, headers=headers)
                            with urlopen(req, timeout=TIMEOUT) as resp:
                                fallback_data = json.loads(resp.read().decode())
                            if 'data' in fallback_data and fallback_data['data']:
                                key = handle.lower()
                                result[key] = fallback_data['data'][0]['id']
                                cache[key] = {"id": result[key], "ts": now}
                        except Exception as e2:
                            logging.warning(f"Individual lookup failed for @{handle}: {e2}")

            if not self.no_cache:
                self._save_id_cache(cache)
        else:
            logging.info(f"All {len(result)} usernames resolved from cache")

        return result

    @staticmethod
    def _parse_date(date_str: str) -> Optional[datetime]:
        try:
            if date_str.endswith('Z'):
                date_str = date_str[:-1] + '+00:00'
            return datetime.fromisoformat(date_str)
        except (ValueError, TypeError):
            logging.debug(f"Failed to parse Twitter date: {date_str}")
            return None

    def _fetch_user_tweets(self, source: Dict[str, Any], cutoff: datetime,
                           user_id: Optional[str] = None) -> Dict[str, Any]:
        handle = source["handle"].lstrip('@')
        topics = source["topics"]

        for attempt in range(RETRY_COUNT + 1):
            try:
                params = {
                    "max_results": min(MAX_TWEETS_PER_USER, 100),
                    "tweet.fields": "created_at,public_metrics,context_annotations,referenced_tweets",
                    "expansions": "author_id",
                    "user.fields": "verified,public_metrics"
                }

                if not user_id:
                    user_url = f"{USER_LOOKUP_ENDPOINT}?{urlencode({'usernames': handle})}"
                    headers = {
                        "Authorization": f"Bearer {self.bearer_token}",
                        "User-Agent": "TechDigest/2.0"
                    }
                    req = Request(user_url, headers=headers)
                    with urlopen(req, timeout=TIMEOUT) as resp:
                        user_data = json.loads(resp.read().decode())
                    if 'data' not in user_data or not user_data['data']:
                        raise ValueError(f"User not found: {handle}")
                    user_id = user_data['data'][0]['id']

                headers = {
                    "Authorization": f"Bearer {self.bearer_token}",
                    "User-Agent": "TechDigest/2.0"
                }

                time.sleep(0.3)

                tweets_url = f"{OFFICIAL_API_BASE}/users/{user_id}/tweets?{urlencode(params)}"
                req = Request(tweets_url, headers=headers)

                with urlopen(req, timeout=TIMEOUT) as resp:
                    tweets_data = json.loads(resp.read().decode())

                articles = []
                if 'data' in tweets_data:
                    for tweet in tweets_data['data']:
                        created_at = self._parse_date(tweet.get('created_at', ''))
                        if not created_at or created_at < cutoff:
                            continue

                        text = tweet.get('text', '')
                        if text.startswith('RT @'):
                            continue
                        referenced = tweet.get('referenced_tweets', [])
                        if any(ref.get('type') == 'replied_to' for ref in referenced):
                            continue

                        articles.append({
                            "title": clean_tweet_text(text),
                            "link": f"https://twitter.com/{handle}/status/{tweet['id']}",
                            "date": created_at.isoformat(),
                            "topics": topics[:],
                            "metrics": tweet.get('public_metrics', {}),
                            "tweet_id": tweet['id'],
                            "source_type": "twitter",
                            "source_name": f"@{handle}",
                        })

                return self._make_result(source, articles, attempt)

            except HTTPError as e:
                if e.code == 429:
                    error_msg = "Rate limit exceeded"
                    logging.warning(f"Rate limit hit for @{handle}, attempt {attempt + 1}")
                    if attempt < RETRY_COUNT:
                        time.sleep(60)
                        continue
                else:
                    error_msg = f"HTTP {e.code}: {e.reason}"

            except Exception as e:
                error_msg = str(e)[:100]
                logging.debug(f"Attempt {attempt + 1} failed for @{handle}: {error_msg}")

            if attempt < RETRY_COUNT:
                time.sleep(RETRY_DELAY * (2 ** attempt))
                continue

            return self._make_error(source, error_msg, attempt)

    def fetch_all(self, sources: List[Dict[str, Any]], cutoff: datetime) -> List[Dict[str, Any]]:
        all_handles = [s["handle"].lstrip('@') for s in sources]
        user_id_map = self._batch_resolve_user_ids(all_handles)

        results: List[Dict[str, Any]] = []
        total = len(sources)
        done = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {}
            for source in sources:
                handle = source["handle"].lstrip('@')
                resolved_id = user_id_map.get(handle.lower())
                futures[pool.submit(self._fetch_user_tweets, source, cutoff, resolved_id)] = source

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                done += 1
                if result["status"] == "ok":
                    logging.info(f"[{done}/{total}] ✅ @{result['handle']}: {result['count']} tweets"
                                 + (f" (top: {result['articles'][0]['metrics']['like_count']}❤️)" if result.get('articles') else ""))
                else:
                    logging.warning(f"[{done}/{total}] ❌ @{result['handle']}: {result.get('error','unknown')}")

        return results


class TwitterApiIoBackend(TwitterBackend):
    """twitterapi.io backend."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._limiter = RateLimiter(qps=5)

    @staticmethod
    def _parse_date(date_str: str) -> Optional[datetime]:
        """Parse twitterapi.io date format: 'Tue Dec 10 07:00:30 +0000 2024'."""
        try:
            return datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")
        except (ValueError, TypeError):
            logging.debug(f"Failed to parse twitterapi.io date: {date_str}")
            return None

    def _parse_tweets_page(self, tweets: list, handle: str, topics: list, cutoff: datetime) -> list:
        """Parse a page of tweets into article dicts."""
        articles = []
        for tweet in tweets:
            # Skip retweets
            if tweet.get("retweeted_tweet"):
                continue
            created_at = self._parse_date(tweet.get("createdAt", ""))
            if not created_at or created_at < cutoff:
                continue

            text = tweet.get("text", "")
            if text.startswith("RT @"):
                continue

            tweet_id = tweet.get("id", "")
            link = tweet.get("url") or f"https://twitter.com/{handle}/status/{tweet_id}"

            articles.append({
                "title": clean_tweet_text(text),
                "link": link,
                "date": created_at.isoformat(),
                "topics": topics[:],
                "metrics": {
                    "like_count": tweet.get("likeCount", 0),
                    "retweet_count": tweet.get("retweetCount", 0),
                    "reply_count": tweet.get("replyCount", 0),
                    "quote_count": tweet.get("quoteCount", 0),
                    "impression_count": tweet.get("viewCount", 0),
                },
                "tweet_id": tweet_id,
                "source_type": "twitter",
                "source_name": f"@{handle}",
            })
        return articles

    def _fetch_user_tweets(self, source: Dict[str, Any], cutoff: datetime) -> Dict[str, Any]:
        handle = source["handle"].lstrip('@')
        topics = source["topics"]

        for attempt in range(RETRY_COUNT + 1):
            try:
                params = urlencode({
                    "userName": handle,
                    "includeReplies": "false",
                })
                url = f"{TWITTERAPIIO_BASE}/twitter/user/last_tweets?{params}"
                headers = {
                    "X-API-Key": self.api_key,
                    "User-Agent": "TechDigest/2.0",
                }

                self._limiter.wait()

                req = Request(url, headers=headers)
                with urlopen(req, timeout=TIMEOUT) as resp:
                    raw = json.loads(resp.read().decode())

                # API wraps response in {"data": {...}} envelope
                data = raw.get("data", raw)

                articles = self._parse_tweets_page(
                    data.get("tweets", []), handle, topics, cutoff
                )

                # Pagination: fetch one more page if available and all tweets still in window
                has_next = data.get("has_next_page", False)
                next_cursor = data.get("next_cursor")
                if has_next and next_cursor and articles:
                    oldest = min(a["date"] for a in articles)
                    if oldest >= cutoff.isoformat():
                        self._limiter.wait()
                        page2_params = urlencode({
                            "userName": handle,
                            "includeReplies": "false",
                            "cursor": next_cursor,
                        })
                        page2_url = f"{TWITTERAPIIO_BASE}/twitter/user/last_tweets?{page2_params}"
                        req2 = Request(page2_url, headers=headers)
                        with urlopen(req2, timeout=TIMEOUT) as resp2:
                            raw2 = json.loads(resp2.read().decode())
                        data2 = raw2.get("data", raw2)
                        articles.extend(self._parse_tweets_page(
                            data2.get("tweets", []), handle, topics, cutoff
                        ))
                        has_next = data2.get("has_next_page", False)

                # Truncation warning
                if has_next and articles:
                    oldest = min(a["date"] for a in articles)
                    if oldest >= cutoff.isoformat():
                        logging.warning(f"@{handle}: results may be truncated ({len(articles)} tweets, more available)")

                return self._make_result(source, articles, attempt)

            except HTTPError as e:
                if e.code == 429:
                    error_msg = "Rate limit exceeded"
                    logging.warning(f"Rate limit hit for @{handle}, attempt {attempt + 1}")
                    if attempt < RETRY_COUNT:
                        time.sleep(5)
                        continue
                else:
                    error_msg = f"HTTP {e.code}: {e.reason}"

            except Exception as e:
                error_msg = str(e)[:100]
                logging.debug(f"Attempt {attempt + 1} failed for @{handle}: {error_msg}")

            if attempt < RETRY_COUNT:
                time.sleep(RETRY_DELAY * (2 ** attempt))
                continue

            return self._make_error(source, error_msg, attempt)

    def fetch_all(self, sources: List[Dict[str, Any]], cutoff: datetime) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        total = len(sources)
        done = 0
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(self._fetch_user_tweets, source, cutoff): source
                       for source in sources}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                done += 1
                if result["status"] == "ok":
                    logging.info(f"[{done}/{total}] ✅ @{result['handle']}: {result['count']} tweets"
                                 + (f" (top: {result['articles'][0]['metrics']['like_count']}❤️)" if result['articles'] else ""))
                else:
                    logging.warning(f"[{done}/{total}] ❌ @{result['handle']}: {result['error']}")

        return results


# ---------------------------------------------------------------------------
# Jina Reader backend (free, no API key needed)
# ---------------------------------------------------------------------------

JINA_READER_BASE = "https://r.jina.ai"


class JinaReaderBackend(TwitterBackend):
    """Jina Reader backend – scrapes X profile pages via r.jina.ai.

    Free tier, no API key required. Rate-limited to ~1 qps.
    Returns parsed plain-text from the user's X profile page.
    """

    def __init__(self):
        self._semaphore = threading.Semaphore(1)  # enforce 1 qps
        self._last_request_time = 0.0
        self._lock = threading.Lock()
        self.logger = logging.getLogger("fetch-twitter.jina")

    def _throttle(self):
        """Ensure at least 1 second between requests."""
        with self._lock:
            now = time.monotonic()
            wait_time = 1.0 - (now - self._last_request_time)
            if wait_time > 0:
                time.sleep(wait_time)
            self._last_request_time = time.monotonic()

    @staticmethod
    def _parse_jina_date(date_str: str) -> Optional[datetime]:
        """Parse various date formats that Jina Reader may return.

        Common formats seen in Jina output for X pages:
        - 'Dec 10, 2024 · 12:34 PM' (X profile page format)
        - '2024-12-10T12:34:56.000Z' (ISO with millis)
        - 'Dec 10, 2024' (date only)
        - '10 Dec 2024' (alternative)
        - Relative dates: '2h', '5m', '1d', '3h ago'
        """
        date_str = date_str.strip()
        if not date_str:
            return None

        # Handle relative time patterns (e.g., "2h", "5m", "1d", "3h ago")
        rel_match = re.match(r'^(\d+)\s*([smhd])\s*(?:ago)?$', date_str.lower())
        if rel_match:
            amount = int(rel_match.group(1))
            unit = rel_match.group(2)
            now = datetime.now(timezone.utc)
            deltas = {'s': timedelta(seconds=amount),
                      'm': timedelta(minutes=amount),
                      'h': timedelta(hours=amount),
                      'd': timedelta(days=amount)}
            return now - deltas.get(unit, timedelta())

        # Handle "X hours ago" / "X minutes ago" patterns
        rel_match2 = re.match(r'^(\d+)\s+(second|minute|hour|day|week)s?\s+ago$', date_str.lower())
        if rel_match2:
            amount = int(rel_match2.group(1))
            unit = rel_match2.group(2)
            now = datetime.now(timezone.utc)
            deltas = {
                'second': timedelta(seconds=amount),
                'minute': timedelta(minutes=amount),
                'hour': timedelta(hours=amount),
                'day': timedelta(days=amount),
                'week': timedelta(weeks=amount),
            }
            return now - deltas.get(unit, timedelta())

        # Remove the middle dot separator and extra text after timezone
        date_str = re.sub(r'\s*[·|]\s*.*$', '', date_str).strip()

        formats = [
            "%b %d, %Y",                    # Dec 10, 2024
            "%B %d, %Y",                    # December 10, 2024
            "%d %b %Y",                     # 10 Dec 2024
            "%Y-%m-%dT%H:%M:%S.%fZ",       # 2024-12-10T12:34:56.000Z
            "%Y-%m-%dT%H:%M:%S%z",          # ISO 8601
            "%Y-%m-%dT%H:%M:%SZ",           # ISO without tz
            "%Y-%m-%d %H:%M:%S",            # Simple datetime
            "%Y-%m-%d",                     # Date only
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except (ValueError, TypeError):
                continue

        return None

    # Patterns for stripping markdown noise from Jina output
    _MD_IMAGE_RE = re.compile(r'!\[.*?\]\(.*?\)')          # ![alt](url)
    _MD_LINK_RE = re.compile(r'\[([^\]]*)\]\(.*?\)')       # [text](url) → text
    _ANALYTICS_VIEW_RE = re.compile(
        r'^\[([\d.]+[KkMmBb]?)\]\(https?://(?:x\.com|twitter\.com)/\w+/status/\d+/analytics\)$'
    )
    _TWEET_PERMALINK_MD_RE = re.compile(
        r'^\[.*?\]\(https?://(?:x\.com|twitter\.com)/\w+/status/\d+\)$'
    )
    _PROFILE_LINK_MD_RE = re.compile(
        r'^\[.*?\]\(https?://(?:x\.com|twitter\.com)/\w+/?(?:about|following|followers|verified_followers|photo|header_photo|media|with_replies)?/?\)$',
        re.IGNORECASE,
    )
    _METRICS_ONLY_RE = re.compile(
        r'^\d+[\.,]?\d*[kKmMbB]?$'  # bare number like "1.2K", "450"
    )
    _DATE_MD_RE = re.compile(
        r'^\[(?:'
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}'
        r'|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}'
        r'|\d{4}-\d{2}-\d{2}'
        r')\]\(https?://(?:x\.com|twitter\.com)/\w+/status/\d+\)$',
        re.IGNORECASE,
    )

    def _parse_jina_text(self, text: str, handle: str, topics: list,
                         cutoff: datetime) -> list:
        """Parse Jina Reader Markdown output into article dicts.

        Jina returns Markdown content for X profile pages.  Each tweet block
        has a predictable structure::

            [Avatar image](link)
            [Author Name](profile_link)
            [@handle](profile_link)
            ·
            [Jan 29, 2025](tweet_permalink)   ← date link → tweet_id here
            Tweet body text …
            [Card image](…)                   ← optional
            [From source.com](…)              ← optional
            1.2K                               ← replies
            2.1K                               ← retweets
            6.3K                               ← likes
            [5.8M](…/analytics)                ← views (markdown link)

        Previous implementation treated the whole Jina output as plain text,
        which caused markdown links like ``[5.8M](…/analytics)`` and
        ``[![Image…]](…)`` to leak into the title field.

        The fix: split on tweet-permalink date-links (``[Date](/status/ID)``)
        to delimit tweets, then strip markdown noise from content lines.
        """
        articles = []

        # ── Helper: extract tweet ID from a permalink URL ──────────
        tweet_id_re = re.compile(
            r'https?://(?:x\.com|twitter\.com)/\w+/status/(\d+)'
        )

        # ── Helper: parse a date string from markdown link or plain text ──
        date_inner_re = re.compile(
            r'(?:'
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}'
            r'|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}'
            r'|\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}'
            r'|\d+[hmsd]\s*(?:ago)?'
            r'|\d+\s+(?:second|minute|hour|day|week)s?\s+ago'
            r')',
            re.IGNORECASE,
        )

        # ── Step 1: Split into tweet blocks ────────────────────────
        # A tweet block starts with a markdown date-link line:
        #   [Jan 29, 2025](https://x.com/handle/status/12345)
        # Use that as the delimiter.
        tweet_start_re = re.compile(
            r'\['                              # opening bracket
            r'(?:'
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}'
            r'|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}'
            r'|\d{4}-\d{2}-\d{2}'
            r')'
            r'\]'                              # closing bracket
            r'\('                              # opening paren
            r'https?://(?:x\.com|twitter\.com)/[^/]+/status/(\d+)'  # capture tweet_id
            r'\)',                             # closing paren
            re.IGNORECASE,
        )

        # Find all tweet-start positions
        starts = list(tweet_start_re.finditer(text))
        if not starts:
            # Fallback: no markdown date-links found; try old-style parsing
            return self._parse_jina_text_legacy(text, handle, topics, cutoff)

        for idx, m in enumerate(starts):
            # Block = from this match to the next match (or end of text)
            block_end = starts[idx + 1].start() if idx + 1 < len(starts) else len(text)
            block = text[m.start():block_end].strip()
            tweet_id = m.group(1)

            # Extract date from the markdown link text
            date_text = m.group(0)
            date_inner_match = date_inner_re.search(date_text)
            if not date_inner_match:
                continue
            created_at = self._parse_jina_date(date_inner_match.group(0))
            if not created_at:
                continue
            if created_at < cutoff:
                continue

            # ── Step 2: Clean content lines ─────────────────────────
            lines = block.split('\n')
            content_lines = []
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue

                # Skip the date-link header line itself
                if stripped == m.group(0):
                    continue

                # Skip "·" separator
                if stripped in ('·', '··', '...'):
                    continue

                # Skip profile links: [Name](profile), [@handle](profile)
                if self._PROFILE_LINK_MD_RE.match(stripped):
                    continue

                # Skip avatar/image lines: [![Image…]](link) or ![Image](url)
                if stripped.startswith('[![') or self._MD_IMAGE_RE.match(stripped):
                    continue

                # Skip analytics view-count: [4.2M](…/analytics)
                if self._ANALYTICS_VIEW_RE.match(stripped):
                    continue

                # Skip tweet permalink in markdown: [Date](permalink)
                if self._TWEET_PERMALINK_MD_RE.match(stripped):
                    continue

                # Skip bare metric numbers ("1.2K", "450", "9.6K")
                if self._METRICS_ONLY_RE.match(stripped):
                    continue

                # Skip "Pinned" label
                if stripped.lower() == 'pinned':
                    continue

                # Skip "Quote" section header
                if stripped.lower() in ('quote', 'quoted'):
                    continue

                # Skip Jina page-level metadata leaks
                if stripped.lower().startswith('published time:'):
                    continue
                if stripped.lower().startswith('url source:'):
                    continue
                if stripped.lower().startswith('markdown content:'):
                    continue
                if stripped.lower().startswith('title:'):
                    continue

                # Strip remaining markdown: inline images → empty, links → text
                cleaned = self._MD_IMAGE_RE.sub('', stripped)
                cleaned = self._MD_LINK_RE.sub(r'\1', cleaned)
                cleaned = cleaned.strip()
                if not cleaned:
                    continue

                content_lines.append(cleaned)

            tweet_text = ' '.join(content_lines).strip()

            # Skip if too short or looks like metadata only
            if not tweet_text or len(tweet_text) < 5:
                continue

            # Skip retweets
            if tweet_text.startswith('RT @'):
                continue

            # Strip embedded tweet URLs (t.co shorteners) from end of text
            tweet_text = re.sub(r'\s*https?://t\.co/\S+$', '', tweet_text).strip()

            link = f"https://x.com/{handle}/status/{tweet_id}"

            articles.append({
                "title": clean_tweet_text(tweet_text),
                "link": link,
                "date": created_at.isoformat(),
                "topics": topics[:],
                "metrics": {},  # Jina Reader doesn't provide structured metrics
                "tweet_id": tweet_id,
                "source_type": "twitter",
                "source_name": f"@{handle}",
            })

        # Deduplicate by tweet_id
        seen_ids = set()
        unique = []
        for a in articles:
            tid = a.get("tweet_id", "")
            if tid and tid in seen_ids:
                continue
            if tid:
                seen_ids.add(tid)
            unique.append(a)

        return unique

    def _parse_jina_text_legacy(self, text: str, handle: str, topics: list,
                                cutoff: datetime) -> list:
        """Legacy fallback parser for non-markdown Jina output.

        Kept for backwards compatibility in case Jina returns plain text
        (e.g. older Jina versions or different Accept headers).
        """
        articles = []
        date_pattern = re.compile(
            r'(?:'
            r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}'
            r'|'
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}'
            r'|'
            r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}'
            r'|'
            r'\d+[hmsd]\s*(?:ago)?'
            r'|'
            r'\d+\s+(?:second|minute|hour|day|week)s?\s+ago'
            r')',
            re.IGNORECASE
        )
        tweet_url_pattern = re.compile(
            r'https?://(?:x\.com|twitter\.com)/\w+/status/(\d+)'
        )
        blocks = re.split(r'\n-{3,}\n|\n\n+', text)

        for block in blocks:
            block = block.strip()
            if not block or len(block) < 10:
                continue
            if any(skip in block.lower() for skip in [
                'sign in', 'log in', 'join today', 'cookies',
                'x.com home', 'search', 'footer',
                'javascript is disabled',
                'published time:', 'url source:', 'markdown content:',
            ]):
                continue
            date_match = date_pattern.search(block)
            if not date_match:
                continue
            date_str = date_match.group(0)
            created_at = self._parse_jina_date(date_str)
            if not created_at or created_at < cutoff:
                continue

            lines = block.split('\n')
            content_lines = []
            for line in lines:
                stripped = line.strip()
                if re.match(r'^\d+[\.,]?\d*[kKmMbB]?(\s*(likes?|reposts?|replies?|views?|bookmarks?|shares?))', stripped, re.IGNORECASE):
                    continue
                if re.match(r'^(likes?|reposts?|replies?|views?|bookmarks?|shares?):?\s*\d+', stripped, re.IGNORECASE):
                    continue
                if date_pattern.fullmatch(stripped):
                    continue
                if tweet_url_pattern.fullmatch(stripped):
                    continue
                if stripped.startswith('http://') or stripped.startswith('https://'):
                    if re.match(r'https?://(?:x\.com|twitter\.com)/\w+/status/\d+', stripped):
                        continue
                if not stripped:
                    continue
                # Skip Jina page-level metadata
                if stripped.lower().startswith('published time:'):
                    continue
                if stripped.lower().startswith('url source:'):
                    continue
                if stripped.lower().startswith('markdown content:'):
                    continue
                if stripped.lower().startswith('title:'):
                    continue
                # Also strip markdown noise in legacy mode
                cleaned = self._MD_IMAGE_RE.sub('', stripped)
                cleaned = self._MD_LINK_RE.sub(r'\1', cleaned)
                cleaned = cleaned.strip()
                if not cleaned:
                    continue
                content_lines.append(cleaned)

            tweet_text = ' '.join(content_lines).strip()
            if not tweet_text or len(tweet_text) < 5:
                continue
            if tweet_text.startswith('RT @'):
                continue

            tweet_id_match = tweet_url_pattern.search(block)
            tweet_id = tweet_id_match.group(1) if tweet_id_match else ""
            link = f"https://x.com/{handle}/status/{tweet_id}" if tweet_id else f"https://x.com/{handle}"

            articles.append({
                "title": clean_tweet_text(tweet_text),
                "link": link,
                "date": created_at.isoformat(),
                "topics": topics[:],
                "metrics": {},
                "tweet_id": tweet_id,
                "source_type": "twitter",
                "source_name": f"@{handle}",
            })

        # Deduplicate
        seen_ids = set()
        seen_titles = set()
        unique = []
        for a in articles:
            tid = a.get("tweet_id", "")
            title_key = a["title"][:80].lower()
            if tid:
                if tid in seen_ids:
                    continue
                seen_ids.add(tid)
            else:
                if title_key in seen_titles:
                    continue
                seen_titles.add(title_key)
            unique.append(a)

        return unique

    def _fetch_user_tweets(self, source: Dict[str, Any],
                           cutoff: datetime) -> Dict[str, Any]:
        handle = source["handle"].lstrip('@')
        topics = source["topics"]

        for attempt in range(RETRY_COUNT + 1):
            try:
                url = f"{JINA_READER_BASE}/https://x.com/{handle}"
                headers = {
                    "Accept": "text/plain",
                    "User-Agent": "TechDigest/2.0",
                }

                self._throttle()

                req = Request(url, headers=headers)
                with urlopen(req, timeout=TIMEOUT) as resp:
                    raw_text = resp.read().decode('utf-8', errors='replace')

                if not raw_text or len(raw_text) < 50:
                    return self._make_error(source, "Empty response from Jina Reader", attempt)

                articles = self._parse_jina_text(raw_text, handle, topics, cutoff)

                if not articles:
                    logging.debug(f"@{handle}: Jina returned text but no parseable tweets")

                return self._make_result(source, articles, attempt)

            except HTTPError as e:
                if e.code == 429:
                    error_msg = "Rate limit exceeded"
                    logging.warning(f"Jina rate limit for @{handle}, attempt {attempt + 1}")
                    if attempt < RETRY_COUNT:
                        time.sleep(5)
                        continue
                elif e.code == 404:
                    error_msg = f"User not found: @{handle}"
                else:
                    error_msg = f"HTTP {e.code}: {e.reason}"

            except Exception as e:
                error_msg = str(e)[:100]
                logging.debug(f"Jina attempt {attempt + 1} failed for @{handle}: {error_msg}")

            if attempt < RETRY_COUNT:
                time.sleep(RETRY_DELAY * (2 ** attempt))
                continue

            return self._make_error(source, error_msg, attempt)

    def fetch_all(self, sources: List[Dict[str, Any]],
                  cutoff: datetime) -> List[Dict[str, Any]]:
        """Sequential fetch – Jina Reader is limited to ~1 qps."""
        results: List[Dict[str, Any]] = []
        total = len(sources)

        for i, source in enumerate(sources):
            result = self._fetch_user_tweets(source, cutoff)
            results.append(result)
            done = i + 1
            if result["status"] == "ok":
                logging.info(f"[{done}/{total}] ✅ @{result['handle']}: {result['count']} tweets (jina)")
            else:
                logging.warning(f"[{done}/{total}] ❌ @{result['handle']}: {result.get('error', 'unknown')} (jina)")

        return results

class GetXApiBackend(TwitterBackend):
    """GetXAPI backend."""

    def __init__(self, api_key: str):
        """Initialize GetXAPI backend with API key validation."""
        if not api_key or len(api_key) < 10:
            raise ValueError("Invalid GETX_API_KEY format - expected at least 10 characters")
        self.api_key = api_key
        self.logger = logging.getLogger("fetch-twitter")

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse GetXAPI date string with multiple format support.
        
        Supported formats:
        - 'Tue Dec 10 07:00:30 +0000 2024' (Twitter format)
        - '2024-12-10T07:00:30+00:00' (ISO 8601)
        - '2024-12-10 07:00:30' (Simple datetime)
        """
        formats = [
            "%a %b %d %H:%M:%S %z %Y",      # Twitter format
            "%Y-%m-%dT%H:%M:%S%z",          # ISO 8601
            "%Y-%m-%d %H:%M:%S",            # Simple datetime
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except (ValueError, TypeError):
                continue
        
        self.logger.debug(f"Failed to parse date '{date_str}' with all known formats")
        return None

    def _parse_tweets_page(self, tweets: list, handle: str, topics: list, cutoff: datetime) -> list:
        articles = []
        for tweet in tweets:
            tweet_id = tweet.get("id")
            text = tweet.get("text")
            created_at_raw = tweet.get("createdAt")
            if not tweet_id or not text or not created_at_raw:
                continue
            if tweet.get("isReply"):
                continue
            if text.startswith("RT @"):
                continue

            created_at = self._parse_date(created_at_raw)
            if not created_at or created_at < cutoff:
                continue

            link = tweet.get("url") or f"https://x.com/{handle}/status/{tweet_id}"

            articles.append({
                "title": clean_tweet_text(text),
                "link": link,
                "date": created_at.isoformat(),
                "topics": topics[:],
                "metrics": {
                    "like_count": tweet.get("likeCount", 0),
                    "retweet_count": tweet.get("retweetCount", 0),
                    "reply_count": tweet.get("replyCount", 0),
                    "quote_count": tweet.get("quoteCount", 0),
                    "impression_count": tweet.get("viewCount", 0),
                },
                "tweet_id": tweet_id,
                "source_type": "twitter",
                "source_name": f"@{handle}",
            })
        return articles

    def _fetch_user_tweets(self, source: Dict[str, Any], cutoff: datetime) -> Dict[str, Any]:
        handle = source["handle"].lstrip('@')
        topics = source["topics"]

        for attempt in range(RETRY_COUNT + 1):
            try:
                url = f"{GETXAPI_BASE}/twitter/user/tweets?{urlencode({'userName': handle})}"
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "User-Agent": "TechDigest/2.0",
                }

                req = Request(url, headers=headers)
                with urlopen(req, timeout=TIMEOUT) as resp:
                    raw = json.loads(resp.read().decode())

                if raw.get("error"):
                    return self._make_error(source, str(raw["error"])[:100], attempt)

                articles = self._parse_tweets_page(
                    raw.get("tweets", []), handle, topics, cutoff
                )

                has_more = raw.get("has_more", False)
                next_cursor = raw.get("next_cursor")
                
                # Fetch page 2 if more results available (with retry)
                if has_more and next_cursor and articles:
                    oldest = min(datetime.fromisoformat(a["date"]) for a in articles)
                    if oldest >= cutoff:
                        for page_attempt in range(RETRY_COUNT + 1):
                            try:
                                page2_url = f"{GETXAPI_BASE}/twitter/user/tweets?{urlencode({'userName': handle, 'cursor': next_cursor})}"
                                req2 = Request(page2_url, headers=headers)
                                with urlopen(req2, timeout=TIMEOUT) as resp2:
                                    raw2 = json.loads(resp2.read().decode())
                                if raw2.get("error"):
                                    raise ValueError(str(raw2["error"])[:100])
                                articles.extend(self._parse_tweets_page(
                                    raw2.get("tweets", []), handle, topics, cutoff
                                ))
                                has_more = raw2.get("has_more", False)
                                break  # Success
                            except Exception as e:
                                self.logger.warning(f"@{handle}: page 2 attempt {page_attempt + 1} failed: {e}")
                                if page_attempt < RETRY_COUNT:
                                    time.sleep(RETRY_DELAY * (2 ** page_attempt))
                                else:
                                    self.logger.warning(f"@{handle}: page 2 failed after {RETRY_COUNT} attempts, keeping page 1 results")
                                    has_more = False

                if has_more and articles:
                    oldest = min(datetime.fromisoformat(a["date"]) for a in articles)
                    if oldest >= cutoff:
                        logging.warning(f"@{handle}: results may be truncated ({len(articles)} tweets, more available)")

                return self._make_result(source, articles, attempt)

            except HTTPError as e:
                if e.code == 429:
                    error_msg = "Rate limit exceeded"
                    logging.warning(f"Rate limit hit for @{handle}, attempt {attempt + 1}")
                    if attempt < RETRY_COUNT:
                        time.sleep(5)
                        continue
                else:
                    error_msg = f"HTTP {e.code}: {e.reason}"

            except Exception as e:
                error_msg = str(e)[:100]
                logging.debug(f"Attempt {attempt + 1} failed for @{handle}: {error_msg}")

            if attempt < RETRY_COUNT:
                time.sleep(RETRY_DELAY * (2 ** attempt))
                continue

            return self._make_error(source, error_msg, attempt)

    def fetch_all(self, sources: List[Dict[str, Any]], cutoff: datetime) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        total = len(sources)
        done = 0
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {pool.submit(self._fetch_user_tweets, source, cutoff): source
                       for source in sources}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                done += 1
                if result["status"] == "ok":
                    logging.info(f"[{done}/{total}] ✅ @{result['handle']}: {result['count']} tweets"
                                 + (f" (top: {result['articles'][0]['metrics']['like_count']}❤️)" if result['articles'] else ""))
                else:
                    logging.warning(f"[{done}/{total}] ❌ @{result['handle']}: {result['error']}")

        return results


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

def select_backend(backend_name: str, no_cache: bool = False) -> Optional[TwitterBackend]:
    """Select and instantiate the appropriate backend.

    Returns None if no credentials are available for the chosen backend.
    """
    if backend_name == "getxapi":
        key = os.getenv("GETX_API_KEY")
        if not key:
            logging.error("GETX_API_KEY not set (required for getxapi backend)")
            return None
        logging.info("Using GetXAPI backend")
        return GetXApiBackend(key)

    if backend_name == "twitterapiio":
        key = os.getenv("TWITTERAPI_IO_KEY")
        if not key:
            logging.error("TWITTERAPI_IO_KEY not set (required for twitterapiio backend)")
            return None
        logging.info("Using twitterapi.io backend")
        return TwitterApiIoBackend(key)

    if backend_name == "official":
        token = os.getenv("X_BEARER_TOKEN")
        if not token:
            logging.error("X_BEARER_TOKEN not set (required for official backend)")
            return None
        logging.info("Using official X API v2 backend")
        return OfficialBackend(token, no_cache=no_cache)

    if backend_name == "jina":
        logging.info("Using Jina Reader backend (free, no API key)")
        return JinaReaderBackend()

    # auto: try getxapi first, then twitterapiio, then official
    if backend_name == "auto":
        getx_key = os.getenv("GETX_API_KEY")
        if getx_key:
            logging.info("Auto-selected GetXAPI backend (GETX_API_KEY set)")
            return GetXApiBackend(getx_key)
        key = os.getenv("TWITTERAPI_IO_KEY")
        if key:
            logging.info("Auto-selected twitterapi.io backend (TWITTERAPI_IO_KEY set)")
            return TwitterApiIoBackend(key)
        token = os.getenv("X_BEARER_TOKEN")
        if token:
            logging.info("Auto-selected official X API v2 backend (X_BEARER_TOKEN set)")
            return OfficialBackend(token, no_cache=no_cache)
        # Fallback: Jina Reader (free, no API key needed)
        logging.info("Auto-selected Jina Reader backend (no API keys found, free fallback)")
        return JinaReaderBackend()

    logging.error(f"Unknown backend: {backend_name}")
    return None


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------

def load_twitter_sources(defaults_dir: Path, config_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load Twitter sources from unified configuration with overlay support."""
    try:
        from config_loader import load_merged_sources
    except ImportError:
        # Fallback for relative import
        import sys
        sys.path.append(str(Path(__file__).parent))
        from config_loader import load_merged_sources

    # Load merged sources from defaults + optional user overlay
    all_sources = load_merged_sources(defaults_dir, config_dir)

    # Filter Twitter sources that are enabled
    twitter_sources = []
    for source in all_sources:
        if source.get("type") == "twitter" and source.get("enabled", True):
            if not source.get("handle"):
                logging.warning(f"Twitter source {source.get('id')} missing handle, skipping")
                continue
            twitter_sources.append(source)

    logging.info(f"Loaded {len(twitter_sources)} enabled Twitter sources")
    return twitter_sources


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main Twitter fetching function."""
    parser = argparse.ArgumentParser(
        description="Fetch recent tweets from Twitter/X KOL accounts. "
                   "Supports official X API v2, GetXAPI, and twitterapi.io backends.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    export X_BEARER_TOKEN="your_token_here"
    python3 fetch-twitter.py
    python3 fetch-twitter.py --defaults config/defaults --config workspace/config --hours 24 -o results.json
    python3 fetch-twitter.py --backend twitterapiio  # use twitterapi.io
    python3 fetch-twitter.py --config workspace/config --verbose  # backward compatibility
        """
    )

    parser.add_argument(
        "--defaults",
        type=Path,
        default=Path("config/defaults"),
        help="Default configuration directory with skill defaults (default: config/defaults)"
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="User configuration directory for overlays (optional)"
    )

    parser.add_argument(
        "--hours",
        type=int,
        default=48,
        help="Time window in hours for tweets (default: 48)"
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output JSON path (default: auto-generated temp file)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass username→ID cache (official backend only)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-fetch even if cached output exists"
    )

    parser.add_argument(
        "--backend",
        choices=["official", "twitterapiio", "getxapi", "jina", "auto"],
        default=None,
        help="Twitter API backend (overrides TWITTER_API_BACKEND env var). "
             "auto = getxapi if GETX_API_KEY set, else twitterapiio if TWITTERAPI_IO_KEY set, "
             "else official if X_BEARER_TOKEN set, else jina (free)"
    )

    args = parser.parse_args()
    logger = setup_logging(args.verbose)

    # Resume support: skip if output exists, is valid JSON, and < 1 hour old
    if args.output and args.output.exists() and not args.force:
        try:
            age_seconds = time.time() - args.output.stat().st_mtime
            if age_seconds < 3600:
                with open(args.output, 'r') as f:
                    json.load(f)
                logger.info(f"Skipping (cached output exists): {args.output}")
                return 0
        except (json.JSONDecodeError, OSError):
            pass

    # Resolve backend: CLI arg > env var > default (auto)
    backend_name = args.backend or os.getenv("TWITTER_API_BACKEND", "auto")

    backend = select_backend(backend_name, no_cache=args.no_cache)
    if not backend:
        logger.warning("No Twitter backend available. Writing empty result and skipping Twitter fetch.")
        empty_result = {
            "generated": datetime.now(timezone.utc).isoformat(),
            "source_type": "twitter",
            "backend": backend_name,
            "hours": args.hours,
            "sources_total": 0,
            "sources_ok": 0,
            "total_articles": 0,
            "sources": [],
            "skipped_reason": f"No credentials for backend '{backend_name}'"
        }
        output_path = args.output or Path("/tmp/td-twitter.json")
        with open(output_path, "w") as f:
            json.dump(empty_result, f, indent=2)
        print(f"Output (empty): {output_path}")
        return 0

    # Auto-generate unique output path if not specified
    if not args.output:
        fd, temp_path = tempfile.mkstemp(prefix="tech-news-digest-twitter-", suffix=".json")
        os.close(fd)
        args.output = Path(temp_path)

    try:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=args.hours)

        # Backward compatibility: if only --config provided, use old behavior
        if args.config and args.defaults == Path("config/defaults") and not args.defaults.exists():
            logger.debug("Backward compatibility mode: using --config as sole source")
            sources = load_twitter_sources(args.config, None)
        else:
            sources = load_twitter_sources(args.defaults, args.config)

        if not sources:
            logger.warning("No Twitter sources found or all disabled")

        logger.info(f"Fetching {len(sources)} Twitter accounts (window: {args.hours}h, backend: {backend_name})")

        results = backend.fetch_all(sources, cutoff)

        # Sort: priority first, then by article count
        results.sort(key=lambda x: (not x.get("priority", False), -x.get("count", 0)))

        ok_count = sum(1 for r in results if r["status"] == "ok")
        total_tweets = sum(r.get("count", 0) for r in results)

        output = {
            "generated": datetime.now(timezone.utc).isoformat(),
            "source_type": "twitter",
            "backend": backend_name,
            "defaults_dir": str(args.defaults),
            "config_dir": str(args.config) if args.config else None,
            "hours": args.hours,
            "sources_total": len(results),
            "sources_ok": ok_count,
            "total_articles": total_tweets,
            "sources": results,
        }

        # Write output
        json_str = json.dumps(output, ensure_ascii=False, indent=2)
        with open(args.output, "w", encoding='utf-8') as f:
            f.write(json_str)

        logger.info(f"✅ Done: {ok_count}/{len(results)} accounts ok, "
                   f"{total_tweets} tweets → {args.output}")

        return 0

    except Exception as e:
        logger.error(f"💥 Twitter fetch failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
