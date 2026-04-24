"""
Bilibili 视频源爬虫 — 突破默认 1000 条搜索上限。

为什么 1000 条上限：B 站 search/type API 硬性限制每个 query 50 页 × 20 条。
解决：把 1 个 query 切成 N 个互不重叠的子查询，每个子查询独立享受 1000 条配额。

5 种切片维度（叠加用）：
  1. 时间窗 pubtime_begin_s / pubtime_end_s   (按月/周切，× N)
  2. 时长档 duration  (1=≤10min, 2=10~30min, 3=30~60min, 4=>60min)
  3. 排序   order     (totalrank, click, pubdate, dm, stow, scores)
  4. 同义词扩展     (强混 / 强混声 / belting / belt mix / 金属感 / ...)
  5. UP主纵深       (爬到优质 up 后 → 直接拉他的 space, 无 1000 条限制)

依赖：
    pip install httpx aiolimiter

使用：
    from dataset_pipeline.sources.bilibili_scraper import BiliScraper
    s = BiliScraper(cookie="...")  # 从浏览器抓 SESSDATA cookie
    rows = await s.harvest_keyword("强混", synonyms=["belting", "强混声"],
                                   start_year=2018, end_year=2025)
    # rows = [{bvid, title, owner, duration, pubdate, ...}, ...]
"""
from __future__ import annotations
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Set, Dict, Any

log = logging.getLogger(__name__)

SEARCH_API = "https://api.bilibili.com/x/web-interface/search/type"
SPACE_API = "https://api.bilibili.com/x/space/wbi/arc/search"   # UP 主投稿
RELATED_API = "https://api.bilibili.com/x/web-interface/archive/related"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

# 排序维度
ORDERS = ["totalrank", "click", "pubdate", "dm", "stow", "scores"]
# 时长档：1=≤10min, 2=10~30min, 3=30~60min, 4=>60min
DURATIONS = [0, 1, 2, 3, 4]   # 0 = 全部，作为兜底


@dataclass
class BiliVideo:
    bvid: str
    aid: int
    title: str
    owner_name: str
    owner_mid: int
    duration_sec: int
    pubdate_ts: int
    play: int
    danmaku: int
    matched_keyword: str = ""
    discovered_via: str = ""    # "search:keyword:order:duration" or "space:mid"

    def key(self) -> str:
        return self.bvid

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def _parse_search_item(item: Dict[str, Any], matched_kw: str, via: str) -> Optional[BiliVideo]:
    try:
        # duration 是 "MM:SS" 或 "HH:MM:SS"
        dur_str = item.get("duration", "0")
        parts = [int(x) for x in dur_str.split(":") if x.isdigit()]
        if len(parts) == 3:
            dur = parts[0]*3600 + parts[1]*60 + parts[2]
        elif len(parts) == 2:
            dur = parts[0]*60 + parts[1]
        else:
            dur = int(parts[0]) if parts else 0
        return BiliVideo(
            bvid=item["bvid"],
            aid=int(item.get("aid", 0)),
            title=item.get("title", "").replace('<em class="keyword">', "").replace("</em>", ""),
            owner_name=item.get("author", ""),
            owner_mid=int(item.get("mid", 0)),
            duration_sec=dur,
            pubdate_ts=int(item.get("pubdate", 0)),
            play=int(item.get("play", 0)),
            danmaku=int(item.get("video_review", 0)),
            matched_keyword=matched_kw,
            discovered_via=via,
        )
    except Exception as e:
        log.debug("parse fail: %s", e)
        return None


class BiliScraper:
    def __init__(self, cookie: Optional[str] = None,
                 qps: float = 1.5, concurrency: int = 4):
        """cookie: 至少包含 SESSDATA, buvid3。否则只能拿前几页。
        qps: 每秒最多请求数（B 站对未登录限制更严，建议 ≤ 2）。
        """
        self.cookie = cookie or ""
        self.qps = qps
        self.concurrency = concurrency
        self._last_call = 0.0
        self._lock = asyncio.Lock()
        self.seen: Set[str] = set()

    def _headers(self) -> Dict[str, str]:
        h = {"User-Agent": USER_AGENT, "Referer": "https://www.bilibili.com/"}
        if self.cookie:
            h["Cookie"] = self.cookie
        return h

    async def _throttle(self):
        async with self._lock:
            now = time.monotonic()
            wait = max(0.0, 1.0 / self.qps - (now - self._last_call))
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()

    async def _get_json(self, client, url: str, params: Dict[str, Any]) -> Optional[dict]:
        await self._throttle()
        try:
            r = await client.get(url, params=params, headers=self._headers(), timeout=20.0)
            r.raise_for_status()
            data = r.json()
            if data.get("code") != 0:
                log.warning("bili api code=%s msg=%s", data.get("code"), data.get("message"))
                return None
            return data.get("data")
        except Exception as e:
            log.warning("bili api fail: %s", e)
            return None

    # =====================================================
    # 搜索：单一 (keyword, order, duration, time_window) 切片
    # =====================================================
    async def search_slice(self, client, keyword: str,
                           order: str, duration: int,
                           pubtime_begin_s: int, pubtime_end_s: int,
                           max_pages: int = 50) -> List[BiliVideo]:
        out: List[BiliVideo] = []
        for page in range(1, max_pages + 1):
            params = {
                "search_type": "video",
                "keyword": keyword,
                "order": order,
                "duration": duration,
                "page": page,
                "pubtime_begin_s": pubtime_begin_s,
                "pubtime_end_s": pubtime_end_s,
            }
            data = await self._get_json(client, SEARCH_API, params)
            if not data:
                break
            results = data.get("result") or []
            if not results:
                break
            for it in results:
                v = _parse_search_item(it, matched_kw=keyword,
                                       via=f"search:{keyword}:{order}:{duration}")
                if v and v.bvid not in self.seen:
                    self.seen.add(v.bvid)
                    out.append(v)
            if len(results) < 20:    # 不足一页，到底
                break
        log.info("[search] kw=%s ord=%s dur=%s win=%s..%s → %d new",
                 keyword, order, duration, pubtime_begin_s, pubtime_end_s, len(out))
        return out

    # =====================================================
    # 主入口：关键词矩阵采集
    # =====================================================
    async def harvest_keyword(self, keyword: str,
                              synonyms: Optional[Iterable[str]] = None,
                              start_year: int = 2017,
                              end_year: Optional[int] = None,
                              window_months: int = 3,
                              orders: Iterable[str] = ("totalrank", "click", "pubdate"),
                              durations: Iterable[int] = (1, 2, 3, 4)) -> List[BiliVideo]:
        """对 keyword + 同义词，做 (time_window × order × duration) 矩阵切片采集。

        默认 3 个 order × 4 个 duration × N 个时间窗 = 单关键词最多 12N × 1000 条配额。
        """
        import httpx  # type: ignore

        end_year = end_year or datetime.now().year
        keywords = [keyword] + list(synonyms or [])
        windows = list(_iter_time_windows(start_year, end_year, window_months))
        log.info("harvest %s + %d synonyms × %d windows × %d orders × %d durations",
                 keyword, len(keywords)-1, len(windows), len(orders), len(durations))

        results: List[BiliVideo] = []
        async with httpx.AsyncClient() as client:
            sem = asyncio.Semaphore(self.concurrency)

            async def one(kw, ordr, dur, wb, we):
                async with sem:
                    rows = await self.search_slice(client, kw, ordr, dur, wb, we)
                    results.extend(rows)

            tasks = [
                one(kw, ordr, dur, wb, we)
                for kw in keywords
                for ordr in orders
                for dur in durations
                for (wb, we) in windows
            ]
            await asyncio.gather(*tasks)
        log.info("harvest %s done: %d unique videos", keyword, len(results))
        return results

    # =====================================================
    # UP 主纵深：拉一个 mid 的全部投稿
    # =====================================================
    async def harvest_uploader(self, mid: int, max_pages: int = 50) -> List[BiliVideo]:
        """爬 UP 主全部公开投稿。无 1000 条上限。"""
        import httpx  # type: ignore
        out: List[BiliVideo] = []
        async with httpx.AsyncClient() as client:
            for page in range(1, max_pages + 1):
                params = {"mid": mid, "ps": 50, "pn": page,
                          "order": "pubdate", "platform": "web"}
                data = await self._get_json(client, SPACE_API, params)
                if not data:
                    break
                items = (data.get("list") or {}).get("vlist") or []
                if not items:
                    break
                for it in items:
                    v = BiliVideo(
                        bvid=it["bvid"], aid=int(it.get("aid", 0)),
                        title=it.get("title", ""),
                        owner_name=it.get("author", ""), owner_mid=mid,
                        duration_sec=int(it.get("length", "0:0").split(":")[0])*60
                                     + int(it.get("length", "0:0").split(":")[-1]),
                        pubdate_ts=int(it.get("created", 0)),
                        play=int(it.get("play", 0)),
                        danmaku=int(it.get("video_review", 0)),
                        discovered_via=f"space:{mid}",
                    )
                    if v.bvid not in self.seen:
                        self.seen.add(v.bvid)
                        out.append(v)
                if len(items) < 50:
                    break
        log.info("[space] mid=%d → %d new", mid, len(out))
        return out


# ---------- 辅助：切时间窗 ----------
def _iter_time_windows(start_year: int, end_year: int, months: int):
    """yield (begin_ts, end_ts) inclusive, 按 N 月切。"""
    cur = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31, 23, 59, 59)
    while cur < end:
        # naive +months
        m = cur.month - 1 + months
        nxt = datetime(cur.year + m // 12, m % 12 + 1, 1)
        nxt = min(nxt, end)
        yield int(cur.timestamp()), int(nxt.timestamp())
        cur = nxt


# ---------- CLI ----------
def _cli():
    import argparse
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Bilibili keyword harvester")
    p.add_argument("--keyword", required=True)
    p.add_argument("--synonyms", nargs="*", default=[])
    p.add_argument("--start-year", type=int, default=2018)
    p.add_argument("--end-year", type=int, default=datetime.now().year)
    p.add_argument("--window-months", type=int, default=3)
    p.add_argument("--cookie", default="", help="包含 SESSDATA 的 Cookie 字符串")
    p.add_argument("--out", type=Path, required=True, help="output jsonl")
    args = p.parse_args()

    s = BiliScraper(cookie=args.cookie)
    rows = asyncio.run(s.harvest_keyword(
        args.keyword, synonyms=args.synonyms,
        start_year=args.start_year, end_year=args.end_year,
        window_months=args.window_months,
    ))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    print(f"\n✅ {len(rows)} unique videos → {args.out}")


if __name__ == "__main__":
    _cli()
