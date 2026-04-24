"""
Stage 2 — LLM 无情法官调用层。
- MiniMax (OpenAI 兼容风格) 并发调用
- Pydantic 强校验 + Fail Fast（不重试 parse 失败）
- 限速：默认 600 req / 5h，≤ 50 req/s
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, Literal, List, Iterable

from pydantic import BaseModel, Field, ValidationError

log = logging.getLogger(__name__)


# =========================================================
# Verdict schema (与 STAGE2_LLM_PROMPT.md 保持一致)
# =========================================================
class Verdict(BaseModel):
    candidate_id: str
    decision: Literal["ACCEPT", "REJECT"]
    reject_reason: Optional[Literal[
        "NO_VALID_DEMO_BLOCK", "NEGATIVE_CONTEXT_DETECTED",
        "TRIGGER_TECHNIQUE_MISMATCH", "ACOUSTIC_BELOW_THRESHOLD",
        "TRANSITION_NOT_FOUND", "INSUFFICIENT_EVIDENCE",
    ]] = None
    chosen_center_tick: Optional[int] = None
    chosen_vad_block: Optional[List[int]] = None
    confidence: float = Field(ge=0, le=1, default=0.0)
    notes: str = ""


SYSTEM_PROMPT_ZH = (Path(__file__).parent / "STAGE2_LLM_PROMPT.md").read_text("utf-8")


# =========================================================
# MiniMax client (OpenAI-compatible)
# =========================================================
class MiniMaxClient:
    """MiniMax 聊天补全 API。用 httpx async。"""

    def __init__(self, api_key: Optional[str] = None,
                 base_url: str = "https://api.minimax.chat/v1",
                 model: str = "MiniMax-M2",
                 max_tokens: int = 512, temperature: float = 0.0):
        self.api_key = api_key or os.getenv("MINIMAX_API_KEY")
        if not self.api_key:
            raise RuntimeError("MINIMAX_API_KEY not set")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def chat_json(self, system: str, user_json: str,
                        client) -> Optional[str]:
        """调一次 chat completion，要求 JSON 输出。返回原始文本或 None（网络错误）。"""
        url = f"{self.base_url}/text/chatcompletion_v2"
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_json},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            # MiniMax 支持 response_format={"type": "json_object"}
            "response_format": {"type": "json_object"},
        }
        try:
            r = await client.post(url, json=payload, headers=headers, timeout=30.0)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            log.warning("MiniMax call failed: %s", e)
            return None


# =========================================================
# Validator (结合 candidate payload 做越界检查)
# =========================================================
def validate_verdict(raw: str, candidate: dict) -> Optional[Verdict]:
    if raw is None:
        return None
    try:
        v = Verdict.model_validate_json(raw)
    except ValidationError as e:
        log.debug("pydantic reject: %s", e)
        return None

    if v.decision != "ACCEPT":
        return v

    intervals = candidate["constraints"]["must_pick_center_within"]
    if not intervals:
        log.debug("accept without intervals → reject")
        return None
    # center 必须落在某个区间内
    center = v.chosen_center_tick or -1
    if not any(a <= center <= b for a, b in intervals):
        log.debug("center %s out of range %s → reject", center, intervals)
        return None
    # chosen_vad_block 必须是原样复制
    if v.chosen_vad_block and list(v.chosen_vad_block) not in [list(x) for x in intervals]:
        # 对 transition_demo 允许 chosen_vad_block=None（中心本来就是 anchor）
        if candidate.get("candidate_type") != "transition_demo":
            log.debug("vad_block %s not in %s → reject", v.chosen_vad_block, intervals)
            return None
    return v


# =========================================================
# 异步并发 + 限速
# =========================================================
class RateLimiter:
    """滑动窗口限速。默认 50 req/s + 600 req/5h。"""

    def __init__(self, per_second: int = 50, per_window: int = 600,
                 window_seconds: int = 18000):
        self.per_second = per_second
        self.per_window = per_window
        self.window_seconds = window_seconds
        self._second_bucket: List[float] = []
        self._window_bucket: List[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            self._second_bucket = [t for t in self._second_bucket if now - t < 1.0]
            self._window_bucket = [t for t in self._window_bucket
                                   if now - t < self.window_seconds]
            if len(self._second_bucket) >= self.per_second:
                wait = 1.0 - (now - self._second_bucket[0]) + 0.01
            elif len(self._window_bucket) >= self.per_window:
                wait = self.window_seconds - (now - self._window_bucket[0]) + 0.1
            else:
                wait = 0
        if wait > 0:
            log.info("rate limit: sleeping %.2fs", wait)
            await asyncio.sleep(wait)
            return await self.acquire()
        async with self._lock:
            t = time.monotonic()
            self._second_bucket.append(t)
            self._window_bucket.append(t)


async def judge_all(candidates_jsonl: Path, out_jsonl: Path,
                    client: MiniMaxClient,
                    concurrency: int = 8,
                    rate_limit: Optional[RateLimiter] = None) -> int:
    """批量裁决。返回 ACCEPT 数。"""
    import httpx  # type: ignore

    candidates = [json.loads(ln) for ln in candidates_jsonl.read_text("utf-8").splitlines() if ln.strip()]
    if not candidates:
        out_jsonl.write_text("", encoding="utf-8")
        return 0

    limiter = rate_limit or RateLimiter()
    sem = asyncio.Semaphore(concurrency)
    results: List[Verdict] = [None] * len(candidates)  # type: ignore

    async with httpx.AsyncClient() as http:
        async def one(i: int, cand: dict):
            async with sem:
                await limiter.acquire()
                user_msg = json.dumps(cand, ensure_ascii=False)
                raw = await client.chat_json(SYSTEM_PROMPT_ZH, user_msg, http)
                v = validate_verdict(raw, cand)
                results[i] = v

        await asyncio.gather(*(one(i, c) for i, c in enumerate(candidates)))

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    accept_cnt = 0
    with out_jsonl.open("w", encoding="utf-8") as f:
        for v in results:
            if v is None:
                continue
            if v.decision == "ACCEPT":
                accept_cnt += 1
            f.write(v.model_dump_json() + "\n")
    log.info("judge %s: total=%d, accept=%d → %s",
             candidates_jsonl.name, len(candidates), accept_cnt, out_jsonl)
    return accept_cnt


def judge_technique_folder(work_dir: Path, technique: str,
                           client: Optional[MiniMaxClient] = None) -> int:
    """同步 wrapper：裁决某个技巧目录下的 candidates.jsonl。"""
    client = client or MiniMaxClient()
    cand = work_dir / technique / "candidates.jsonl"
    out = work_dir / technique / "verdicts.jsonl"
    if not cand.exists():
        log.warning("no candidates for %s", technique)
        return 0
    return asyncio.run(judge_all(cand, out, client))
