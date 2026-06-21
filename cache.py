"""Redis 기반 답변 캐시.

query(+chat_history) -> 최종 답변 문자열을 통째로 캐싱한다.
Redis가 꺼져 있거나 오류가 나도 앱이 죽지 않도록 모든 동작은 graceful degradation.
"""

import hashlib
import re
import time

import redis

from config import REDIS_URL, CACHE_TTL, CACHE_ENABLED

# 모든 ChatDDS 캐시 키에 붙는 접두사 (목록/일괄삭제 시 선별용)
KEY_PREFIX = "chatdds:answer:"

# from_url은 lazy connect라 import 시점에 Redis가 없어도 예외가 나지 않는다.
# 실제 연결 실패는 각 get/set의 try/except에서 흡수하고, 복구되면 자동 재연결된다.
_client = (
    redis.Redis.from_url(
        REDIS_URL,
        decode_responses=True,
        socket_connect_timeout=1,
        socket_timeout=1,
    )
    if CACHE_ENABLED
    else None
)


# 문장부호(물음표/마침표/느낌표/물결 등) 제거용
_PUNCT_RE = re.compile(r"[?？!！.。,，~〜…\"'`]+")


def _normalize(text: str) -> str:
    """표현 차이를 줄여 사소한 변형도 같은 캐시로 잡히게 정규화한다.

    - 소문자화
    - 문장부호 제거 ("뭐야?" == "뭐야")
    - 모든 공백 제거 ("홍수 지질학" == "홍수지질학")
    의미가 다른 질문(예: "뭐야" vs "왜 중요해")은 여전히 구분된다.
    """
    text = text.lower()
    text = _PUNCT_RE.sub("", text)
    text = re.sub(r"\s+", "", text)
    return text


def _make_key(query: str, chat_history: str = "") -> str:
    """질문과 대화 맥락을 합쳐 해시 키를 만든다.

    질문은 정규화해 사소한 표현 차이(공백/문장부호)를 흡수한다.
    chat_history를 키에 포함해 같은 질문이라도 맥락이 다르면 다른 답을 캐싱한다.
    (첫 턴 질문은 chat_history가 비어 있어 자연스럽게 캐시 히트된다.)
    """
    raw = f"{_normalize(query)}||{_normalize(chat_history)}".encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    return f"{KEY_PREFIX}{digest}"


def get_cached_answer(query: str, chat_history: str = ""):
    """캐시된 답변을 반환한다. 없거나 Redis 오류면 None."""
    if _client is None:
        return None
    try:
        # 각 항목은 question/answer/created_at 필드를 가진 Redis 해시로 저장된다.
        return _client.hget(_make_key(query, chat_history), "answer")
    except Exception:
        return None


def set_cached_answer(query: str, answer: str, chat_history: str = "") -> None:
    """답변을 질문 원문과 함께 TTL을 걸어 캐싱한다. Redis 오류면 조용히 무시."""
    if _client is None or not answer:
        return
    try:
        key = _make_key(query, chat_history)
        _client.hset(key, mapping={
            "question": query,
            "answer": answer,
            "created_at": str(int(time.time())),
        })
        _client.expire(key, CACHE_TTL)
    except Exception:
        pass


def list_cached_answers():
    """캐시된 모든 항목을 최신순 리스트로 반환한다.

    각 항목: {"question", "answer", "created_at"(epoch초), "ttl"(남은초)}
    Redis가 없으면 빈 리스트.
    """
    if _client is None:
        return []
    items = []
    try:
        for key in _client.scan_iter(match=f"{KEY_PREFIX}*"):
            try:
                data = _client.hgetall(key)
            except Exception:
                continue  # 구버전(문자열) 키 등은 건너뜀
            if not data or "answer" not in data:
                continue
            items.append({
                "question": data.get("question", "(질문 정보 없음)"),
                "answer": data.get("answer", ""),
                "created_at": int(data.get("created_at", 0) or 0),
                "ttl": _client.ttl(key),
            })
    except Exception:
        return items
    items.sort(key=lambda x: x["created_at"], reverse=True)
    return items


def clear_cache() -> int:
    """ChatDDS 캐시 항목(KEY_PREFIX)만 삭제하고 삭제된 개수를 반환한다."""
    if _client is None:
        return 0
    try:
        keys = list(_client.scan_iter(match=f"{KEY_PREFIX}*"))
        if keys:
            return _client.delete(*keys)
        return 0
    except Exception:
        return 0


def cache_available() -> bool:
    """Redis에 실제로 연결 가능한지 확인한다 (UI 안내용)."""
    if _client is None:
        return False
    try:
        return bool(_client.ping())
    except Exception:
        return False
