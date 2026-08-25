"""유닛 번역의 single-flight 조율 + 유닛 캐시 — engine.run_translation에서 승격.

같은 캐시 키를 두 스레드가 동시에 API로 보내지 않기 위한 조율기다.
지금까지 중복 유닛 제거는 "먼저 끝난 유닛이 메인 루프에서 캐시에 기록되고,
늦게 시작한 중복이 그걸 읽는" 레이스에 기대고 있었다 — 동시성을 올릴수록
그 레이스가 덜 맞아 같은 문단을 두 번 번역하게 된다(문서별 중복률 4~16% 실측).

완료 flight도 한 run 동안 보존한다. translated는 영속 cache로, kept는
메모리 outcome으로 후속 중복에 재사용해 실패한 같은 문단을 다시 두드리지 않는다.
"""

from __future__ import annotations

import threading
from typing import Callable


class Flight:
    """한 캐시 키의 진행 상태. owner가 result 또는 error를 공개한 뒤 event를 set한다."""

    __slots__ = ("event", "result", "error")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.result = None
        self.error: tuple[type[BaseException], tuple] | None = None

    def wait_for(self, halted: Callable[[], bool], interval: float = 0.5) -> bool:
        """선점 스레드 완료까지 대기. 완료면 True, 취소·abort로 중단하면 False.

        취소 응답성을 위해 0.5초씩 끊어 확인한다. 한 유닛은 재시도·repair·분할로
        cfg.timeout_s보다 오래 걸릴 수 있다. 여기서 임의 deadline을 두면 정상 owner가
        도는 중 같은 API를 중복 호출한다. owner는 모든 종료 경로의 finally에서
        event를 set하므로 별도 상한이 필요 없다.
        """
        while not self.event.wait(interval):
            if halted():
                return False
        return True

    def capture(self, exc: BaseException) -> None:
        """owner의 예외를 waiter가 복제할 수 있는 형태(타입+args)로 보관한다."""
        self.error = (type(exc), exc.args)

    def reraise(self) -> None:
        """owner가 남긴 예외를 waiter 스레드에서 복제해 던진다 (항상 raise)."""
        error_type, error_args = self.error  # type: ignore[misc]
        try:
            cloned_error = error_type(*error_args)
        except TypeError as clone_error:
            raise RuntimeError("동일 번역 유닛 처리 중 오류가 발생했습니다") from clone_error
        raise cloned_error


class SingleFlight:
    """캐시 키별 owner/waiter 조율 + 유닛 캐시(dict: cache_key → 번역문).

    캐시 접근과 flight 등록은 서로 다른 잠금으로 보호한다 — worker가 결과를
    공개하는 동안 주기 flush가 캐시를 직렬화해도 서로를 막지 않게 하기 위함.
    """

    def __init__(self, cache: dict[str, str] | None = None) -> None:
        self.cache: dict[str, str] = dict(cache) if cache else {}
        # run 시작 시점의 units.json 키와, 그중 이번 run에서 실제로 적중한 키.
        # cached 카운트로는 이 신호를 낼 수 없다 — 문서 내 중복 유닛의 single-flight
        # 재사용도 cached를 올리므로, 캐시가 전량 무효인 실 논문에서도 >0이 된다(실측 2).
        self.prior_keys: frozenset[str] = frozenset(self.cache)
        self.prior_hits: set[str] = set()
        self._cache_lock = threading.Lock()
        self._inflight: dict[str, Flight] = {}
        self._inflight_lock = threading.Lock()

    @property
    def prior_count(self) -> int:
        """run 시작 시점의 units.json 항목 수."""
        return len(self.prior_keys)

    def read(self, key: str) -> tuple[bool, str]:
        """공유 캐시의 단일 키를 원자적으로 읽는다 (빈 문자열도 값으로 구분)."""
        with self._cache_lock:
            if key in self.cache:
                if key in self.prior_keys:
                    self.prior_hits.add(key)  # 기존 units.json 재사용 계측(락 안에서)
                return True, self.cache[key]
        return False, ""

    def publish(self, key: str, value: str) -> None:
        with self._cache_lock:
            self.cache[key] = value

    def snapshot(self) -> dict[str, str]:
        """직렬화용 메모리 스냅샷.

        worker가 single-flight 결과를 공개하는 동안 json.dumps(cache)가 dict를
        순회하면 "dictionary changed size"로 잡 전체가 실패할 수 있다. 잠금은
        메모리 스냅샷까지만 잡고 디스크 I/O 동안에는 worker를 막지 않는다.
        """
        with self._cache_lock:
            return dict(self.cache)

    def purge(self, matches: Callable[[str], bool]) -> None:
        """값이 조건에 맞는 캐시 항목을 제거한다 (축퇴 출력 무효화용)."""
        with self._cache_lock:
            for ckey in [k for k, v in self.cache.items() if matches(v)]:
                self.cache.pop(ckey, None)

    def acquire(self, key: str) -> tuple[Flight, bool]:
        """키의 flight를 얻는다. (flight, owner) — owner면 이 스레드가 번역을 수행한다."""
        with self._inflight_lock:
            flight = self._inflight.get(key)
            if flight is None:
                flight = self._inflight[key] = Flight()
                return flight, True
        return flight, False
