import ipaddress
import os
import threading
import time
from urllib.parse import urlparse

try:
    import requests
except ImportError:
    requests = None

from sources.logger import Logger


logger = Logger("backend.log")
USPTO_API_HOST = "api.uspto.gov"
USPTO_API_CONCURRENCY_TIMEOUT_SECONDS = 1.0
_uspto_api_semaphore = threading.BoundedSemaphore(value=1)


class OutboundHttpError(Exception):
    """Base error for centralized outbound HTTP policy failures."""


class OutboundHttpBlockedError(OutboundHttpError):
    """Raised when an outbound URL is blocked by local policy."""


class OutboundHttpConcurrencyTimeoutError(OutboundHttpError):
    """Raised when a host-specific outbound concurrency slot is unavailable."""


def _domain_blacklist() -> set[str]:
    raw_value = os.getenv("OUTBOUND_HTTP_DOMAIN_BLACKLIST", "")
    return {
        item.strip().lower().rstrip(".")
        for item in raw_value.split(",")
        if item.strip()
    }


def _is_localhost(hostname: str) -> bool:
    normalized = hostname.lower().rstrip(".")
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def validate_outbound_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise OutboundHttpBlockedError("Only http and https outbound URLs are allowed")

    hostname = parsed.hostname
    if not hostname:
        raise OutboundHttpBlockedError("Outbound URL must include a hostname")

    if _is_localhost(hostname):
        raise OutboundHttpBlockedError("Outbound HTTP to localhost is blocked")

    normalized_host = hostname.lower().rstrip(".")
    if normalized_host in _domain_blacklist():
        raise OutboundHttpBlockedError("Outbound HTTP domain is blocked")


def _is_uspto_api_url(url: str) -> bool:
    hostname = urlparse(url).hostname
    return bool(hostname and hostname.lower().rstrip(".") == USPTO_API_HOST)


def _acquire_uspto_api_slot(url: str) -> bool:
    if not _is_uspto_api_url(url):
        return False
    acquired = _uspto_api_semaphore.acquire(timeout=USPTO_API_CONCURRENCY_TIMEOUT_SECONDS)
    if not acquired:
        raise OutboundHttpConcurrencyTimeoutError(
            f"Outbound HTTP concurrency limit reached for {USPTO_API_HOST}"
        )
    return True


class OutboundHttpClient:
    # Status codes that trigger an automatic retry for USPTO API calls.
    # 429 is the official rate-limit code; 400 is often returned by USPTO
    # for transient issues (throttling, URL expiry, service instability).
    _USPTO_RETRYABLE_STATUSES: set[int] = {400, 429}
    _USPTO_MAX_RETRIES: int = 10
    _USPTO_RETRY_DELAY_SECONDS: float = 1.0

    def request(self, method: str, url: str, *, purpose: str = "general", **kwargs):
        global requests
        if requests is None:
            import requests as requests_module

            requests = requests_module

        validate_outbound_url(url)

        is_uspto = _is_uspto_api_url(url)
        max_attempts = self._USPTO_MAX_RETRIES if is_uspto else 1

        last_status = None
        for attempt in range(max_attempts):
            acquired_uspto_slot = False
            if is_uspto:
                acquired_uspto_slot = _acquire_uspto_api_slot(url)
            try:
                started_at = time.monotonic()
                response = requests.request(method, url, **kwargs)
                elapsed_ms = int((time.monotonic() - started_at) * 1000)
                self._log_request(method, url, purpose, response.status_code, elapsed_ms)

                if (
                    is_uspto
                    and response.status_code in self._USPTO_RETRYABLE_STATUSES
                    and attempt + 1 < max_attempts
                ):
                    last_status = response.status_code
                    logger.info(
                        f"outbound_http uspto_{response.status_code}_retry "
                        f"attempt={attempt+1}/{max_attempts} url={url[:120]}"
                    )
                    time.sleep(self._USPTO_RETRY_DELAY_SECONDS)
                    continue

                return response
            finally:
                if acquired_uspto_slot:
                    _uspto_api_semaphore.release()

        raise RuntimeError(
            f"USPTO retries exhausted (status={last_status}) for {url[:120]}"
        )

    async def arequest(self, method: str, url: str, *, purpose: str = "general", **kwargs):
        import asyncio
        import httpx

        validate_outbound_url(url)

        is_uspto = _is_uspto_api_url(url)
        max_attempts = self._USPTO_MAX_RETRIES if is_uspto else 1

        last_status = None
        for attempt in range(max_attempts):
            acquired_uspto_slot = False
            if is_uspto:
                acquired_uspto_slot = await asyncio.to_thread(
                    _uspto_api_semaphore.acquire,
                    True,
                    USPTO_API_CONCURRENCY_TIMEOUT_SECONDS,
                )
                if not acquired_uspto_slot:
                    raise OutboundHttpConcurrencyTimeoutError(
                        f"Outbound HTTP concurrency limit reached for {USPTO_API_HOST}"
                    )
            try:
                started_at = time.monotonic()
                async with httpx.AsyncClient() as client:
                    response = await client.request(method, url, **kwargs)
                elapsed_ms = int((time.monotonic() - started_at) * 1000)
                self._log_request(method, url, purpose, response.status_code, elapsed_ms)

                if (
                    is_uspto
                    and response.status_code in self._USPTO_RETRYABLE_STATUSES
                    and attempt + 1 < max_attempts
                ):
                    last_status = response.status_code
                    logger.info(
                        f"outbound_http uspto_{response.status_code}_retry "
                        f"attempt={attempt+1}/{max_attempts} url={url[:120]}"
                    )
                    await asyncio.sleep(self._USPTO_RETRY_DELAY_SECONDS)
                    continue

                return response
            finally:
                if acquired_uspto_slot:
                    _uspto_api_semaphore.release()

        raise RuntimeError(
            f"USPTO retries exhausted (status={last_status}) for {url[:120]}"
        )

    def get(self, url: str, *, purpose: str = "general", **kwargs):
        return self.request("GET", url, purpose=purpose, **kwargs)

    def post(self, url: str, *, purpose: str = "general", **kwargs):
        return self.request("POST", url, purpose=purpose, **kwargs)

    def put(self, url: str, *, purpose: str = "general", **kwargs):
        return self.request("PUT", url, purpose=purpose, **kwargs)

    def delete(self, url: str, *, purpose: str = "general", **kwargs):
        return self.request("DELETE", url, purpose=purpose, **kwargs)

    def patch(self, url: str, *, purpose: str = "general", **kwargs):
        return self.request("PATCH", url, purpose=purpose, **kwargs)

    async def apost(self, url: str, *, purpose: str = "general", **kwargs):
        return await self.arequest("POST", url, purpose=purpose, **kwargs)

    def _log_request(self, method: str, url: str, purpose: str, status_code: int, elapsed_ms: int) -> None:
        parsed = urlparse(url)
        logger.info(
            "outbound_http "
            f"purpose={purpose} method={method.upper()} "
            f"host={parsed.hostname} path={parsed.path or '/'} "
            f"status={status_code} elapsed_ms={elapsed_ms}"
        )


outbound_http = OutboundHttpClient()
