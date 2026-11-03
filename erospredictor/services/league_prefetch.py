"""One cancellable speculative League-v4 page; main SQL/cursors stay single-threaded."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event


class PrefetchGate:
    def __init__(self, shared, stopped):
        self.shared, self.stopped = shared, stopped

    def wait_until(self, region, endpoint_name, method_name):
        self.shared.acquire(region, endpoint_name, method_name, cancelled=self.stopped.is_set)

    def record_response(self, *args, **kwargs):
        return self.shared.record_response(*args, **kwargs)

    def call(self, operation, region, max_retries=0, retry_transient=False):
        return self.shared.call(operation, region, max_retries=0)


class LeaguePrefetcher:
    def __init__(self, riot, region, continent):
        from model.riot import Riot
        self.stopped = Event()
        self.client = Riot(api_key=riot.api_key, strict_errors=True, max_rate_retries=0,
                           rate_limiter=PrefetchGate(riot.rate_limiter, self.stopped))
        self.client.region, self.client.cont = region, continent
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="league-prefetch")
        self.key, self.future = None, None

    def schedule(self, tier, division, page):
        key = tier, division, page
        if self.future is not None or self.stopped.is_set():
            return
        self.key = key
        self.future = self.pool.submit(self.client.get_league_exp_players, tier, division, page=page)

    def get(self, tier, division, page, fallback):
        if self.future is not None and self.key != (tier, division, page):
            self.future.cancel()
            self.key, self.future = None, None
        if self.key == (tier, division, page) and self.future is not None:
            future, self.future = self.future, None
            self.key = None
            try:
                return future.result()
            except Exception:
                # Failed speculation never advances the collection cursor.
                return fallback()
        return fallback()

    def close(self):
        self.stopped.set()
        self.pool.shutdown(wait=True, cancel_futures=True)
