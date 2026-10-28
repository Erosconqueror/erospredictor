"""Rate-limit regression checks with fake time and no live Riot requests."""

import unittest
from unittest.mock import patch

from services.riot_rate_limit import RiotRateLimiter, retry_after_seconds, shared_limiter


class FakeTime:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, delay):
        self.sleeps.append(delay)
        self.now += delay


class FakeError(Exception):
    def __init__(self, status, headers=None):
        self.response = type("Response", (), {"status_code": status, "headers": headers or {}})()


class LimiterTests(unittest.TestCase):
    def setUp(self):
        self.time = FakeTime()
        self.limiter = RiotRateLimiter(clock=self.time.clock, sleeper=self.time.sleep,
                                      logger=None, wall_clock=lambda: 0)

    def test_twenty_per_second_is_shared_by_different_endpoints(self):
        for index in range(20):
            self.limiter.acquire("europe", "match" if index % 2 else "account", "fetch")
        self.assertEqual(self.time.now, 0)
        self.limiter.acquire("europe", "match", "fetch")
        self.assertEqual(self.time.now, 1)

    def test_hundred_per_two_minutes_limits_high_match_method_limit(self):
        self.limiter.record_response("europe", "match", "fetch", 200,
                                     {"X-Method-Rate-Limit": "2000:10"})
        for _ in range(100):
            self.limiter.acquire("europe", "match", "fetch")
        self.assertEqual(self.time.now, 4)
        self.limiter.acquire("europe", "match", "fetch")
        self.assertEqual(self.time.now, 120)

    def test_routes_have_distinct_application_budgets(self):
        for _ in range(20):
            self.limiter.acquire("europe", "match", "fetch")
        self.limiter.acquire("euw1", "league", "entries")
        self.assertEqual(self.time.now, 0)

    def test_server_count_from_another_process_is_respected(self):
        self.limiter.record_response("europe", "match", "fetch", 200, {
            "X-App-Rate-Limit": "20:1,100:120", "X-App-Rate-Limit-Count": "1:1,100:120"})
        self.limiter.acquire("europe", "account", "fetch")
        self.assertEqual(self.time.now, 120)

    def test_method_limit_applies_only_to_its_method(self):
        self.limiter.record_response("europe", "match", "fetch", 200,
                                     {"X-Method-Rate-Limit": "1:10", "X-Method-Rate-Limit-Count": "1:10"})
        self.limiter.acquire("europe", "match", "list")
        self.assertEqual(self.time.now, 0)
        self.limiter.acquire("europe", "match", "fetch")
        self.assertEqual(self.time.now, 10)

    def test_retry_after_is_never_shortened_and_chunks_keep_progress_possible(self):
        attempts = []
        def operation():
            self.limiter.acquire("europe", "match", "fetch")
            attempts.append(self.time.now)
            if len(attempts) == 1:
                raise FakeError(429, {"Retry-After": "125"})
            return "success"
        self.assertEqual(self.limiter.call(operation, "europe"), "success")
        self.assertEqual(attempts, [0, 125])
        self.assertLessEqual(max(self.time.sleeps), 30)

    def test_retry_exhaustion_is_bounded(self):
        attempts = []
        def operation():
            self.limiter.acquire("europe", "match", "fetch")
            attempts.append(self.time.now)
            raise FakeError(429, {"Retry-After": "2"})
        with self.assertRaises(FakeError):
            self.limiter.call(operation, "europe", max_retries=2)
        self.assertEqual(attempts, [0, 2, 4])

    def test_missing_or_malformed_retry_header_uses_backoff(self):
        for headers in ({}, {"Retry-After": "bad-value"}):
            with self.subTest(headers=headers):
                time = FakeTime()
                limiter = RiotRateLimiter(clock=time.clock, sleeper=time.sleep, logger=None)
                attempts = []
                def operation():
                    limiter.acquire("europe", "match", "fetch")
                    attempts.append(time.now)
                    if len(attempts) < 3:
                        raise FakeError(429, headers)
                    return True
                self.assertTrue(limiter.call(operation, "europe"))
                self.assertEqual(attempts, [0, 5, 15])

    def test_auth_and_other_errors_are_not_retried(self):
        for status in (401, 403, 404, 500):
            with self.subTest(status=status):
                calls = []
                def operation():
                    calls.append(status)
                    raise FakeError(status)
                with self.assertRaises(FakeError):
                    self.limiter.call(operation, "europe")
                self.assertEqual(calls, [status])

    def test_retry_after_parses_http_date_and_rejects_non_finite(self):
        self.assertEqual(retry_after_seconds("Thu, 01 Jan 1970 00:02:00 GMT", 0), 120)
        self.assertEqual(retry_after_seconds("1.5"), 1.5)
        for value in (None, "nan", "inf", "-5", "garbage"):
            self.assertIsNone(retry_after_seconds(value))

    def test_bad_headers_do_not_remove_default_safety_limits(self):
        self.limiter.record_response("europe", "match", "fetch", 200,
                                     {"X-App-Rate-Limit": "bad,0:1,10000:120,1:nan"})
        for _ in range(20):
            self.limiter.acquire("europe", "match", "fetch")
        self.limiter.acquire("europe", "match", "fetch")
        self.assertEqual(self.time.now, 1)

    def test_same_key_instances_share_gate_without_plaintext_registry_keys(self):
        self.assertIs(shared_limiter("test-application-one"), shared_limiter("test-application-one"))
        self.assertIsNot(shared_limiter("test-application-one"), shared_limiter("test-application-two"))


class RiotWatcherIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.time = FakeTime()
        self.limiter = RiotRateLimiter(clock=self.time.clock, sleeper=self.time.sleep, logger=None)

    @staticmethod
    def response(status, body=b"[]", headers=None):
        import requests
        response = requests.Response()
        response.status_code, response._content = status, body
        response.headers.update(headers or {})
        response.url = "https://europe.api.riotgames.com/lol/match/v5/matches/by-puuid/test/ids"
        return response

    def test_real_watcher_adapter_retries_same_call_and_counts_actual_attempts_once(self):
        from model.riot import Riot
        riot = Riot(api_key="test-not-a-real-key", strict_errors=True, rate_limiter=self.limiter)
        responses = [self.response(429, headers={"Retry-After": "3"}), self.response(200)]
        with patch.object(riot.api._base_api._session, "get", side_effect=responses) as request:
            self.assertEqual(riot.get_match_ids("test-player"), [])
        self.assertEqual(request.call_count, 2)
        self.assertEqual(self.time.now, 3)
        self.assertEqual(list(self.limiter._apps["europe"]["attempts"]), [0, 3])

    def test_direct_account_request_uses_the_same_limiter_and_retry(self):
        from model.riot import Riot
        riot = Riot(api_key="test-not-a-real-key", strict_errors=True, rate_limiter=self.limiter)
        responses = [self.response(429, headers={"Retry-After": "2"}), self.response(200, b'{}')]
        with patch("model.riot.requests.get", side_effect=responses) as request:
            self.assertEqual(riot.get_account("test-name", "tag"), {})
        self.assertEqual(request.call_count, 2)
        self.assertEqual(self.time.now, 2)


if __name__ == "__main__":
    unittest.main()
