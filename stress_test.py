import asyncio
import httpx
import time
import sys
import numpy as np

# Allow switching ports from the command line: python stress_test.py 8001
PORT = sys.argv[1] if len(sys.argv) > 1 else "8000"
URL = f"http://localhost:{PORT}/predict"

# Use normalized ASR-style inputs (lowercase, unpunctuated) for realistic testing
SAMPLES = [
    "yeah that works fine for me",
    "hold on let me check the",
    "can you call me back around five oclock",
    "well if you really think that we should",
    "sounds good thank you",
]


async def send_request(client: httpx.AsyncClient):
    payload = {"text": np.random.choice(SAMPLES), "threshold": 0.5}
    start = time.perf_counter()
    try:
        resp = await client.post(URL, json=payload, timeout=2.0)
        latency = (time.perf_counter() - start) * 1000
        return resp.status_code, latency
    except Exception:
        return 500, None


async def run_load_test(total_requests: int = 500, concurrency: int = 25):
    limits = httpx.Limits(max_keepalive_connections=concurrency, max_connections=concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        start_time = time.perf_counter()
        sem = asyncio.Semaphore(concurrency)

        async def worker():
            async with sem:
                return await send_request(client)

        tasks = [worker() for _ in range(total_requests)]
        results = await asyncio.gather(*tasks)
        total_wall_time = time.perf_counter() - start_time

    status_codes = [r[0] for r in results]
    latencies = [r[1] for r in results if r[1] is not None]

    print(f"\n--- Stress Test Results for Port {PORT} ---")
    print(f"Total Requests:      {total_requests}")
    print(f"Concurrency Level:   {concurrency}")
    print(f"Successful Requests: {status_codes.count(200)}")
    print(f"Failed Requests:     {len(status_codes) - status_codes.count(200)}")
    print(f"Throughput (QPS):    {total_requests / total_wall_time:.2f} req/sec")
    print(f"Latency P50:         {np.percentile(latencies, 50):.2f} ms")
    print(f"Latency P95:         {np.percentile(latencies, 95):.2f} ms")
    print(f"Latency P99:         {np.percentile(latencies, 99):.2f} ms")


if __name__ == "__main__":
    asyncio.run(run_load_test())