import asyncio
import httpx
import time
import numpy as np

URL = "http://localhost:8000/predict"
SAMPLES = [
    "Yeah that works for me",
    "Hold on let me check the",
    "Can you call me back around five o'clock",
    "Well if you really think that we should",
    "Sounds good thank you",
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

    print("\n--- Stress Test Results ---")
    print(f"Total Requests:      {total_requests}")
    print(f"Concurrency Level:   {concurrency}")
    print(f"Successful Requests: {status_codes.count(200)}")
    print(f"Failed Requests:     {len(status_codes) - status_codes.count(200)}")
    print(f"Total Wall Clock:    {total_wall_time:.2f} s")
    print(f"Throughput (QPS):    {total_requests / total_wall_time:.2f} req/sec")
    print(f"Latency P50:         {np.percentile(latencies, 50):.2f} ms")
    print(f"Latency P95:         {np.percentile(latencies, 95):.2f} ms")
    print(f"Latency P99:         {np.percentile(latencies, 99):.2f} ms")


if __name__ == "__main__":
    asyncio.run(run_load_test())