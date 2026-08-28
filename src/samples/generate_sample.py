#!/usr/bin/env python3
#
# Generates the sample dataset shipped with Squey.
#
# The file is committed alongside this script; run it again only to change what
# the sample shows. A fixed seed keeps successive runs identical, so that
# regenerating it does not produce a needlessly different file.
#
# Usage: python3 generate_sample.py [output.parquet]

import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROWS = 150_000
SEED = 20260827

# A week of traffic, so that the time axis shows day/night and weekday/weekend.
START = np.datetime64("2026-03-02T00:00:00")
DAY_MS = 24 * 3600 * 1000
WEEK_MS = 7 * DAY_MS

rng = np.random.default_rng(SEED)

COUNTRIES = ["FR", "DE", "US", "GB", "NL", "ES", "IT", "PL", "SE", "CA", "JP", "BR"]
COUNTRY_WEIGHTS = np.array([28, 14, 16, 9, 6, 5, 5, 4, 3, 4, 3, 3], dtype=float)
COUNTRY_WEIGHTS /= COUNTRY_WEIGHTS.sum()

HOSTS = [
    "www.example.com", "api.example.com", "cdn.example.com", "mail.example.com",
    "intranet.example.com", "vpn.example.com", "db-primary.example.com",
    "search.example.com", "auth.example.com", "static.example.com",
]
HOST_WEIGHTS = np.array([22, 20, 15, 8, 9, 4, 5, 7, 6, 4], dtype=float)
HOST_WEIGHTS /= HOST_WEIGHTS.sum()

PATHS = [
    "/", "/index.html", "/api/v2/search", "/api/v2/users", "/api/v2/orders",
    "/login", "/logout", "/static/app.js", "/static/theme.css", "/assets/logo.svg",
    "/health", "/metrics", "/download/report.pdf", "/upload", "/admin/console",
]

AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Firefox/136.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6) Safari/18.3",
    "Mozilla/5.0 (X11; Linux x86_64) Chrome/134.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_3) Safari/18.3",
    "curl/8.12.0",
    "python-requests/2.32",
    "Squey-monitoring/1.4",
]
AGENT_WEIGHTS = np.array([26, 18, 30, 12, 6, 5, 3], dtype=float)
AGENT_WEIGHTS /= AGENT_WEIGHTS.sum()


def business_hours_times(count):
    """Timestamps over the week, denser during working hours."""
    base = rng.integers(0, WEEK_MS, size=count)
    hour = (base % DAY_MS) / DAY_MS * 24
    # Rejection sampling against a daily activity curve: quiet at night, busy
    # between 9 and 18, with a dip at lunchtime.
    weight = 0.12 + 0.88 * np.exp(-0.5 * ((hour - 13.5) / 4.2) ** 2)
    weight *= 1 - 0.35 * np.exp(-0.5 * ((hour - 12.5) / 0.6) ** 2)
    keep = rng.random(count) < weight
    # Whatever the curve rejected is redrawn uniformly rather than dropped, so
    # the row count stays exact.
    base[~keep] = rng.integers(0, WEEK_MS, size=int((~keep).sum()))
    return base


def ipv4(prefix, count, octets=2):
    """Random addresses under a /16 or /24 prefix, as strings."""
    if octets == 2:
        third = rng.integers(0, 32, size=count)
        fourth = rng.integers(1, 255, size=count)
        return np.char.add(np.char.add(prefix, third.astype(str)),
                           np.char.add(".", fourth.astype(str)))
    fourth = rng.integers(1, 255, size=count)
    return np.char.add(prefix, fourth.astype(str))


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "sample-network-traffic.parquet"

    n = ROWS
    offsets = business_hours_times(n)

    src_ip = ipv4("10.0.", n)
    dst_host = rng.choice(HOSTS, size=n, p=HOST_WEIGHTS)
    dst_ip = ipv4("192.168.4.", n, octets=1)
    country = rng.choice(COUNTRIES, size=n, p=COUNTRY_WEIGHTS)
    user_agent = rng.choice(AGENTS, size=n, p=AGENT_WEIGHTS)
    url_path = rng.choice(PATHS, size=n)

    # Ports follow the service, so that filtering on one narrows the other --
    # a correlation the parallel view is meant to reveal.
    port_pool = np.array([443, 443, 443, 80, 80, 8080, 22, 53, 5432, 25])
    dst_port = rng.choice(port_pool, size=n).astype(np.int32)
    protocol = np.where(dst_port == 53, "UDP", "TCP")

    method_pool = np.array(["GET", "GET", "GET", "GET", "POST", "POST", "PUT", "DELETE", "HEAD"])
    http_method = rng.choice(method_pool, size=n)

    # Server errors are kept rare here, so that the outage planted below reads
    # as a spike rather than as more of the same.
    status_values = np.array([200, 204, 301, 302, 304, 400, 401, 403, 404, 500, 503])
    status_weights = np.array([70, 4, 4, 5, 6, 2, 2, 1.5, 4.5, 0.6, 0.4])
    status_weights /= status_weights.sum()
    http_status = rng.choice(status_values, size=n, p=status_weights).astype(np.int32)

    # Log-normal sizes: most responses small, a long tail of large ones.
    bytes_received = (rng.lognormal(mean=7.6, sigma=1.5, size=n)).astype(np.int32)
    bytes_sent = (rng.lognormal(mean=5.9, sigma=1.2, size=n)).astype(np.int32)
    duration_ms = (rng.lognormal(mean=3.9, sigma=1.1, size=n)).astype(np.int32)

    # --- Three stories planted in the noise, each visible as a shape ---

    # A port scan: one host sweeping a wide port range in a short window.
    scan = rng.choice(n, size=1800, replace=False)
    offsets[scan] = 2 * DAY_MS + 3 * 3600 * 1000 + rng.integers(0, 20 * 60 * 1000, size=scan.size)
    src_ip[scan] = "10.0.14.87"
    dst_ip[scan] = ipv4("192.168.4.", scan.size, octets=1)
    dst_port[scan] = rng.integers(1, 9000, size=scan.size).astype(np.int32)
    protocol[scan] = "TCP"
    http_status[scan] = 403
    bytes_sent[scan] = rng.integers(40, 120, size=scan.size)
    bytes_received[scan] = 0
    duration_ms[scan] = rng.integers(1, 12, size=scan.size)
    country[scan] = "FR"
    user_agent[scan] = "curl/8.12.0"
    url_path[scan] = "/"

    # An exfiltration: one host pushing very large payloads, at night, abroad.
    leak = rng.choice(n, size=700, replace=False)
    offsets[leak] = 4 * DAY_MS + 2 * 3600 * 1000 + rng.integers(0, 2 * 3600 * 1000, size=leak.size)
    src_ip[leak] = "10.0.7.31"
    dst_host[leak] = "backup-mirror.example.net"
    dst_ip[leak] = "203.0.113.45"
    dst_port[leak] = 443
    protocol[leak] = "TCP"
    http_method[leak] = "POST"
    http_status[leak] = 200
    url_path[leak] = "/upload"
    bytes_sent[leak] = rng.integers(5_000_000, 50_000_000, size=leak.size)
    bytes_received[leak] = rng.integers(120, 400, size=leak.size)
    duration_ms[leak] = rng.integers(4_000, 30_000, size=leak.size)
    country[leak] = "RU"
    user_agent[leak] = "python-requests/2.32"

    # An outage: one service failing for two hours.
    outage = rng.choice(n, size=2600, replace=False)
    offsets[outage] = 5 * DAY_MS + 14 * 3600 * 1000 + rng.integers(0, 2 * 3600 * 1000, size=outage.size)
    dst_host[outage] = "api.example.com"
    dst_port[outage] = 443
    protocol[outage] = "TCP"
    failing = rng.random(outage.size) < 0.7
    http_status[outage] = np.where(failing, 500, 200).astype(np.int32)
    duration_ms[outage] = np.where(failing,
                                   rng.integers(8_000, 30_000, size=outage.size),
                                   rng.integers(20, 300, size=outage.size)).astype(np.int32)
    bytes_received[outage] = np.where(failing, 512, bytes_received[outage]).astype(np.int32)

    time = START + offsets.astype("timedelta64[ms]")
    order = np.argsort(time, kind="stable")

    table = pa.table({
        "time": pa.array(time[order], type=pa.timestamp("ms")),
        "src_ip": pa.array(src_ip[order]).dictionary_encode(),
        "dst_ip": pa.array(dst_ip[order]).dictionary_encode(),
        "dst_host": pa.array(dst_host[order]).dictionary_encode(),
        "dst_port": pa.array(dst_port[order], type=pa.int32()),
        "protocol": pa.array(protocol[order]).dictionary_encode(),
        "http_method": pa.array(http_method[order]).dictionary_encode(),
        "http_status": pa.array(http_status[order], type=pa.int32()),
        "url_path": pa.array(url_path[order]).dictionary_encode(),
        "bytes_sent": pa.array(bytes_sent[order], type=pa.int32()),
        "bytes_received": pa.array(bytes_received[order], type=pa.int32()),
        "duration_ms": pa.array(duration_ms[order], type=pa.int32()),
        "country": pa.array(country[order]).dictionary_encode(),
        "user_agent": pa.array(user_agent[order]).dictionary_encode(),
    })

    pq.write_table(table, out, compression="zstd", version="2.6")
    print(f"{out}: {table.num_rows} rows, {table.num_columns} columns")


if __name__ == "__main__":
    main()
