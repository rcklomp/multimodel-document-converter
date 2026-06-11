#!/usr/bin/env python3
"""Localhost TCP relay to bridge conda-python -> LAN inference servers.

Context (2026-06-11): the mmrag-v2 conda python on this Mac Mini cannot reach the
M5 (10.0.10.235:8000) or GX10 (10.0.10.239:8001) inference servers - EHOSTUNREACH,
a utun/VPN scoped-route fault that affects that interpreter specifically - while
the SYSTEM python (this process) and curl reach them fine, and conda python reaches
127.0.0.1 fine. So this relay (run by /usr/bin/python3, on the working side) listens
on localhost and forwards to the servers; the conda env points its endpoints at the
local ports. No server or routing reconfiguration.

  127.0.0.1:18000  ->  10.0.10.235:8000   (M5 Qwen)
  127.0.0.1:18001  ->  10.0.10.239:8001   (GX10 MinerU)

Run: /usr/bin/python3 /tmp/phase5_relay.py
"""

import asyncio

MAPPINGS = [
    ("127.0.0.1", 18000, "10.0.10.235", 8000),
    ("127.0.0.1", 18001, "10.0.10.239", 8001),
]


async def pipe(reader, writer):
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except Exception:
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def handle(local_reader, local_writer, dst_host, dst_port):
    try:
        remote_reader, remote_writer = await asyncio.open_connection(dst_host, dst_port)
    except Exception as e:
        print(f"[relay] connect to {dst_host}:{dst_port} failed: {e}", flush=True)
        local_writer.close()
        return
    await asyncio.gather(
        pipe(local_reader, remote_writer),
        pipe(remote_reader, local_writer),
    )


async def main():
    servers = []
    for lhost, lport, dhost, dport in MAPPINGS:
        srv = await asyncio.start_server(
            lambda r, w, dh=dhost, dp=dport: handle(r, w, dh, dp), lhost, lport
        )
        print(f"[relay] listening {lhost}:{lport} -> {dhost}:{dport}", flush=True)
        servers.append(srv)
    await asyncio.gather(*(s.serve_forever() for s in servers))


if __name__ == "__main__":
    asyncio.run(main())
