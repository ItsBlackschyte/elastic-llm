"""Job ledger: the coordination brain.

One row per training job. A worker must hold the job's *lease* to train it --
this is what makes it impossible for two sessions (you + friend) to clobber
each other. Leases expire automatically, so a dead worker frees the job
without any cleanup.

Works with SQLite (local dev: sqlite:///ledger.db) and Postgres
(Neon free tier: postgresql://...). Same code, different URL.

CLI:
  python ledger.py create-job --name run1 --preset 100m --data-dir data/fineweb2b \
      --hf-repo you/elastic-100m --target-tokens 2_000_000_000
  python ledger.py list
  python ledger.py pause <job_id> | resume <job_id>
"""
import argparse
import datetime as dt
import json
import os
import uuid

from sqlalchemy import (BigInteger, Column, DateTime, Float, Integer,
                        MetaData, String, Table, Text, create_engine, select)

meta = MetaData()

jobs = Table(
    "jobs", meta,
    Column("id", String(36), primary_key=True),
    Column("name", String(80)),
    Column("status", String(16), default="active"),  # active | paused | done
    Column("preset", String(16)),
    Column("data_dir", Text),
    Column("hf_repo", Text),
    Column("target_tokens", BigInteger),
    Column("config_json", Text),          # extra CLI args as JSON
    Column("leased_by", String(80), nullable=True),
    Column("lease_expires", DateTime, nullable=True),
    Column("created_at", DateTime),
)

metrics = Table(
    "metrics", meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("job_id", String(36), index=True),
    Column("worker_id", String(80)),
    Column("step", Integer),
    Column("loss", Float),
    Column("tokens_seen", BigInteger),
    Column("tok_per_sec", Float),
    Column("ts", DateTime, index=True),
)


def now():
    return dt.datetime.utcnow()


class Ledger:
    def __init__(self, url=None):
        url = url or os.environ.get("LEDGER_URL", "sqlite:///ledger.db")
        self.engine = create_engine(url, pool_pre_ping=True)
        meta.create_all(self.engine)

    # ---- jobs ----
    def create_job(self, name, preset, data_dir, hf_repo, target_tokens, config=None):
        jid = str(uuid.uuid4())[:8]
        with self.engine.begin() as c:
            c.execute(jobs.insert().values(
                id=jid, name=name, status="active", preset=preset,
                data_dir=data_dir, hf_repo=hf_repo, target_tokens=target_tokens,
                config_json=json.dumps(config or {}), created_at=now()))
        return jid

    def get_job(self, job_id):
        with self.engine.begin() as c:
            row = c.execute(select(jobs).where(jobs.c.id == job_id)).mappings().first()
        return dict(row) if row else None

    def list_jobs(self):
        with self.engine.begin() as c:
            return [dict(r) for r in c.execute(
                select(jobs).order_by(jobs.c.created_at.desc())).mappings()]

    def set_status(self, job_id, status):
        assert status in ("active", "paused", "done")
        with self.engine.begin() as c:
            c.execute(jobs.update().where(jobs.c.id == job_id).values(status=status))

    def update_config(self, job_id, config: dict):
        with self.engine.begin() as c:
            c.execute(jobs.update().where(jobs.c.id == job_id)
                      .values(config_json=json.dumps(config)))

    # ---- leases (the mutual-exclusion mechanism) ----
    def claim(self, job_id, worker_id, ttl_minutes=15):
        """Atomically claim the job. True iff no live lease held by someone else."""
        with self.engine.begin() as c:
            r = c.execute(
                jobs.update()
                .where(jobs.c.id == job_id)
                .where((jobs.c.leased_by.is_(None))
                       | (jobs.c.lease_expires < now())
                       | (jobs.c.leased_by == worker_id))
                .values(leased_by=worker_id,
                        lease_expires=now() + dt.timedelta(minutes=ttl_minutes)))
        return r.rowcount == 1

    def release(self, job_id, worker_id):
        with self.engine.begin() as c:
            c.execute(jobs.update()
                      .where(jobs.c.id == job_id, jobs.c.leased_by == worker_id)
                      .values(leased_by=None, lease_expires=None))

    # ---- heartbeat: renew lease + log metrics + learn desired status ----
    def heartbeat(self, job_id, worker_id, step, loss, tokens_seen, tps,
                  ttl_minutes=15):
        ok = self.claim(job_id, worker_id, ttl_minutes)  # renew == re-claim
        with self.engine.begin() as c:
            c.execute(metrics.insert().values(
                job_id=job_id, worker_id=worker_id, step=step, loss=loss,
                tokens_seen=tokens_seen, tok_per_sec=tps, ts=now()))
            status = c.execute(
                select(jobs.c.status).where(jobs.c.id == job_id)).scalar()
        return {"lease_ok": ok, "status": status}

    def recent_metrics(self, job_id, limit=500):
        with self.engine.begin() as c:
            rows = c.execute(select(metrics).where(metrics.c.job_id == job_id)
                             .order_by(metrics.c.id.desc()).limit(limit)).mappings()
            return [dict(r) for r in rows][::-1]

    def worker_last_seen(self, job_id):
        out = {}
        for m in self.recent_metrics(job_id, 2000):
            out[m["worker_id"]] = m
        return out


def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["create-job", "list", "pause", "resume"])
    ap.add_argument("job_id", nargs="?")
    ap.add_argument("--url", default=None)
    ap.add_argument("--name")
    ap.add_argument("--preset", default="100m")
    ap.add_argument("--data-dir")
    ap.add_argument("--hf-repo")
    ap.add_argument("--target-tokens", type=lambda s: int(s.replace("_", "")))
    args = ap.parse_args()
    led = Ledger(args.url)
    if args.cmd == "create-job":
        jid = led.create_job(args.name, args.preset, args.data_dir,
                             args.hf_repo, args.target_tokens)
        print(f"created job {jid}")
    elif args.cmd == "list":
        for j in led.list_jobs():
            print(f"{j['id']}  {j['name']:<16} {j['status']:<8} "
                  f"lease={j['leased_by']}")
    else:
        led.set_status(args.job_id, "paused" if args.cmd == "pause" else "active")
        print(f"{args.job_id} -> {args.cmd}d")


if __name__ == "__main__":
    _cli()
