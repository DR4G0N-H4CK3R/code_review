"""
Flow-record sources.

Week-7 deliverable "real data integration". Three adapters share one output
format so nothing downstream changes when you swap the source:

    FlowRecord(ts, src, dst, sport, dport, proto, bytes_, pkts, duration, label)

  * `SyntheticSource`   - runs today, no download. Models a home/office subnet
                          with device roles, then injects lateral movement.
  * `CICIoT2023Source`  - the CSV release. Per-flow labels, no MAC columns, so
                          device identity is reconstructed from IP.
  * `IoT23Source`       - Zeek conn.log / labelled CSV from Stratosphere IPS.

Both real adapters are column-mapped, not column-guessed: if the header does not
contain the fields we need, the loader tells you which ones are missing instead
of silently producing garbage graphs.
"""
from __future__ import annotations

import csv
import gzip
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

BENIGN = 0
MALICIOUS = 1


@dataclass
class FlowRecord:
    ts: float
    src: str
    dst: str
    sport: int
    dport: int
    proto: str
    bytes_: float
    pkts: float
    duration: float
    label: int = BENIGN
    attack_type: str = "benign"


# --------------------------------------------------------------------------
# 1. Synthetic source
# --------------------------------------------------------------------------
DEVICE_ROLES = {
    "camera": {"peers": ["nvr", "gateway"], "ports": [554, 443], "bytes": (8e4, 3e4)},
    "bulb": {"peers": ["hub"], "ports": [5683, 443], "bytes": (2e3, 8e2)},
    "thermostat": {"peers": ["hub", "gateway"], "ports": [443], "bytes": (4e3, 1e3)},
    "speaker": {"peers": ["gateway"], "ports": [443, 123], "bytes": (2e4, 9e3)},
    "nas": {"peers": ["laptop", "gateway"], "ports": [445, 443], "bytes": (5e5, 2e5)},
    "laptop": {"peers": ["gateway", "nas", "printer"], "ports": [443, 445, 631], "bytes": (1e5, 6e4)},
    "printer": {"peers": ["laptop"], "ports": [631], "bytes": (3e4, 1e4)},
    "hub": {"peers": ["gateway"], "ports": [443], "bytes": (1e4, 4e3)},
    "nvr": {"peers": ["gateway"], "ports": [443], "bytes": (2e5, 8e4)},
    "gateway": {"peers": [], "ports": [443, 53], "bytes": (5e4, 2e4)},
}


class SyntheticSource:
    """
    A device-relationship generator, not a packet generator.

    The point of the project is that compromise shows up in WHO TALKS TO WHOM.
    So the benign process draws edges from a per-site role graph, and the attack
    process adds edges that are individually unremarkable (normal ports, normal
    byte counts) but structurally new - a camera reaching the NAS.
    """

    def __init__(
        self,
        site_id: int = 0,
        n_devices: int = 24,
        seed: int = 0,
        role_mix: Optional[Dict[str, float]] = None,
    ) -> None:
        self.site_id = site_id
        self.rng = random.Random(seed * 1000 + site_id)
        self.role_mix = role_mix or {
            "camera": 0.20, "bulb": 0.22, "thermostat": 0.08, "speaker": 0.10,
            "nas": 0.04, "laptop": 0.14, "printer": 0.04, "hub": 0.06,
            "nvr": 0.04, "gateway": 0.08,
        }
        self.devices: Dict[str, str] = {}   # device id -> role
        self._build_devices(n_devices)

    def _build_devices(self, n: int) -> None:
        roles = list(self.role_mix)
        weights = [self.role_mix[r] for r in roles]
        for i in range(n):
            role = self.rng.choices(roles, weights=weights, k=1)[0]
            self.devices[f"s{self.site_id}_{role}_{i}"] = role
        # every site needs at least one gateway and one hub
        for must in ("gateway", "hub"):
            if must not in self.devices.values():
                self.devices[f"s{self.site_id}_{must}_x"] = must

    def _by_role(self, role: str) -> List[str]:
        return [d for d, r in self.devices.items() if r == role]

    def generate(
        self,
        duration_s: float = 1800.0,
        attack_start_frac: float = 0.6,
        attack_kind: str = "lateral",
        attack_rate: float = 0.02,
    ) -> List[FlowRecord]:
        flows: List[FlowRecord] = []
        t = 0.0
        while t < duration_s:
            t += self.rng.expovariate(1.0 / 1.2)          # ~1 flow every 1.2 s
            src = self.rng.choice(list(self.devices))
            role = self.devices[src]
            peer_roles = DEVICE_ROLES[role]["peers"] or ["gateway"]
            peers: List[str] = []
            for pr in peer_roles:
                peers.extend(self._by_role(pr))
            if not peers:
                continue
            dst = self.rng.choice(peers)
            mu, sd = DEVICE_ROLES[role]["bytes"]
            flows.append(
                FlowRecord(
                    ts=t,
                    src=src,
                    dst=dst,
                    sport=self.rng.randint(32768, 60999),
                    dport=self.rng.choice(DEVICE_ROLES[role]["ports"]),
                    proto="tcp" if self.rng.random() < 0.8 else "udp",
                    bytes_=max(64.0, self.rng.gauss(mu, sd)),
                    pkts=max(1.0, self.rng.gauss(mu / 800.0, 4.0)),
                    duration=abs(self.rng.gauss(1.5, 0.9)),
                    label=BENIGN,
                )
            )

        flows.extend(self._inject(duration_s, attack_start_frac, attack_kind, attack_rate))
        flows.sort(key=lambda f: f.ts)
        return flows

    def _inject(self, duration_s, start_frac, kind, rate) -> List[FlowRecord]:
        """
        Lateral movement: a compromised low-privilege device (a bulb or camera)
        starts contacting peers its role has never contacted.
        """
        out: List[FlowRecord] = []
        candidates = self._by_role("camera") + self._by_role("bulb")
        if not candidates:
            return out
        victim = self.rng.choice(candidates)
        victim_role = self.devices[victim]

        # Ground-truth hygiene: only count an edge as lateral movement if the
        # victim's ROLE never legitimately talks to that peer role. Otherwise a
        # camera -> NVR flow would be labelled malicious while being structurally
        # identical to benign traffic, and the evaluation would be measuring noise.
        legit = set(DEVICE_ROLES[victim_role]["peers"])
        target_roles = {"nas", "laptop", "printer", "nvr", "thermostat"} - legit
        targets = [d for d, r in self.devices.items() if r in target_roles and d != victim]
        if not targets:
            return out

        t = duration_s * start_frac
        while t < duration_s:
            t += self.rng.expovariate(1.0 / (1.2 / max(rate, 1e-6)) ) if rate < 1 else 1.0
            if t >= duration_s:
                break
            dst = self.rng.choice(targets)
            out.append(
                FlowRecord(
                    ts=t,
                    src=victim,
                    dst=dst,
                    sport=self.rng.randint(32768, 60999),
                    dport=self.rng.choice([22, 445, 3389, 8080]),
                    proto="tcp",
                    # deliberately ordinary volumes: a per-flow IDS sees nothing
                    bytes_=max(64.0, self.rng.gauss(6e4, 2e4)),
                    pkts=max(1.0, self.rng.gauss(70, 20)),
                    duration=abs(self.rng.gauss(1.4, 0.7)),
                    label=MALICIOUS,
                    attack_type=kind,
                )
            )
        return out


# --------------------------------------------------------------------------
# 2 & 3. Real CSV adapters
# --------------------------------------------------------------------------
def _open(path: str):
    return gzip.open(path, "rt", newline="") if path.endswith(".gz") else open(path, newline="")


class _CsvSource:
    COLUMNS: Dict[str, Sequence[str]] = {}
    BENIGN_LABELS = {"benign", "normal", "background", "-", ""}

    def __init__(self, path: str, limit: Optional[int] = None) -> None:
        self.path = path
        self.limit = limit
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Download the dataset and pass --data <path>, "
                f"or run with --source synthetic."
            )

    def _resolve(self, header: Sequence[str]) -> Dict[str, str]:
        lower = {h.strip().lower(): h for h in header}
        mapping, missing = {}, []
        for field, candidates in self.COLUMNS.items():
            hit = next((lower[c] for c in candidates if c in lower), None)
            if hit is None:
                missing.append(f"{field} (tried {list(candidates)})")
            else:
                mapping[field] = hit
        if missing:
            raise KeyError(
                f"{self.__class__.__name__}: these fields are not in the CSV header:\n  "
                + "\n  ".join(missing)
                + f"\nheader was: {list(header)[:25]}"
            )
        return mapping

    @staticmethod
    def _num(v, default=0.0) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def load(self) -> List[FlowRecord]:
        out: List[FlowRecord] = []
        with _open(self.path) as fh:
            reader = csv.DictReader(fh)
            m = self._resolve(reader.fieldnames or [])
            for i, row in enumerate(reader):
                if self.limit and i >= self.limit:
                    break
                lab = str(row.get(m["label"], "")).strip().lower()
                out.append(
                    FlowRecord(
                        ts=self._num(row.get(m["ts"]), i * 0.001),
                        src=str(row.get(m["src"])),
                        dst=str(row.get(m["dst"])),
                        sport=int(self._num(row.get(m["sport"]))),
                        dport=int(self._num(row.get(m["dport"]))),
                        proto=str(row.get(m["proto"], "tcp")).lower(),
                        bytes_=self._num(row.get(m["bytes"])),
                        pkts=self._num(row.get(m["pkts"])),
                        duration=self._num(row.get(m["duration"])),
                        label=BENIGN if lab in self.BENIGN_LABELS else MALICIOUS,
                        attack_type=lab or "benign",
                    )
                )
        out.sort(key=lambda f: f.ts)
        return out


class CICIoT2023Source(_CsvSource):
    """CICIoT2023 merged CSV release."""

    COLUMNS = {
        "ts": ("ts", "timestamp", "flow_start", "time"),
        "src": ("src_ip", "source ip", "srcip", "id.orig_h", "src"),
        "dst": ("dst_ip", "destination ip", "dstip", "id.resp_h", "dst"),
        "sport": ("src_port", "source port", "id.orig_p", "sport"),
        "dport": ("dst_port", "destination port", "id.resp_p", "dport"),
        "proto": ("protocol", "proto", "protocol_type"),
        "bytes": ("total_size", "flow_bytes", "tot_size", "orig_bytes", "bytes"),
        "pkts": ("number", "tot_pkts", "orig_pkts", "packets", "pkts"),
        "duration": ("duration", "flow_duration"),
        "label": ("label", "attack", "class"),
    }


class IoT23Source(_CsvSource):
    """Stratosphere IoT-23 labelled conn.log exported to CSV."""

    COLUMNS = {
        "ts": ("ts", "timestamp"),
        "src": ("id.orig_h", "src_ip", "srcaddr"),
        "dst": ("id.resp_h", "dst_ip", "dstaddr"),
        "sport": ("id.orig_p", "src_port"),
        "dport": ("id.resp_p", "dst_port"),
        "proto": ("proto", "protocol"),
        "bytes": ("orig_bytes", "orig_ip_bytes", "bytes"),
        "pkts": ("orig_pkts", "packets"),
        "duration": ("duration",),
        "label": ("label", "detailed-label", "tunnel_parents   label   detailed-label"),
    }


def load_source(kind: str, **kw) -> List[FlowRecord]:
    kind = kind.lower()
    if kind == "synthetic":
        gen = SyntheticSource(
            site_id=kw.get("site_id", 0),
            n_devices=kw.get("n_devices", 24),
            seed=kw.get("seed", 0),
            role_mix=kw.get("role_mix"),
        )
        return gen.generate(
            duration_s=kw.get("duration_s", 1800.0),
            attack_start_frac=kw.get("attack_start_frac", 0.6),
            attack_rate=kw.get("attack_rate", 0.02),
        )
    if kind in ("ciciot2023", "ciciot"):
        return CICIoT2023Source(kw["path"], kw.get("limit")).load()
    if kind in ("iot23", "iot-23"):
        return IoT23Source(kw["path"], kw.get("limit")).load()
    raise ValueError(f"unknown source {kind!r}")
