"""Pin small official source files without downloading repository data archives."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
report_path = ROOT / "handoff/community-source-snapshots.json"
report = json.loads(report_path.read_text(encoding="utf-8"))
repos = {"decider": "Mapika/decider", "openjev": "lookski/openjev"}


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "jev-local-research"})
    with urllib.request.urlopen(request, timeout=40) as response:
        return response.read()


for name, repo in repos.items():
    destination = ROOT / "external" / (name + "-upstream")
    if name in report:
        print(f"Already pinned: {name}", flush=True)
        continue
    pin_file = ROOT / "handoff" / (name + "-source-fetch.json")
    if pin_file.exists():
        pin = json.loads(pin_file.read_text(encoding="utf-8"))
        commit = pin["commit"]
    else:
        metadata = json.loads(fetch(f"https://api.github.com/repos/{repo}"))
        commit = json.loads(fetch(f'https://api.github.com/repos/{repo}/commits/{metadata["default_branch"]}'))["sha"]
        pin = {"repo": repo, "commit": commit, "status": "fetching", "files": {}}
        pin_file.write_text(json.dumps(pin, indent=2), encoding="utf-8")
    tree = json.loads(fetch(f"https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1"))
    if tree.get("truncated"):
        raise ValueError("Git tree incomplete; refusing partial source discovery")
    entries = [entry for entry in tree["tree"] if entry["type"] == "blob"
               and entry.get("size", 0) < 1024 * 1024
               and (entry["path"].endswith((".py", ".toml"))
                    or Path(entry["path"]).name in {"AGENTS.md", "README.md", "LICENSE", "requirements.txt"})]
    print(f"{name}: fixed {commit}, {len(entries)} small source files", flush=True)
    for entry in entries:
        relative = entry["path"]
        target = (destination / relative).resolve()
        if not target.is_relative_to(destination.resolve()):
            raise ValueError("Unsafe source path")
        if target.exists():
            content = target.read_bytes()
        else:
            content = fetch(f"https://raw.githubusercontent.com/{repo}/{commit}/{relative}")
            if len(content) != entry["size"]:
                raise ValueError("Source size differs from fixed Git tree")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        # A Git blob hash checks both the content and the pin, including resumed files.
        blob_hash = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if blob_hash != entry["sha"]:
            raise ValueError(f"Fixed Git blob hash differs: {relative}")
        pin["files"][relative] = {"sha256": hashlib.sha256(content).hexdigest(), "git_blob": blob_hash}
        pin_file.write_text(json.dumps(pin, indent=2), encoding="utf-8")
    pin["status"] = "complete-selected-source"
    pin_file.write_text(json.dumps(pin, indent=2), encoding="utf-8")
    report[name] = {"repo": repo, "commit": commit, "path": str(destination),
                    "scope": "Selected Python, TOML, README, AGENTS, LICENSE and requirements files; data/assets excluded",
                    "file_manifest": str(pin_file)}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Completed {name}", flush=True)
