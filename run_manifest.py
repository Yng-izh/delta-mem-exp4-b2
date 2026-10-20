"""run_manifest.py — 实验 manifest / 配置指纹 / 断点续跑校验（纯标准库，无 torch 依赖）。

解决的问题（P0：不同实验配置复用同一个结果文件）
------------------------------------------------
旧版 `run_ar_v7.py` 输出固定为 `ar_{MODE}_{SRC}.jsonl`，而 `AR_RESUME=1` 默认开启。
于是 `AR_DELTA_WRITE=0/1`、`AR_V9_L18=0/1`、不同 EventQA 窗口、不同 `AR_TOP_K`……
全部写进同一个文件；第二次运行会跳过"已完成"的题目，直接打印第一次实验的分数，
本该相互对照的两组实验变成同一组结果。

这里的做法：
  1. 把**影响结果的全部配置**规范化后算一个 8 位 `config_id`
     （不含时间与 commit，保证同配置同名、可续跑）；
  2. 输出文件名带上 `config_id`，不同配置天然是不同文件；
  3. 同目录写 `<out>.manifest.json`，记录 config_id / 完整参数 / git commit / 运行环境；
  4. 续跑前校验 manifest：config_id、参数、commit 任一不一致就**拒绝恢复**（返回原因）。

用法：
    m = run_manifest.build(kind="ar", src=SRC, params=params)
    out = run_manifest.default_out("/share/exp4group/results", "ar_v9", SRC, m["config_id"])
    run_manifest.write(out, m)
    ok, why = run_manifest.check_resume(out, m)      # ok=False -> 必须拒绝续跑
"""

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time

SCHEMA = 1


# ----------------------------- 规范化 / 指纹 -----------------------------
def _jsonable(v):
    """递归转成可 JSON 序列化的规范形式（dict 键排序）。"""
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in sorted(v.items())}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    return str(v)


def _flat(obj, prefix=""):
    """把嵌套参数拍平成 "a.b.c" -> 值，便于逐项比对。"""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flat(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        out[prefix] = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    else:
        out[prefix] = obj
    return out


def config_id(params, kind="", src=""):
    """配置指纹：同 kind+src+params -> 同 id；8 位十六进制。"""
    blob = json.dumps({"kind": kind, "src": src, "params": _jsonable(params)},
                      sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]


# ----------------------------- 环境信息 -----------------------------
def git_info(cwd=None):
    """当前代码版本：commit + 是否有未提交改动（脏树会让 commit 不足以定位代码）。"""
    cwd = cwd or os.path.dirname(os.path.abspath(__file__))

    def _run(*args):
        try:
            return subprocess.check_output(args, cwd=cwd, stderr=subprocess.DEVNULL,
                                           timeout=5).decode().strip()
        except Exception:
            return None

    commit = _run("git", "rev-parse", "HEAD") or "unknown"
    status = _run("git", "status", "--porcelain")
    return {"commit": commit, "dirty": bool(status),
            "dirty_files": len(status.splitlines()) if status else 0}


def env_info():
    env = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda"] = getattr(torch.version, "cuda", None)
    except Exception:
        env["torch"] = None
        env["cuda"] = None
    return env


# ----------------------------- 构建 / 读写 -----------------------------
def build(kind, src, params, extra=None, now=None, cwd=None):
    """构建 manifest。`params` 必须包含所有影响结果的配置（否则指纹会撞）。"""
    m = {
        "schema": SCHEMA,
        "config_id": config_id(params, kind, src),
        "kind": kind,
        "src": src,
        "params": _jsonable(params),
        "git": git_info(cwd),
        "env": env_info(),
        "host": socket.gethostname(),
        "created_at": now or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if extra:
        m["extra"] = _jsonable(extra)
    return m


def default_out(base_dir, prefix, src, cid):
    """带配置指纹的默认输出路径：不同配置不会落到同一个文件。"""
    safe = (src or "nosrc").replace("*", "star").replace("/", "_")
    return os.path.join(base_dir, f"{prefix}_{safe}_{cid}.jsonl")


def sidecar_path(out):
    return out + ".manifest.json"


def write(out, manifest):
    """原子写 sidecar（先写 .tmp 再 rename），返回 sidecar 路径。"""
    d = os.path.dirname(os.path.abspath(out))
    if d:
        os.makedirs(d, exist_ok=True)
    path = sidecar_path(out)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)
    return path


def load(out):
    path = sidecar_path(out)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def check_resume(out, manifest, strict_commit=True):
    """续跑前校验：返回 (ok, reason)。ok=False 时调用方必须拒绝恢复。"""
    if not os.path.exists(out):
        return True, "new run"
    old = load(out)
    if old is None:
        return False, (f"结果文件已存在但缺少 {os.path.basename(sidecar_path(out))}"
                       f"（旧版结果或 manifest 被删），无法确认配置一致")
    if old.get("schema") != manifest.get("schema"):
        return False, f"manifest schema 不同：{old.get('schema')} != {manifest.get('schema')}"
    if old.get("config_id") != manifest.get("config_id"):
        return False, (f"配置指纹不同：已有 {old.get('config_id')} != 当前 "
                       f"{manifest.get('config_id')}")
    a = _flat(old.get("params") or {})
    b = _flat(manifest.get("params") or {})
    diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    if diff:
        shown = ", ".join(f"{k}: {a.get(k)!r}->{b.get(k)!r}" for k in diff[:5])
        return False, f"参数不同（{len(diff)} 项）：{shown}"
    old_commit = (old.get("git") or {}).get("commit")
    new_commit = (manifest.get("git") or {}).get("commit")
    if strict_commit and old_commit and new_commit and old_commit != new_commit:
        return False, (f"代码 commit 不同：已有 {old_commit[:8]} != 当前 {new_commit[:8]}"
                       f"（代码变了，旧结果不可续跑）")
    note = ""
    if (manifest.get("git") or {}).get("dirty"):
        note = "；当前工作树有未提交改动，commit 不足以定位代码，请自行确认"
    return True, f"resume {old.get('config_id')}{note}"


def describe(manifest):
    """日志用的一行摘要。"""
    g = manifest.get("git") or {}
    commit = (g.get("commit") or "unknown")[:8]
    return (f"id={manifest.get('config_id')} kind={manifest.get('kind')} "
            f"src={manifest.get('src')} commit={commit}"
            f"{'+dirty' if g.get('dirty') else ''}")
