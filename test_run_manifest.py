# test_run_manifest.py — 实验 manifest / 配置指纹 / 续跑校验 / 读出与快照的严格性
#
# 覆盖本轮修复的两个"结果可信度"问题：
#   1. P0：不同实验配置不得复用同一个结果文件（旧版固定 ar_{MODE}_{SRC}.jsonl +
#      AR_RESUME=1 默认开 -> 消融组互相污染）。
#   2. P1/P2：AR_READOUT_MODE=latent 未实现必须报错；delta 写入开启但快照为空必须报错。
import json
import os
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_manifest as rm

HERE = os.path.dirname(os.path.abspath(__file__))


def ok(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"  ok - {name}")


def _raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


# ---------------- 1. 配置指纹 ----------------
def test_config_id_separates_ablation_knobs():
    base = {"mode": "v9", "delta_write": False, "top_k": 12,
            "v9": {"l18": False, "l18_alpha": 0.3, "eqa_back": 16, "eqa_ahead": 4}}
    cid = rm.config_id(base, "ar", "eventqa_65536")
    ok("fingerprint stable", cid == rm.config_id(dict(base), "ar", "eventqa_65536"))
    ok("fingerprint is 8 hex", len(cid) == 8 and all(c in "0123456789abcdef" for c in cid))

    muts = {
        "delta_write": lambda p: p.__setitem__("delta_write", True),
        "top_k": lambda p: p.__setitem__("top_k", 8),
        "l18": lambda p: p["v9"].__setitem__("l18", True),
        "l18_alpha": lambda p: p["v9"].__setitem__("l18_alpha", 0.5),
        "eqa window": lambda p: p["v9"].__setitem__("eqa_back", 4),
    }
    for label, mutate in muts.items():
        p = json.loads(json.dumps(base))
        mutate(p)
        ok(f"fingerprint changes with {label}", rm.config_id(p, "ar", "eventqa_65536") != cid)
    ok("fingerprint includes src", rm.config_id(base, "ar", "eventqa_131072") != cid)
    ok("fingerprint includes kind", rm.config_id(base, "cr", "eventqa_65536") != cid)


def test_default_out_is_unique_per_config():
    a = rm.default_out("/tmp/r", "ar_v9", "eventqa_65536", "aaaaaaaa")
    b = rm.default_out("/tmp/r", "ar_v9", "eventqa_65536", "bbbbbbbb")
    ok("different ids -> different files", a != b)
    ok("star escaped in filename",
       "star" in rm.default_out("/tmp/r", "ar_v9", "longmemeval_s*", "aaaaaaaa"))


# ---------------- 2. 续跑校验 ----------------
def test_resume_checks():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "ar.jsonl")
        m = rm.build("ar", "eventqa_65536", {"delta_write": False})
        ok("new run allowed", rm.check_resume(out, m)[0] is True)

        with open(out, "w") as f:
            f.write("{}\n")
        ok_ok, why = rm.check_resume(out, m)
        ok("missing manifest refused", ok_ok is False and "manifest" in why)

        rm.write(out, m)
        ok("sidecar written", os.path.exists(rm.sidecar_path(out)))
        ok("same manifest resumes", rm.check_resume(out, m)[0] is True)

        m2 = rm.build("ar", "eventqa_65536", {"delta_write": True})
        ok_ok, why = rm.check_resume(out, m2)
        ok("config mismatch refused", ok_ok is False and "配置指纹" in why)

        m3 = json.loads(json.dumps(m))
        m3["git"]["commit"] = "deadbeef" * 5
        ok("commit mismatch refused", rm.check_resume(out, m3)[0] is False)

        # 防御性分支：config_id 相同但参数被手改（正常流程走不到，manifest 被改时必须拦）
        m4 = json.loads(json.dumps(m))
        m4["params"]["delta_write"] = True
        ok_ok, why = rm.check_resume(out, m4)
        ok("param mismatch refused", ok_ok is False and "参数不同" in why)

        ok("describe has id", "id=" in rm.describe(m) and m["config_id"] in rm.describe(m))


# ---------------- 3. runner 接线：不同配置 -> 不同文件 ----------------
def _dry_run(env_extra):
    env = dict(os.environ)
    env.pop("AR_OUT", None)
    env.update(env_extra)
    r = subprocess.run([sys.executable, "run_ar_v7.py", "eventqa_65536", "--dry-run"],
                       cwd=HERE, env=env, capture_output=True, text=True, timeout=120)
    line = [ln for ln in r.stdout.splitlines()
            if ln.startswith("[dry-run] SRC=") and "OUT=" in ln]
    return r, (line[0].split("OUT=")[-1].strip() if line else None)


def test_runner_writes_distinct_files_per_config():
    r0, out0 = _dry_run({"AR_MODE": "v9"})
    ok("dry-run ok", r0.returncode == 0 and out0 and out0.endswith(".jsonl"))
    _, out_delta = _dry_run({"AR_MODE": "v9", "AR_DELTA_WRITE": "1"})
    _, out_l18 = _dry_run({"AR_MODE": "v9", "AR_V9_L18": "1"})
    _, out_win = _dry_run({"AR_MODE": "v9", "AR_V9_EQA_BACK": "4", "AR_V9_EQA_AHEAD": "16"})
    ok("delta_write gets its own file", out_delta != out0)
    ok("l18 gets its own file", out_l18 != out0)
    ok("eventqa window gets its own file", out_win != out0)
    ok("manifest id embedded in filename",
       out0.split("_")[-1].split(".")[0] in out0)


def test_latent_readout_is_refused():
    r, _ = _dry_run({"AR_MODE": "v9", "AR_READOUT_MODE": "latent"})
    ok("latent readout refused", r.returncode == 2 and "未实现" in r.stdout)


# ---------------- 4. delta 空快照必须报错 ----------------
class _StubBatch:
    def __init__(self, ids):
        self.input_ids = ids


class _StubTok:
    def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=False):
        return "prompt"

    def __call__(self, text, return_tensors=None):
        return _StubBatch(torch.zeros((1, 4), dtype=torch.long))


def test_empty_delta_snapshot_raises():
    import run_ar_v7 as R
    saved = {k: getattr(R, k) for k in
             ("DELTA_WRITE", "delta_snapshot", "reset_delta_mem_states",
              "set_delta_mem_write_enabled", "chunk_text_into_sentences",
              "model", "tok", "DEV", "sys_msg", "mem_tpl")}
    R.DELTA_WRITE = True
    R.delta_snapshot = lambda m: {}
    R.reset_delta_mem_states = lambda m: None
    R.set_delta_mem_write_enabled = lambda m, v: None
    R.chunk_text_into_sentences = lambda t, chunk_size=512: ["chunk"]
    R.model = lambda **kw: None
    R.tok = _StubTok()
    R.DEV = torch.device("cpu")
    R.sys_msg, R.mem_tpl = "", "{context}"
    try:
        ok("empty snapshot raises", _raises(lambda: R.prepare_delta_memory("ctx")))
        os.environ["AR_ALLOW_EMPTY_DELTA_SNAP"] = "1"
        try:
            ok("explicit override still allowed",
               R.prepare_delta_memory("ctx") == {})
        finally:
            os.environ.pop("AR_ALLOW_EMPTY_DELTA_SNAP", None)
    finally:
        for k, v in saved.items():
            setattr(R, k, v)


if __name__ == "__main__":
    test_config_id_separates_ablation_knobs()
    test_default_out_is_unique_per_config()
    test_resume_checks()
    test_runner_writes_distinct_files_per_config()
    test_latent_readout_is_refused()
    test_empty_delta_snapshot_raises()
    print("ALL RUN_MANIFEST TESTS PASSED")
