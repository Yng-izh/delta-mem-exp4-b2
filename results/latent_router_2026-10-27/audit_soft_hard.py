# 注意：本脚本硬编码了服务器路径（/share/exp4group、/share/hf_home），需在 v100-2 上运行。
"""核查：软门控 vs 硬选的一致性、槽边界丢失、q_feat 区分度。"""
import sys, torch
sys.path.insert(0, "/share/exp4group")
torch.manual_seed(0)
import latent_memory as LM

K, m, d, d_key = 4, 4, 2560, 256
router = LM.LatentRouter(d, n_slots=m, d_key=d_key).eval()
slot_z = torch.randn(1, K, m, d) * 1.0
q = torch.randn(1, 2 * d)

zs, logits, _, _ = router(q, slot_z, hard=False)
zh, logits_h, _, _ = router(q, slot_z, hard=True)
idx = int(logits.argmax())
print(f"[A] 同一 logits: soft_argmax={idx} hard={int(logits_h.argmax())}")
print(f"[A] 范数: 每个槽 ||z_k||={[round(float(slot_z[0,i].norm()),2) for i in range(K)]}")
print(f"[A] 范数: ||z_soft||={float(zs[0].norm()):.2f}  ||z_hard||={float(zh[0].norm()):.2f}"
      f"  -> 软聚合把范数压到 {float(zs[0].norm())/float(zh[0].norm()):.1%}")
print(f"[A] 距离: ||z_soft - z_hard||={float((zs-zh).norm()):.2f}"
      f"  (相对 {float((zs-zh).norm())/float(zh.norm()):.1%} of ||z_hard||)")
w = torch.softmax(logits, dim=-1)
print(f"[A] 权重 w={[round(float(x),3) for x in w[0]]}  max_w={float(w.max()):.3f}")

# 初始化时 logits 量级 -> 早期训练到底有多"均匀"
tau = router.log_temp.exp()
print(f"[A] tau={float(tau):.4f}  logits={[round(float(x),3) for x in logits[0]]}")

# B: 槽边界。若某 unit 被切 2 块，cat 后 K' = 5，unsqueeze 会把它当 5 个槽
p1 = torch.randn(2, m, d)   # 槽0 切成 2 块
p2 = torch.randn(1, m, d)
p3 = torch.randn(1, m, d)
p4 = torch.randn(1, m, d)
cat = torch.cat([p1, p2, p3, p4], dim=0)
stacked = LM.stack_payloads([p1, p2, p3, p4])
print(f"[B] 4 槽,槽0 切 2 块 -> cat={tuple(cat.shape)} (被当成 {cat.shape[0]} 个槽!)"
      f" vs stack_payloads={tuple(stacked.shape)} (正确的 {stacked.shape[0]} 个槽)")
