# 注意：本脚本硬编码了服务器路径（/share/exp4group、/share/hf_home），需在 v100-2 上运行。
import sys, torch
sys.path.insert(0, "/share/exp4group")
torch.manual_seed(0)
import latent_memory as LM
K, m, d = 4, 4, 2560
r = LM.LatentRouter(d, n_slots=m, d_key=256)
slot_z = torch.randn(1, K, m, d)
q = torch.randn(1, 2*d)
r.train()
z_tr, lg, ki, ko = r(q, slot_z, hard=False)
r.eval()
z_ev, _, _, _ = r(q, slot_z, hard=False)
z_hd, _, _, _ = r(q, slot_z, hard=True)
idx = int(lg.argmax())
print(f"[STE] picked={idx}  ||z_train||={float(z_tr.norm()):.3f} "
      f"||z_eval||={float(z_ev.norm()):.3f} ||z_hard||={float(z_hd.norm()):.3f}")
print(f"[STE] train vs hard 逐位相同? {torch.equal(z_tr, z_hd)}")
print(f"[STE] eval  vs hard 逐位相同? {torch.equal(z_ev, z_hd)}")
print(f"[STE] 命中的是第 {idx} 个槽? {torch.equal(z_hd[0], slot_z[0, idx])}")
# 梯度：router 必须拿到梯度，且 z_hard 部分能把梯度送回 slot_z
r.train()
slot_z2 = slot_z.clone().requires_grad_(True)
z2, lg2, _, _ = r(q, slot_z2, hard=False)
z2.sum().backward()
g_router = sum(float(p.grad.abs().sum()) for p in r.parameters() if p.grad is not None)
print(f"[STE] router 梯度总量={g_router:.3f} (应>0)  slot_z 梯度={float(slot_z2.grad.abs().sum()):.1f} (应>0)")
