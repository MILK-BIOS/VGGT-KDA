import torch
from vggt.models.vggt import VGGT
from streamvggt.models.streamvggt import StreamVGGT

def convert_weight(state_dict: dict, qkv_bias: bool = False) -> dict:
    """
    Convert global blocks:
      aggregator.global_blocks.X.attn.qkv.weight/bias
      aggregator.global_blocks.X.attn.proj.weight/bias
    into:
      aggregator.global_blocks.X.attn.q_proj.*, k_proj.*, v_proj.*
      aggregator.global_blocks.X.proj.weight/bias
    """
    out = dict(state_dict)
    keys = list(out.keys())

    def _is_global_qkv(key: str) -> bool:
        return ("aggregator.global_blocks." in key) and ("attn." in key)

    for k in keys:
        if k.endswith("attn.qkv.weight") and _is_global_qkv(k):
            w = out.pop(k)
            prefix = k[: -len("qkv.weight")]  # keep "...attn."

            if w.ndim != 2:
                raise ValueError(f"Unexpected qkv.weight ndim={w.ndim} for key={k}, shape={tuple(w.shape)}")

            # Most common: [3*D, D] -> split on dim=0
            if w.shape[0] % 3 == 0:
                q_w, k_w, v_w = w.chunk(3, dim=0)
            else:
                raise ValueError(f"Cannot split qkv.weight for key={k}, shape={tuple(w.shape)}")

            out[prefix + "q_proj.weight"] = q_w
            out[prefix + "k_proj.weight"] = k_w
            out[prefix + "v_proj.weight"] = v_w

        elif k.endswith("attn.qkv.bias") and qkv_bias and _is_global_qkv(k):
            b = out.pop(k)
            prefix = k[: -len("qkv.bias")]

            if b.ndim != 1 or (b.shape[0] % 3 != 0):
                raise ValueError(f"Cannot split qkv.bias for key={k}, shape={tuple(b.shape)}")

            q_b, k_b, v_b = b.chunk(3, dim=0)
            out[prefix + "q_proj.bias"] = q_b
            out[prefix + "k_proj.bias"] = k_b
            out[prefix + "v_proj.bias"] = v_b
        
        elif k.endswith("attn.proj.weight") and _is_global_qkv(k):
            w = out.pop(k)
            prefix = k[: -len("proj.weight")]

            if w.ndim != 2:
                raise ValueError(f"Unexpected qkv.weight ndim={w.ndim} for key={k}, shape={tuple(w.shape)}")
            out[prefix + "o_proj.weight"] = w

        elif k.endswith("attn.proj.bias") and _is_global_qkv(k):
            b = out.pop(k)
            prefix = k[: -len("proj.bias")] 
            out[prefix + "o_proj.bias"] = b
    return out

if __name__ == "__main__":
    ckpt_teacher = torch.load('/commondocument/fhs/VideoVGGT/ckpt/model.pt', map_location='cuda')
    ckpt = torch.load('/commondocument/fhs/StreamVGGT/ckpt/streamvggt.pth', map_location='cuda')
    # 打印 checkpoint 的键（外层）
    if isinstance(ckpt_teacher, dict):
        print("[teacher ckpt] top-level keys:", list(ckpt_teacher.keys()))
    else:
        print("[teacher ckpt] type:", type(ckpt_teacher))

    teacher_sd = ckpt
    teacher_sd = convert_weight(teacher_sd)

    model = StreamVGGT()
    model.load_state_dict(teacher_sd, strict=True)
    model = model.to("cuda")
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
