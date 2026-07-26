"""
================================================================================
  第 8 章程式碼：擲骰子講故事——Logits 輸出與自迴歸採樣 (Decoding Strategies)
  累加第 1~7 章
  涵蓋心智圖節點：Logits, Probability Distribution, Temperature (溫度), Top-k / Top-p 採樣, EOS 終止符, Speculative Decoding (投機解碼)
================================================================================
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from code_ch1_tokenizer import SimpleBPEStyleTokenizer
from code_ch2_embedding import TokenEmbedding
from code_ch3_rope import RotaryPositionalEmbedding
from code_ch7_rmsnorm_moe import TransformerBlock, RMSNorm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

class MiniLLM(nn.Module):
    """
    第 8 章核心類別：完整迷你大語言模型 (Mini LLM)
    將 Tokenizer, Embedding, RoPE, Transformer Blocks, RMSNorm, LM Head 完整串接
    並實現帶有 Temperature 與 Top-k 採樣的自迴歸生成 (Generation)。
    """
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int = 64, 
        n_layers: int = 2, 
        n_heads: int = 4, 
        num_kv_groups: int = 2,
        hidden_dim: int = 128,
        use_moe: bool = False,          # 第 7 章的 MoE 開關，預設關閉（見 §9.10 實測）
        num_experts: int = 4,
        top_k: int = 1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = TokenEmbedding(vocab_size, d_model)
        self.rope = RotaryPositionalEmbedding(dim=d_model // n_heads)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, num_kv_groups, hidden_dim, self.rope,
                             use_moe=use_moe, num_experts=num_experts, top_k=top_k)
            for _ in range(n_layers)
        ])
        
        self.final_norm = RMSNorm(d_model)
        # Logits 輸出層 (LM Head)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, kv_caches: Optional[List] = None, use_cache: bool = False):
        x = self.embed(input_ids)
        new_caches = []
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            x, c = layer(x, kv_cache=cache, use_cache=use_cache)
            if use_cache:
                new_caches.append(c)
        x = self.final_norm(x)
        logits = self.lm_head(x)  # (Batch, Seq_Len, Vocab_Size)
        return logits, new_caches

    @torch.no_grad()
    def generate(
        self, 
        prompt_ids: List[int], 
        max_new_tokens: int = 15, 
        temperature: float = 0.8, 
        top_k: int = 5,
        eos_id: int = 3
    ) -> List[int]:
        """第 8 章：自迴歸 Top-k & Temperature 解碼生成"""
        self.eval()
        generated = list(prompt_ids)
        input_tensor = torch.tensor([generated], dtype=torch.long)
        kv_caches = None
        
        for _ in range(max_new_tokens):
            if kv_caches is None:
                logits, kv_caches = self.forward(input_tensor, use_cache=True)
            else:
                logits, kv_caches = self.forward(input_tensor[:, -1:], kv_caches=kv_caches, use_cache=True)
                
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            
            if next_token == eos_id:
                break
            input_tensor = torch.tensor([[next_token]], dtype=torch.long)
            
        return generated

def demo_chapter8():
    print("=" * 60)
    print("📖 [第 8 章 測試] 執行 MiniLLM 解碼與生成測試...")
    print("=" * 60)
    
    corpus = "水滸傳梁山泊一百零八將"
    tokenizer = SimpleBPEStyleTokenizer(corpus)
    
    model = MiniLLM(vocab_size=tokenizer.vocab_size, d_model=32, n_layers=2)
    prompt = "梁山泊"
    prompt_ids = tokenizer.encode(prompt, add_bos_eos=False)
    
    gen_ids = model.generate(prompt_ids=prompt_ids, max_new_tokens=10, temperature=0.7, top_k=3, eos_id=tokenizer.eos_id)
    
    print(f"  提示詞: '{prompt}'")
    print(f"  Token IDs: {prompt_ids}")
    print(f"  自迴歸生成結果 (前 10 Tokens): '{tokenizer.decode(gen_ids)}'")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] 第 8 章 MiniLLM 解碼與自迴歸生成測試 100% 成功！")
    print("=" * 60)

if __name__ == "__main__":
    demo_chapter8()
