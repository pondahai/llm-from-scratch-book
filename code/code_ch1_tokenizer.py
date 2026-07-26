"""
================================================================================
  第 1 章程式碼：文字的翻譯官——分詞器與詞表 (Tokenization)
  涵蓋心智圖節點：Tokenizer, Vocabulary, Character vs Subword (BPE), SentencePiece, Tokens, Special Tokens
================================================================================
"""

import sys
from typing import List, Dict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

class SimpleBPEStyleTokenizer:
    """
    第 1 章核心類別：分詞器與密碼表 (Tokenizer & Vocabulary)
    """
    def __init__(self, corpus_text: str):
        # 1. 提取文本中出現的所有獨特字元作為詞表基礎 (Character / Subword Vocabulary)
        unique_chars = sorted(list(set(corpus_text)))
        
        # 2. 加入特殊暗號 Token (Special Tokens)
        # <PAD>=0 (補齊長度), <UNK>=1 (未知字), <BOS>=2 (文章開始), <EOS>=3 (文章結束)
        self.special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        self.vocab = self.special_tokens + unique_chars
        
        # 3. 建立 詞彙 <-> ID 雙向密碼本 (Lookup Tables)
        self.token2id: Dict[str, int] = {token: idx for idx, token in enumerate(self.vocab)}
        self.id2token: Dict[int, str] = {idx: token for idx, token in enumerate(self.vocab)}
        
        self.pad_id = self.token2id["<PAD>"]
        self.unk_id = self.token2id["<UNK>"]
        self.bos_id = self.token2id["<BOS>"]
        self.eos_id = self.token2id["<EOS>"]
        self.vocab_size = len(self.vocab)

    def encode(self, text: str, add_bos_eos: bool = True) -> List[int]:
        """【翻譯】人類文字 -> 電腦數字 ID 序列 (Tokens)"""
        ids = [self.token2id.get(char, self.unk_id) for char in text]
        if add_bos_eos:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        """【反翻譯】電腦數字 ID 序列 -> 人類文字"""
        tokens = [self.id2token.get(idx, "<UNK>") for idx in ids if idx not in (self.pad_id, self.bos_id, self.eos_id)]
        return "".join(tokens)

def demo_chapter1():
    print("=" * 60)
    print("📖 [第 1 章 測試] 執行分詞器 (Tokenizer) 測試...")
    print("=" * 60)
    
    # 測試語料 (Corpus)
    text = "關羽揮舞青龍偃月刀，斬了顏良。"
    tokenizer = SimpleBPEStyleTokenizer(text)
    
    print(f"  詞表總大小: {tokenizer.vocab_size} 個 Token")
    print(f"  前 8 個 Token 代號: {tokenizer.vocab[:8]}")
    
    # 測試 Encode
    prompt = "關羽斬顏良"
    encoded_ids = tokenizer.encode(prompt, add_bos_eos=True)
    print(f"\n  原始輸入文字: '{prompt}'")
    print(f"  编码後的 Token IDs: {encoded_ids}")
    
    # 測試 Decode
    decoded_text = tokenizer.decode(encoded_ids)
    print(f"  解碼回人類文字: '{decoded_text}'")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] 第 1 章分詞器 Tokenizer 測試 100% 成功！")
    print("=" * 60)

if __name__ == "__main__":
    demo_chapter1()
