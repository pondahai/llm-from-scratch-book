"""
================================================================================
  全專案共用的語料與分詞器建置工具 (Single Source of Truth)

  【為什麼需要這支檔案】
  本專案曾經發生過一個難以察覺的錯誤：不同腳本各自寫 f.read()[:30000] /
  [:50000] / [:20000]，導致算出來的詞表大小不一致 (2144 / 2505 / 1887)。
  訓練好的權重與伺服器的分詞器對不上，模型會「正常回應但全是亂碼」。

  更隱蔽的是：combined 檔是四本書「依序接起來」的，所以 f.read()[:50000]
  取到的其實 100% 是《三國演義》開頭，水滸/紅樓/西遊一個字都沒讀到。

  解法：詞表一律由【完整語料】建立 (V = 6178)，訓練資料則另外指定。
  兩者分開，才能做「同一個詞表、只改資料量」的對照實驗。
================================================================================
"""

import os
import json
from code_ch1_tokenizer import SimpleBPEStyleTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

COMBINED = os.path.join(DATA_DIR, "four_great_novels_combined.txt")
VOCAB_JSON = os.path.join(DATA_DIR, "vocab.json")
NOVELS = {
    "sanguo": os.path.join(DATA_DIR, "sanguo_yanyi.txt"),
    "shuihu": os.path.join(DATA_DIR, "shuihu_zhuan.txt"),
    "honglou": os.path.join(DATA_DIR, "honglou_meng.txt"),
    "xiyou": os.path.join(DATA_DIR, "xiyou_ji.txt"),
}


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_full_corpus() -> str:
    """完整四大名著合集 (約 291.5 萬字)。詞表一律由此建立。"""
    if os.path.exists(COMBINED):
        return _read(COMBINED)
    return "".join(_read(p) for p in NOVELS.values() if os.path.exists(p))


def load_sanguo_sample(n_chars: int = 50000) -> str:
    """A 組對照用：《三國演義》前 n 個字 (預設 5 萬字)。"""
    path = NOVELS["sanguo"] if os.path.exists(NOVELS["sanguo"]) else COMBINED
    return _read(path)[:n_chars]


def _apply_vocab(tok: SimpleBPEStyleTokenizer, vocab: list) -> SimpleBPEStyleTokenizer:
    """把存檔的詞表換進一個 Tokenizer 實例 (不修改第 1 章的教學類別本身)。"""
    tok.vocab = vocab
    tok.token2id = {t: i for i, t in enumerate(vocab)}
    tok.id2token = {i: t for i, t in enumerate(vocab)}
    tok.pad_id = tok.token2id["<PAD>"]
    tok.unk_id = tok.token2id["<UNK>"]
    tok.bos_id = tok.token2id["<BOS>"]
    tok.eos_id = tok.token2id["<EOS>"]
    tok.vocab_size = len(vocab)
    return tok


def save_vocab(tok: SimpleBPEStyleTokenizer, path: str = VOCAB_JSON) -> str:
    """把詞表另存為 vocab.json，讓檢查點不再依賴語料檔逐位元組一致。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"vocab_size": tok.vocab_size, "vocab": tok.vocab},
                  f, ensure_ascii=False)
    return path


def build_tokenizer(prefer_vocab_file: bool = True) -> SimpleBPEStyleTokenizer:
    """
    全專案唯一的分詞器建置入口，確保 V 永遠一致 (6178)。
    絕對不要在別處用語料切片自建分詞器，那正是先前 bug 的根源。

    優先順序：
      1. 若 data/vocab.json 存在 -> 直接載入 (最可靠，與語料檔無關)
      2. 否則從完整語料建立，並自動存出 vocab.json 供下次與其他人使用

    為什麼要有 vocab.json？因為詞表是從語料「算」出來的，語料檔只要有任何
    差異 (少一本書、換行符不同、編碼不同)，算出來的詞表就會不一樣，
    導致別人下載的檢查點載入失敗。把詞表釘死成一個檔案就沒有這個風險——
    這也是 Hugging Face 上每個模型旁邊都有 tokenizer.json 的原因。
    """
    if prefer_vocab_file and os.path.exists(VOCAB_JSON):
        with open(VOCAB_JSON, encoding="utf-8") as f:
            vocab = json.load(f)["vocab"]
        return _apply_vocab(SimpleBPEStyleTokenizer(""), vocab)

    tok = SimpleBPEStyleTokenizer(load_full_corpus())
    try:
        save_vocab(tok)
    except OSError:
        pass          # 唯讀環境下略過存檔，不影響使用
    return tok


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    full = load_full_corpus()
    tok = build_tokenizer()
    sample = load_sanguo_sample()
    print(f"完整語料 : {len(full):,} 字")
    print(f"統一詞表 : {tok.vocab_size} tokens")
    print(f"A 組語料 : {len(sample):,} 字 (三國演義前段)")
    for name in ["曹操", "武松", "寶玉", "悟空"]:
        print(f"  {name} — 全語料 {full.count(name):>5} 次 | A 組 {sample.count(name):>4} 次")
