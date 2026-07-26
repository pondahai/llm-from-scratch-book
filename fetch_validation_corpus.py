"""
================================================================================
  驗證集語料抓取：《儒林外史》(供書稿 §9.8 / §9.9 / §9.10)

  【為什麼需要這個檔案】
  本書原本的所有 Loss 都是在【訓練資料】上量的。訓練 Loss 低，可能是模型
  學會了，也可能只是把訓練資料背起來了——§9.8 的 A 組就是活生生的例子。
  要分辨這兩者，必須有一份【模型從未見過】的文本。

  四大名著全部餵進了訓練，任何切片都被污染了，所以驗證集只能來自外部。
  選《儒林外史》的理由：
    - 吳敬梓，約 1750 年，與《紅樓夢》幾乎同期
    - 白話章回小說，文體與訓練語料最接近（比文言的《聊齋》合適）
    - 維基文庫全文，公有領域，與現有語料的授權一致

  ⚠️ 詞表 V=6,178 是從四大名著算出來的，不會因為這份語料而改變。
     《儒林外史》裡沒出現過的字會變成 <UNK>，評估時必須跳過那些位置，
     並如實揭露未知字比例 (見 eval_validation.py)。

  用法：
      python fetch_validation_corpus.py
================================================================================
"""

import os
import re
import sys
import time

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT = os.path.join(DATA_DIR, "rulin_waishi.txt")

BOOK, TOTAL = "儒林外史", 56
CN = "零一二三四五六七八九十"

# 維基文庫的章節標題命名不統一，多準備幾種樣式輪流試
PATTERNS = [
    "儒林外史/第{ch3}回",
    "儒林外史/第{ch2}回",
    "儒林外史/第{ch}回",
    "儒林外史/第{cn}回",
    "儒林外史第{ch}回",
]
HEADERS = {"User-Agent": "LLMBookDatasetFetcher/2.0 "
                        "(Educational/Research Purposes; Contact: reader@example.org)"}
API = "https://zh.wikisource.org/w/api.php"
session = requests.Session()


def cn_num(n: int) -> str:
    """1 -> 一, 11 -> 十一, 20 -> 二十, 56 -> 五十六"""
    if n <= 10:
        return CN[n]
    tens, ones = divmod(n, 10)
    return ("十" if tens == 1 else CN[tens] + "十") + (CN[ones] if ones else "")


def fetch(title: str, max_retries: int = 3):
    params = {"action": "query", "prop": "extracts", "titles": title,
              "explaintext": 1, "format": "json"}
    for attempt in range(max_retries):
        try:
            r = session.get(API, params=params, headers=HEADERS, timeout=15)
            if r.status_code == 429:                      # 禮貌退避
                time.sleep(2.0 * (attempt + 1))
                continue
            if r.status_code != 200:
                return None
            for pid, page in r.json().get("query", {}).get("pages", {}).items():
                if pid != "-1" and "extract" in page:
                    text = page["extract"].strip()
                    if len(text) > 200:
                        return re.sub(r"\n+", "\n", text)
            return None
        except Exception:
            time.sleep(1.0)
    return None


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    chapters, misses = {}, []
    hit_pattern = None

    for ch in range(1, TOTAL + 1):
        text = None
        # 第一回成功之後就固定用那個樣式，不再每回都試五種
        pats = [hit_pattern] if hit_pattern else PATTERNS
        for pat in pats:
            title = pat.format(ch=ch, ch2=f"{ch:02d}", ch3=f"{ch:03d}", cn=cn_num(ch))
            text = fetch(title)
            if text:
                hit_pattern = pat
                break
        if text:
            chapters[ch] = text
            print(f"  ✅ 第 {ch:2d}/{TOTAL} 回 ({len(text):,} 字)", flush=True)
        else:
            misses.append(ch)
            print(f"  ❌ 第 {ch:2d}/{TOTAL} 回 未找到", flush=True)
        time.sleep(0.5)                                    # 禮貌延遲

    if not chapters:
        raise SystemExit("一回都沒抓到，請檢查網路或標題樣式")

    body = "".join(f"\n\n=== {BOOK} 第 {c} 回 ===\n\n{chapters[c]}"
                   for c in sorted(chapters))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(body)

    print(f"\n《{BOOK}》完成 {len(chapters)}/{TOTAL} 回，共 {len(body):,} 字")
    if misses:
        print(f"  未取得：{misses}")
    print(f"  已存至 {os.path.relpath(OUT, PROJECT_ROOT)}")
    print(f"  使用的標題樣式：{hit_pattern}")


if __name__ == "__main__":
    main()
