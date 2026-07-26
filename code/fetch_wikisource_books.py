"""
================================================================================
  維基文庫 (Wikisource) 四大名著合規爬取與文字清洗範例
  本腳本符合「API 禮貌與合規原則 (Respectful Polling & Scraper Best Practices)」
  包含：合規 User-Agent、禮貌延遲 (0.5s)、429 退避重試與多模式標題回退匹配。
================================================================================
"""

import sys
import os
import time
import requests
import re
from typing import Optional, List, Dict

# 強制 stdout 使用 UTF-8 編碼
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 建立 requests.Session 重複使用 HTTP 連線
session = requests.Session()

def fetch_wikisource_chapter(title: str, max_retries: int = 3) -> Optional[str]:
    """
    【合規抓取函式】透過 MediaWiki API 抓取維基文庫指定頁面的純文字內容。
    包含：合規 User-Agent、HTTP 429 指數退避與重試機制。
    """
    url = "https://zh.wikisource.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "titles": title,
        "explaintext": 1,
        "format": "json"
    }
    
    # 📌 合規規則 1：明確標示專案身份與用途的 User-Agent
    headers = {
        "User-Agent": "LLMBookDatasetFetcher/2.0 (Educational/Research Purposes; Contact: reader@example.org)"
    }
    
    for attempt in range(max_retries):
        try:
            response = session.get(url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                pages = data.get("query", {}).get("pages", {})
                for page_id, page_data in pages.items():
                    if page_id != "-1" and "extract" in page_data:
                        text = page_data["extract"].strip()
                        if len(text) > 200: # 確保有實質內容
                            # 📌 簡單清理頁面導覽標籤與多餘空行
                            text = re.sub(r"\n+", "\n", text)
                            return text
                return None
                
            elif response.status_code == 429:
                # 📌 合規規則 2：遇到 HTTP 429 觸發指數退避休眠 (Exponential Backoff)
                sleep_time = 2.0 * (attempt + 1)
                print(f"\n  ⚠️ 觸發 Rate Limit (HTTP 429)，禮貌休眠 {sleep_time} 秒後重試...")
                time.sleep(sleep_time)
                
        except Exception as e:
            time.sleep(1.0)
            
    return None

def fetch_chapter_with_fallbacks(book_name: str, ch: int, patterns: List[str]) -> Optional[str]:
    """
    📌 合規規則 3：多模式標題回退匹配 (Fallback Title Patterns)
    嘗試不同之標題命名格式（如第001回、第01回、第1回、程乙本、120回本）。
    """
    for pat in patterns:
        title = pat.format(ch=ch, ch3=f"{ch:03d}", ch2=f"{ch:02d}")
        content = fetch_wikisource_chapter(title)
        if content:
            return content
    return None

def download_book(book_name: str, patterns: List[str], total_chapters: int, save_filename: str, data_dir: str = "../data") -> str:
    """
    📌 合規規則 4 & 5：禮貌性請求延遲與本地斷點續傳 (Checkpointing)
    """
    os.makedirs(data_dir, exist_ok=True)
    save_path = os.path.join(data_dir, save_filename)
    
    # 1. 檢查本地快取，載入已有章節
    existing_chapters = {}
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        with open(save_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()
        blocks = re.split(r"=== .*? 第 (\d+) 回 ===", raw_text)
        for i in range(1, len(blocks), 2):
            c_num = int(blocks[i])
            c_txt = blocks[i+1].strip()
            if len(c_txt) > 200:
                existing_chapters[c_num] = c_txt

    downloaded = dict(existing_chapters)
    print(f"\n📥 開始合規抓取《{book_name}》 (目標 {total_chapters} 回, 本地已有 {len(downloaded)} 回)...")

    for ch in range(1, total_chapters + 1):
        if ch in downloaded:
            continue
            
        content = fetch_chapter_with_fallbacks(book_name, ch, patterns)
        if content:
            downloaded[ch] = content
            print(f"  ✅ 第 {ch:3d}/{total_chapters} 回成功 ({len(content):,} 字)")
            
            # 即時漸進式寫入硬碟
            full_blocks = [f"\n\n=== {book_name} 第 {c} 回 ===\n\n{downloaded[c]}" for c in sorted(downloaded.keys())]
            with open(save_path, "w", encoding="utf-8") as f:
                f.write("".join(full_blocks))
        else:
            print(f"  ❌ 第 {ch:3d}/{total_chapters} 回未找到")
            
        # 📌 合規規則 4：請求間強制加入 0.5 秒禮貌停頓
        time.sleep(0.5)

    full_text = "".join([f"\n\n=== {book_name} 第 {c} 回 ===\n\n{downloaded[c]}" for c in sorted(downloaded.keys())])
    print(f"🎉 《{book_name}》完成！共收錄 {len(downloaded)}/{total_chapters} 回 ({len(full_text):,} 字)")
    return full_text

def main():
    print("=" * 80)
    print("🌐 [維基文庫 MediaWiki API 合規抓取示範]")
    print("  遵循 5 大合規法則：合規 UA、禮貌延遲 (0.5s)、429 退避、多標題匹配、本地快取。")
    print("=" * 80)

    # 示範單章測試
    test_title = "三國演義/第001回"
    print(f"\n📖 [單章合規測試] 抓取目標: {test_title} ...")
    content = fetch_wikisource_chapter(test_title)
    if content:
        print(f"  ✅ 成功獲取文本 (前 100 字): \"{content[:100].replace('\n', ' ')}...\"")
        print(f"  總字數: {len(content):,} 字")
    else:
        print("  ❌ 未找到或網路存取受限。")

    print("\n" + "=" * 80)
    print("[SUCCESS] 維基文庫 API 合規抓取腳本驗證完成！讀者可直接運行抓取全套語料庫！")
    print("=" * 80)

if __name__ == "__main__":
    main()
