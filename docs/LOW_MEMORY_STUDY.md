# 降低 VRAM + DRAM 需求：分析筆記與學習路線

這份筆記整理一個問題：**Strata 能不能用更少的 VRAM 和 DRAM 跑起來，SSD 能幫多少？** 它是之後寫成教學的底稿，
所以會解釋用到的概念，每一節最後也附上可以動手做的練習。

數字有兩種，標示方式如下：

- **實測**：來自 repo 裡的文件或 benchmark，會註明在什麼硬體上量的。
- **推算**：用公式或統計檔算出來的，沒有在真的機器上跑過。只能當方向參考，不能當結論。

分析時（2026-10-03）的目標機器：RTX 5080 16 GB、DRAM 31.8 GB、i7-14700F（沒有內顯，只有 AVX2）、兩顆 NVMe
（C: MSI M450 PCIe 4.0、D: KIOXIA EXCERIA G2 PCIe 3.0）。

---

## 1. 背景概念

### 1.1 MoE 和 expert

Qwen3.8-Flash-Next 是 **mixture-of-experts（MoE）** 模型。每一層除了共用的部分，還有 512 個 **expert**（一個小的
FFN），router 會為每個 token 從中挑 10 個來算。模型有 48 層，所以總共 48 × 512 = **24,576 個 experts**，但每個 token
只會用到 48 × 10 = **480 個**。

所以模型雖然很大，每個 token 實際碰到的權重只占一小部分。Strata 整個設計就是利用這一點。

| 模型 | experts 總大小 | 每個 expert | 每個 token 啟用 |
|---|---:|---:|---:|
| Q2_0 | 34.0 GB | ~1.38 MB | ~664 MB |
| Coder（IQ1_M，每層 256 個 experts） | 23.4 GB | ~1.90 MB | ~914 MB |

（大小取自 `setup.py` 的 `MODELS[...]["arena_gb"]`，每個 expert 的大小 = 總大小 ÷ expert 數。）

### 1.2 三層儲存：容量和頻寬

| 層 | 例子 | 頻寬 | 每 GB 價格 |
|---|---|---|---|
| VRAM | RTX 5080 16 GB | 數百 GB/s | 最貴 |
| DRAM | DDR5 雙通道 | 約 70–90 GB/s | 貴 |
| SSD | NVMe | 規格 2–7 GB/s；Strata 隨機讀實測約 1.5–2 GB/s（見 2.3） | 便宜 |
| HDD | — | 隨機讀約 0.1–0.2 GB/s | 最便宜 |

（頻寬是硬體規格的大概數字，NVMe 那一列除外。）

核心觀念：**容量便宜，頻寬貴。** SSD 裝得下整個模型，但每秒能送出的 bytes 比 DRAM 少了 30–50 倍。要降低 DRAM 需求，
真正的問題不是「放不放得下」，而是「每個 token 要從慢的那一層讀多少 bytes」。

### 1.3 Strata 怎麼分配（`docs/DETAILS.md` 的 How it works）

- **VRAM**：attention、router、共用 expert、KV cache，以及一個 **expert cache**，用剩下的 VRAM 放最常用的
  experts，並且會跟著對話調整（`--adapt-every`）。
- **DRAM**：正常模式下放**全部** experts（pinned）。GPU 沒命中的 experts 由 CPU 直接在 DRAM 裡算，不搬到 GPU。
- **SSD**：28.8 GB 的 n-gram/PLE 表，每個 token 透過 OS cache 讀幾列。

這也是為什麼正常模式要「RAM ≥ experts + 約 10 GB」：VRAM 再大，DRAM 還是要放全部 experts。

### 1.4 會用到的系統名詞

- **pinned / page-locked memory**：鎖在實體記憶體、不會被換出的 DRAM，GPU 可以直接用 DMA 讀。Strata 用它讓 GPU 分攤
  一部分 miss 的運算（`--pcie-frac`）。
- **mmap 與 OS file cache**：把檔案映射成記憶體，讀到哪一頁，OS 才從硬碟載入哪一頁，並留在 file cache。DRAM 不夠時，
  OS 可以把這些頁面丟掉，但之後要再讀一次硬碟。
- **page file（swap）**：**不能**拿來取代上面的機制。pinned memory 不會被換出；Windows 上 page file 太小時，GPU 配置
  記憶體反而會失敗（issue #60）。
- **speculative decoding / verify window**：先用小的草稿模型（MTP layer）猜幾個 token，再讓大模型一次驗證。一次
  驗證多個 token，讀進來的 experts 就能被多個 token 共用。

### 練習 1：讀 setup 的判斷公式

打開 `setup.py`，找 `low_ram_needed`、`low_ram_gpu_gb`、`low_ram_resident`、`low_ram_fits`（約第 2023–2058 行），
再用自己機器的數字跑一次：

```
python -c "import setup as s; [print(m, s.low_ram_needed(m,31.8), round(s.low_ram_gpu_gb(m,16),1), s.low_ram_resident(m,31.8,16), s.low_ram_fits(m,31.8,16)) for m in ('IQ1_M','Q2_0','IQ2_XS','IQ3_XXS')]"
```

在 31.8 GB / 16 GB 上的輸出（2026-10-03）：

```
IQ1_M True 11.0 True True
Q2_0 True 11.0 False True
IQ2_XS True 11.0 False True
IQ3_XXS True 11.0 False False
```

欄位依序是：需要 low-RAM 模式、GPU 約能放多少 GB experts、能不能用 resident 變體、mapped 模式放不放得下。

---

## 2. Strata 已經有的低記憶體機制

詳細說明在 `docs/DETAILS.md` 的 Low-RAM mode 段落。

| 機制 | engine 版本 | 做什麼 |
|---|---|---|
| `--mmap-experts` | 0.1.26 | experts 不複製進 DRAM，改用 mmap 從檔案讀，交給 OS file cache |
| `--resident-experts` | 0.1.30 | GPU 放不下的 experts 如果 DRAM 裝得下，就在啟動時一次複製進 DRAM 並鎖住，執行中不讀 SSD |
| 直接讀 GGUF | 0.1.31 | 不需要另外產生 `experts.bin`（省 23–50 GB 硬碟空間） |
| `--resident-budget-gib N` | 0.1.31 | 最熱的 N GiB 放 DRAM，其餘從 SSD 讀；lookahead 會用下一層的 router 預測要用的 experts 並提前讀入 |

### 2.1 實測數字

- Coder 用 `--mmap-experts`：engine 的 committed memory 從 36 GB 降到約 13 GB，輸出相同（`docs/DETAILS.md`）。
- 2.3 的 Unsloth UD-Q4_K_XL 是目前唯一實測過「大部分 experts 從 SSD 讀」的例子。

### 2.2 在 5080 16 GB / 31.8 GB 上（推算）

| 模型 | GPU 放不下的部分 | 模式 | 會不會讀 SSD |
|---|---:|---|---|
| Coder | ~12.4 GB | resident | 不會 |
| Q2_0 | ~23 GB | mmap 或 budget（離 resident 的條件差約 1–2 GB） | 少量（最冷的 experts） |
| IQ2_XS | ~24.5 GB | mmap 或 budget | 少量 |
| IQ3_XXS | ~31.9 GB | setup 判定放不下 | 大量 |

注意：14700F 沒有內顯，螢幕只能接 5080，桌面大約會占 0.5–1.5 GB VRAM。setup 的公式用的是「VRAM 總量 − 5 GB」，
沒有扣掉這部分，所以在這種機器上會偏樂觀。

可以參考的實測：有貢獻者在 RTX 5080 單卡（Ryzen 9 9950X3D，RAM 充足的一般模式）跑 Coder，decode 83–87 tok/s（故事）、
88–105 tok/s（程式碼），prompt 1,726–2,017 tok/s（`docs/MULTI_GPU.md`）。

### 2.3 SSD 的實際速度（實測）

Unsloth UD-Q4_K_XL（77 GB experts）在 RTX 5070 12 GB、64 GB RAM、Windows、NVMe 上（`docs/UNSLOTH_Q4.md`）：

| RAM budget | 輸出 | SSD 讀取量 |
|---|---|---|
| 24 GiB | 5.1–5.2 tok/s | 每輪 verify 2.2 GB（約 3.5 個 token） |
| 40 GiB | 7–8.5 tok/s | 每輪 verify 1.1 GB，約每個 token 0.33 GB |

在 40 GiB 時，每輪 verify 約 550–750 ms 花在讀 SSD，CPU 約 75 ms，GPU 約 13 ms，**瓶頸是 SSD**。換算有效頻寬約
1.5–2 GB/s，比 NVMe 規格上的循序讀取慢很多，因為 experts 是隨機的小區塊讀取。

### 練習 2：看懂 `--stats` 的 expert tiers

先讀 `docs/DETAILS.md` 的 "How much came from where"：`--stats` 會印出每一層各提供了多少 blobs、多少 MB；server 的
`GET /metrics` 有每個請求的 `ram_blobs`、`file_blobs`、`file_mb`。之後實測時，要看的就是這幾個數字。

---

## 3. 關鍵數據：路由分布有多集中

### 3.1 從哪裡來

`data/expert-profile.bin`（`tools/make_profile.py` 產生的）格式如下：

```
"STRP" | version, n_layers, n_expert, slots, n_ranked（5 × int32）
       | n_ranked 個 (layer, expert)（uint16 × 2），依頻率排序
       | n_layers × n_expert 個路由次數（uint32）
```

完整模型的檔案記錄了約 62.9 萬個 token 的路由次數；Coder 的 `data/expert-profile-coder.bin` 約 15.7 萬個。

### 3.2 結果（推算：靜態快取）

依頻率排序，把最熱的 X GB 放進「快層」（VRAM＋DRAM）時：

**Q2_0**（完整模型，512 experts／層）

| 快層容量 | 命中率 | 每個 token 從 SSD 讀 |
|---:|---:|---:|
| 4 GB | 22.2% | 517 MB |
| 8 GB | 41.5% | 388 MB |
| 10 GB | 50.2% | 331 MB |
| 12 GB | 58.1% | 278 MB |
| 16 GB | 72.0% | 186 MB |
| 20 GB | 83.0% | 113 MB |
| 24 GB | 91.4% | 57 MB |
| 28 GB | 96.9% | 21 MB |
| 30 GB | 98.6% | 9 MB |
| 32 GB | 99.7% | 2 MB |

**Coder**（256 experts／層）

| 快層容量 | 命中率 | 每個 token 從 SSD 讀 |
|---:|---:|---:|
| 8 GB | 56.7% | 396 MB |
| 12 GB | 76.3% | 217 MB |
| 16 GB | 90.0% | 91 MB |
| 20 GB | 97.9% | 19 MB |
| 22 GB | 99.6% | 3 MB |

### 3.3 怎麼解讀

1. **完整模型的路由分布很平。** 放進 16 GB（約一半的 experts）只有 72% 命中。快層從 30 GB 降到 16 GB，SSD 流量增加約
   20 倍。結論：**光靠快取策略，DRAM 降不到 16 GB 以下。**
2. **Q2_0 在 31.8 GB 的機器上用 budget 模式可能還不錯。** 快層約 10 GB VRAM 加 18–20 GB DRAM，約 28–30 GB，每個 token
   只需從 SSD 讀 9–21 MB。
3. **這是靜態快取的數字。** 實際的 adaptive cache 會跟著對話換 experts，單一對話的局部性應該更好，但好多少目前沒有
   數據。trace 用的是哪種工作負載，也還沒確認。
4. 假設每個 expert 一樣大。Q2_0 的 experts 大小幾乎一樣，這個假設大致成立。

### 練習 3：自己算這條曲線（不用 GPU、不用下載）

```python
import struct, array

for fn, gb in (("data/expert-profile.bin", 34.0), ("data/expert-profile-coder.bin", 23.4)):
    b = open(fn, "rb").read()
    v, L, E, S, R = struct.unpack_from("<5i", b, 4)
    a = array.array("I"); a.frombytes(b[24 + R*4 : 24 + R*4 + L*E*4])
    c = sorted(a, reverse=True); tot = sum(c); per = gb / (L*E)
    print(f"{fn}: {tot/(L*10):,.0f} tokens, {per*1000:.2f} MB/expert, {L*10*per*1000:.0f} MB/token")
    acc, marks, j = 0, list(range(2, 34, 2)), 0
    for i, x in enumerate(c):
        acc += x
        if j < len(marks) and (i+1)*per >= marks[j]:
            miss = 1 - acc/tot
            print(f"  top {marks[j]:>2} GB: hit {acc/tot:6.1%}  -> {miss*L*10*per*1000:4.0f} MB/token from SSD")
            j += 1
```

可以延伸的方向：改成「每層各自一個快取」（engine 的實際做法比較接近這種），看結果差多少；或畫成圖。

---

## 4. 第一原理：速度由什麼決定

```
每個 token 的時間 ≈ Σ（該層提供的 bytes ÷ 該層頻寬）
tok/s ≈ 1 ÷ 每個 token 的時間
```

例子（推算）：一台「VRAM 16 GB + DRAM 16 GB」的機器，快層約 16 GB，每個 token 從 SSD 讀 186 MB；SSD 有效頻寬
2 GB/s 時，光 SSD 就要約 93 ms，**上限約 10 tok/s**（正常模式約 80–90）。

所以要降低 DRAM 需求，能做的只有三件事：

1. **減少每個 token 要讀的 bytes**
2. **讓每次讀進來的 bytes 服務更多 token**（攤提）
3. **提高 SSD 的有效頻寬**

---

## 5. 更底層的機制（構想，都還沒實作）

基準：上一節 16/16 的例子，約 10 tok/s。

| # | 機制 | 屬於 | 原理 | 估計效果（推算） | 代價／風險 |
|---|---|---|---|---|---|
| 1 | **冷 experts 用更低精度** | 減少 bytes | 熱的維持原精度放在 VRAM/DRAM；冷的在 SSD 上用約 1–1.3 bpw 存（Q2_0 約 2.1 bpw） | SSD 流量約減半，約 20 tok/s | 有損，但只影響少用的 experts；需要離線重新量化，再用 KL／needle 測試 |
| 2 | **拉長 verify window 或批次** | 攤提 | 一個 expert 讀一次，服務多個 token | 取決於連續 token 之間路由的重疊程度，**沒有數據** | 受草稿模型的接受率限制；單一使用者沒辦法靠多個請求批次 |
| 3 | **兩顆 NVMe 條帶化** | 提高頻寬 | experts 檔切兩份放兩顆 SSD，同時讀 | 約 1.5–1.8 倍 | 要改 engine；不需要 OS RAID |
| 4 | **Direct I/O 取代 mmap** | 提高頻寬、省 DRAM | 不經過 OS file cache，用 unbuffered I/O（`FILE_FLAG_NO_BUFFERING`／DirectStorage）加深佇列，讀進固定的 staging buffer | DRAM 用量可預測，省掉 file cache；頻寬可能提高 | Windows 的載入路徑已有 unbuffered 讀取（#357），decode 路徑還沒有 |
| 5 | **miss 時減少 top-k** | 減少 bytes | 權重最小的幾個 expert 沒命中時就跳過，並重新正規化權重 | 每跳過一個約省 1.4 MB | 有損；`--dump-routing` 的 trace 有權重，可以先離線估計 |
| 6 | **只讀 expert 內部的部分 rows** | 減少 bytes | 類似 Apple 的 "LLM in a flash"：預測會活化的神經元，只讀那些 rows | 理論上很大 | SwiGLU 的活化稀疏度不高，預測器本身也有成本；最像研究題目 |
| 7 | **剪掉更多 experts**（像 Coder） | 模型變小 | 直接減少 experts | 最確定有效 | 品質損失、綁定特定領域（#438：中文變差） |

1＋3 疊加，16/16 的機器上限可能到約 30–35 tok/s；2 和 5 還能再加，但沒有數據前不估。

### 物理下限

- **DRAM 不會降到 0。** Windows、server、engine 的 buffer 大約要 6–10 GB。能降的是「experts 占用的 DRAM」。
- **極端情況：DRAM 完全不放 experts**，只用約 10 GB VRAM 加 SSD。每個 token 要讀 331 MB，以 2 GB/s 計約 6 tok/s；
  加上條帶化和低精度冷 experts，約 15–20 tok/s（推算）。這時 GPU 負責所有運算，PCIe 5.0 x16 的頻寬遠高於 SSD，
  不會是瓶頸。
- **HDD**：隨機讀比 NVMe 慢 10–20 倍，以上估計都要再除以 10–20，加上 n-gram 表的尋道時間，不實用。

---

## 6. Hands-on 路線（由便宜到貴）

| 步驟 | 內容 | 需要什麼 |
|---|---|---|
| 1 | 練習 1：跑 setup 的判斷公式 | 只要 Python |
| 2 | 練習 3：算命中率曲線，改成每層一個快取 | 只要 Python |
| 3 | 讀 `src/core/expert_cache.cpp`、`include/strata/core/expert_cache.hpp`，弄懂 expert cache 怎麼填、怎麼換 | 只要讀程式 |
| 4 | 安裝 Coder，`START-HERE.bat --setup --family coder`。setup 會選 resident 模式 | 下載約 58 GB |
| 5 | 短 prompt、4K context 的 smoke run，加 `--stats`，確認 `resident RAM: ... blob reads from the file` 是 0 | GPU |
| 6 | 用 `--dump-routing FILE` 錄一份自己的路由 trace，離線模擬：(a) adaptive cache 比靜態快取好多少；(b) verify window 裡的 experts 重疊多少（機制 2）；(c) 跳過低權重 miss 的影響（機制 5） | GPU，一次短跑 |
| 7 | 安裝 Q2_0，比較 plain mmap、`--resident-budget-gib 18/20/22`、`--low-ram resident`，每種至少跑 3 次，記錄 tok/s、`file_mb`、free RAM | 下載約 66 GB；建議放在 C:（PCIe 4.0） |
| 8 | 選一個機制做原型（建議先做 #1 冷 experts 低精度），用 repo 的 KL／needle 工具量化品質 | 開發 |

跑之前，要先關掉 ComfyUI 等占用 VRAM/DRAM 的程式。分析當下 VRAM 已用約 15 GB，DRAM 只剩 10.5 GB。

---

## 7. 還沒回答的問題

- `expert-profile.bin` 的 trace 用的是哪種工作負載？在自己的使用情境下，曲線會一樣嗎？
- adaptive cache 在單一對話裡，比靜態快取的命中率高多少？
- 一個 verify window（約 3.5 個 token）裡，不重複的 experts 有多少？
- 在這台機器上，Q2_0 budget 模式實際是多少 tok/s？
- 冷 experts 降到約 1.2 bpw，KL 會增加多少？
