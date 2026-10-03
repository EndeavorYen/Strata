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

這些機制多數已經有相關研究，見第 8 節。

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

---

## 8. 相關研究

2026-10-03 上網查的，內容整理自各論文的 arXiv 摘要或 HTML 版本，我沒有自己重現。表中的數字都是**作者在他們自己的硬體和模型上量的**，不能直接換算成 Strata 的數字。

### 8.1 基礎：MoE offloading 怎麼開始的

| 論文 | 時間 | 重點 | 和 Strata 的關係 |
|---|---|---|---|
| [Fast Inference of MoE LMs with Offloading](https://arxiv.org/abs/2312.17238)（Eliseev、Mazur） | 2023-12 | Mixtral-8x7B 跑在桌機 GPU 上：GPU 放 experts 的 LRU 快取，加上 speculative prefetch。觀察到相鄰 token 會重複用到相同的 experts，而且前面幾層的 hidden state 已經「知道」後面幾層要用哪些 experts | Strata 的 GPU expert cache 和 lookahead prefetch 都是這條路線 |
| [Fiddler](https://arxiv.org/abs/2402.07033)（ICLR 2025） | 2024-02 | 沒命中的 expert 直接由 CPU 算，不搬權重到 GPU。單張 24 GB GPU 跑未壓縮的 Mixtral-8x7B（>90 GB），超過 3 tok/s | Strata「CPU 在 RAM 裡就地計算 miss」的同一個想法 |
| [KTransformers](https://github.com/kvcache-ai/ktransformers)（SOSP 2025） | 2025 | attention 放 GPU，experts 放 CPU/DRAM，搭配 AMX/AVX-512 kernel。單張 24 GB GPU 加上約 512 GB RAM，跑 DeepSeek-R1/V3 671B | CPU/GPU 混合推論的代表，但需要大量 DRAM |
| [LLM in a flash](https://arxiv.org/abs/2312.11514)（Apple，ACL 2024） | 2023-12 | 權重放在 flash，按需載入 DRAM。windowing 重用最近活化過的神經元；row-column bundling 讓每次讀取的區塊更大。可以跑大到 DRAM 2 倍的模型 | 機制 #6（只讀部分 rows）的出處 |

### 8.2 冷 experts 用低精度（對應機制 #1）

**這個方向已經有人做過，而且有正面結果。** 要實作的話，從這幾篇開始讀。

| 論文 | 時間 | 重點 | 作者報告的數字 |
|---|---|---|---|
| [HOBBIT](https://arxiv.org/abs/2411.01433) | 2024-11 | 每個 expert 存好幾種精度；沒命中、而且不重要的 expert，改從 CPU 記憶體或 SSD 讀低精度版本（FP16→INT4、INT8→INT2）。另有逐層預測 prefetch，以及多維度的快取策略 | expert 載入 I/O 最多減少 4 倍；桌機 GPU 上 decode 快 3–4 倍；精度損失很小（Mixtral、Phi-MoE） |
| [DynaExq](https://arxiv.org/abs/2511.15015) | 2025-11（2026-09 修訂） | 依執行時的流量決定精度：常用的 experts 用高精度常駐，其他保留一份低精度備援 | Qwen3-80B MoE 的準確率，比靜態 PTQ 從 73.09% 提高到 77.57% |
| [Low-Rank Compensation](https://arxiv.org/abs/2512.17073) | 2025-12 | 所有 experts 都存低 bit；router 挑出最重要的 n 個（n < k）時，再傳一份小的 low-rank 補償因子把精度補回來 | 摘要只說 bandwidth 和準確率的取捨更好，沒有具體數字 |

### 8.3 SSD 當第三層（對應機制 #3、#4）

| 論文 | 時間 | 重點 | 作者報告的數字 |
|---|---|---|---|
| [SSD-LLaMA](https://arxiv.org/abs/2609.18110) | 2026-09 | **最接近我們的問題**。主要測試機：RTX 5090 32 GB、**16 GB DRAM**、PCIe 5.0 NVMe（循序讀 9 GiB/s）。做法：(1) expert-pack layout，把一個 expert 的權重排成連續區塊，一次 `O_DIRECT` 讀完；(2) 用 rANS 做無損壓縮，在 GPU 上解壓；(3) SSD/RAM/VRAM 三層快取（看 recency 和 frequency）；(4) CPU/GPU 的動態分工 | 無損壓縮讓 SSD 流量少了 33.2%（每 token 從 8.89 GB 降到 5.93 GB）；SSD 達到循序頻寬峰值的 77.7%（baseline 43.2%）；decode 比 llama.cpp 快 2.10–15.58 倍；Kimi-K3（2.8T 參數）decode 0.465 tok/s |
| [FlashMoE](https://arxiv.org/abs/2601.17063) | 2026-01 | experts 放在 SSD；用一個小的 ML 模型結合 recency 和 frequency，決定快取要換掉誰 | 命中率比 LRU/LFU 最多高 51%；最多快 2.6 倍 |
| [SSD Offloading … Considered Harmful in Energy Efficiency](https://arxiv.org/abs/2508.06978) | 2025-08 | 從**耗電**角度看：SSD 每 bit 的讀取能耗比 DRAM 高很多 | 每個 token 的能耗最多是 HBM baseline 的約 12 倍；prefetch 可以藏住延遲，但省不了電 |
| [Memory-Sovereign Inference](https://arxiv.org/abs/2608.23805) | 2026-08 | 用 Qwen3-Next（48 × 512 個 layer-expert，**和本專案的模型形狀相同**）做儲存支援的推論，重點是怎麼**嚴格驗證**記憶體用量和輸出正確性。指出常見量法的漏洞：行程的記憶體讀數不含 page cache；能生成不代表非同步路徑是對的 | host 11 GiB + GPU 24 GiB；64 個 token 範圍內的 logits 完全一致 |

SSD-LLaMA 的兩個做法可以直接對照 Strata：
- **expert-pack + `O_DIRECT`**：對應機制 #4。Strata 的 `experts.bin` 本來就是每個 expert 連續存放，差別在 decode 時走的是 mmap。
- **無損壓縮省 33%**：我之前沒有把壓縮列進機制表。他們的模型權重格式和 Strata 的 2–3 bit i-quant 不同，量化很重的權重還能壓多少，要自己測。

### 8.4 預測、快取策略、路由局部性（對應機制 #2、#5）

| 論文 | 時間 | 重點 | 作者報告的數字 |
|---|---|---|---|
| [Not All Models Suit Expert Offloading](https://arxiv.org/abs/2505.16056) | 2025-05（2026-02 修訂） | 分析 20 個 MoE 模型的「局部路由一致性」，也就是一段連續 token 會不會一直用同一批 experts。有 shared expert 的模型一致性較低；experts 依領域分工的模型比依詞彙分工的好 | 多數模型的快取大小約為啟用 experts 數的 2 倍時，效果和成本比較平衡 |
| [DALI](https://arxiv.org/abs/2602.03495) | 2026-02 | 本地 PC：用 0-1 整數最佳化（貪婪解）動態分配 CPU/GPU 的工作；利用層與層之間的 residual 預測接下來的高負載 experts；快取替換也考慮 workload | 摘要只說「顯著加速」 |
| [ReMoE](https://arxiv.org/abs/2605.27081) | 2026-05 | **微調 router**，讓它偏好最近用過的 experts，路由在時間上更穩定，重用率更高 | expert 重用率 +26%；Jetson Orin NX 上 TPOT 少 43.6–49.8%；作者報告下游任務的表現維持不變 |
| [Importance-Driven Expert Scheduling](https://arxiv.org/abs/2508.18983) | 2025-08 | 被選中但不重要、又沒命中的 expert，換成 GPU 上已經有、功能相近的 expert | decode 延遲少 48%，命中率超過 60%，準確率「幾乎無損」 |
| [Mixture of Cache-Conditional Experts](https://arxiv.org/abs/2412.00099) | 2024-12 | 路由時優先選已經在快取裡的 experts，不嚴格照 top-K | 行動裝置 |
| [Cache-Aware Joint Router Adaptation](https://arxiv.org/abs/2609.04895) | 2026-09 | 用 post-training 調整 backbone 和小型輔助 router，讓快取更有效，推論時仍維持原本的 top-K 規則 | — |

這幾篇和機制 #5（miss 時減少 top-k）是同一類：都在**改變模型的計算**，換取更少的 I/O。差別在於「跳過」、「換成相近的」，還是「訓練 router 讓它自己偏好快取裡的」。

### 8.5 其他值得一看

- [MoBiLE](https://arxiv.org/abs/2510.12357)（2025-10）：消費級 GPU 上的 offloading，混用「大」和「小」的 experts。
- [CPU-GPU Collaborative Inference on Memory-Limited Systems](https://arxiv.org/abs/2512.16473)（2025-12）：GPU expert cache 加上 CPU 協同計算。
- [CoX-MoE](https://arxiv.org/abs/2605.17889)（DAC 2026）：AMX CPU 和 GPU 協同執行，把 expert 的計算合併起來做。
- [Automated Tensor Scheduling for Hybrid CPU-GPU Inference on Consumer Devices](https://arxiv.org/abs/2607.10183)（2026-07）。
- 論文清單：[awesome-moe-inference](https://github.com/MoE-Inf/awesome-moe-inference/)。

### 8.6 對這份筆記的影響

1. **機制 #1（冷 experts 低精度）**：HOBBIT 和 DynaExq 已經證明可行，難點會在 Strata 本身的格式已經是 2–3 bit，比這些論文的 INT4/INT8 起點低很多，還能往下降多少不確定。
2. **機制 #3／#4（SSD 頻寬）**：SSD-LLaMA 在 16 GB DRAM 上的做法（expert-pack、`O_DIRECT`、無損壓縮）是最直接的參考；壓縮應該加進機制表，當作第 8 項候選。
3. **第 7 節的問題「verify window 裡的 experts 重疊多少」**：可以先參考 2505.16056 的 SCH 指標怎麼定義，再拿自己錄的 `--dump-routing` trace 來算。
4. **測量的嚴謹度**：Memory-Sovereign Inference 指出「行程記憶體不含 page cache」，Strata 的 mmap 模式剛好有這個問題。實測時要同時記錄系統的 free RAM，不能只看 engine 自己報的數字。
5. **耗電**：SSD 路線省的是 DRAM 的錢，但每個 token 更耗電（2508.06978）。桌機可能不在意，筆電要考慮。

### 建議閱讀順序

1. Eliseev & Mazur（2312.17238）：短，讀完就懂 offloading 的基本問題
2. Fiddler（2402.07033）：為什麼 CPU 直接算比搬資料好
3. HOBBIT（2411.01433）：混合精度
4. SSD-LLaMA（2609.18110）：SSD 當第三層，硬體條件最接近
5. Not All Models Suit Expert Offloading（2505.16056）：怎麼量化「局部性」
6. 其餘依興趣挑
