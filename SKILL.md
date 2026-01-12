## skill: table_formatting
**說明（Description）：**  
產出表格時一律使用標準 Markdown 語法：

- 第一列為標題列，必須加粗（例如 `**欄位**`）。
- 欄位名稱盡量貼近 FDA 常用術語（如 Device Name, Indications for Use）。
- 若資料缺漏，請填寫 `Not Provided`，不可留白。
- 避免過寬的欄位名稱，保持可讀性。

---

## skill: regulatory_citation
**說明：**  
在引用法規或指引要求時：

- 儘量標明具體法規或指引段落，例如：  
  - `21 CFR 807.92`  
  - `FDA Guidance "Format and Content of a 510(k) Summary" (年份)`  
- 若文字明確引用來源文件某頁或某節，可附上 `(ref: p.XX)` 形式註記。
- 不可捏造不存在的條文編號。

---

## skill: checklist_generation
**說明：**  
將要求轉換為可操作的檢核表：

- 每一項目應為一句清楚的「可判斷」陳述（可用「是/否」評估）。
- 需包含：**項目描述**、**預期文件或證據**（如 Test Report, Protocol, Labeling）。
- 優先使用 Markdown 清單（`- [ ]`），必要時可再以表格呈現。

---

## skill: risk_analysis
**說明：**  
在評估風險時：

- 明確指出風險來源（例如：材質差異、滅菌方式改變、軟體更新）。
- 連結至 ISO 14971 或相關 FDA 指引的概念（如 hazard, harm, risk control）。
- 區分「已充分控制」與「可能需要額外資料」的風險議題。
- 不要給出臨床或法規「保證」，僅就風險觀點提出專業判斷。

---

## skill: note_structuring
**說明：**  
整理雜亂筆記為具結構的內容：

- 自動辨識主題，加入合適標題層級（`#`, `##`, `###`）。
- 把列表、決策點、待辦事項轉成條列或表格。
- 清楚區分「事實紀錄」與「待確認事項 / 風險點」。

---

## skill: language_polishing
**說明：**  
進行用語潤飾與校正：

- 修正文法、拼字，保留技術內容不變。
- 中英混合時，尊重原英文專有名詞（如 predicate device, SE, RTA）。
- 對醫療器材名詞與統計用語保持嚴謹、專業。

---

## skill: executive_summary
**說明：**  
撰寫高階管理層可讀的「執行摘要」：

- 在 300–800 字內點出審查重點、關鍵風險與建議結論。
- 避免過多技術細節，但可簡短提及最重要的試驗或差異。
- 句子清楚、段落分明，適合放入審查報告封面摘要。

---

## skill: question_generation
**說明：**  
產生可用於 FDA Reviewer 或內部 QA 的提問清單：

- 問句應具體、可行，指向缺失的資料或不清楚的論點。
- 優先使用「Have you considered…」「Where is the evidence for…」等開頭。
- 適用於 AI request、deficiency letter 初稿。

---

## skill: keyword_extraction
**說明：**  
擷取並整理關鍵字與關鍵實體（entities）：

- 包含：裝置名稱、申請人名稱、規格、測試類型、法規條文編號。
- 可依類別歸納（例如：Device, Test, Regulation, Risk）。
- 輸出為條列或表格，方便後續著色或檢索使用。

---

## skill: pattern_detection
**說明：**  
從多段文字或多份文件找出重複出現的主題與問題模式：

- 例如：多次提及 biocompatibility 疑慮、滅菌流程不清楚等。
- 整理成「模式說明」與「可能影響」兩欄，協助 reviewer 快速聚焦。

---

## skill: narrative_generation
**說明：**  
將零散資訊組織成連貫敘事：

- 適用於「Substantial Equivalence 論述」、「開發歷程敘述」、「臨床證據故事」。
- 應維持客觀、正式口吻，避免過度宣傳感。

---

## skill: forecasting
**說明：**  
根據目前資訊推估可能的審查走向：

- 例如：是否可能收到 Additional Information (AI) 要求、常見的缺失類型。
- 必須明確標註為「AI 預測，非法律意見」。
- 不應給出保證或明確法規結論，而是提供參考性的風險判斷。

---

## skill: sentiment_regulatory
**說明：**  
針對文本進行情緒 / 信心評估，特別是「審查信心」：

- 將結果量化為 0–100 分，並給出簡短文字標籤（如「高風險缺失多」「資料接近完整」）。
- 提供簡短理據，說明主要影響信心的因素。

---

## skill: translation_zh_en
**說明：**  
將繁體中文內容翻譯成專業、自然的英文：

- 保留法律及技術術語精準度。
- 可根據 FDA 常見用語調整措辭（如「substantial equivalence」而非「similarity」）。

---

## skill: translation_en_zh
**說明：**  
將英文技術 / 法規內容翻譯為繁體中文：

- 儘量保留英文原文關鍵詞於括號內，例如「實質等同性（Substantial Equivalence, SE）」。
- 適合用於內部訓練或跨部門溝通。

---

## skill: yaml_safe_editor
**說明：**  
協助使用者理解與調整 `agents.yaml` 設定（概念層面）：

- 解釋各欄位（model, temperature, max_tokens, system_prompt）的意義與影響。
- 提供保守、安全的參數建議。
- 不輸出實際程式碼變更，只給建議與示範片段。

---

## skill: audit_trail
**說明：**  
為模型輸出建立可追溯的「提示紀錄」摘要：

- 列出使用的系統提示重點、使用者指令、模型名稱、重要參數（temperature, max_tokens）。
- 與正式輸出分開，方便放入審查檔案附錄。

---

## skill: meeting_minutes
**說明：**  
將會議紀錄轉成正式會議記錄（Minutes）：

- 包含：會議目的、出席者、討論重點、決議事項、後續行動與負責人。
- 以 Markdown 標題與列表呈現，便於轉存到文件系統。

---

## skill: deficiency_letter
**說明：**  
協助草擬 FDA deficiency letter / AI letter 的技術性段落：

- 每一缺失項目應包含：背景說明、具體缺失描述、相關法規 / 指引引用。
- 採用禮貌、專業且明確的語氣。

---

## skill: ai_request_letter
**說明：**  
針對廠商回應，草擬 Additional Information (AI) 要求：

- 指明需補件的資料類型（試驗報告、風險分析、標籤樣稿等）。
- 清楚說明補件目的與期望格式。

---

## skill: statistics_clarity
**說明：**  
檢視並解釋統計方法與結果：

- 確認樣本數、檢定方法、顯著水準是否合理。
- 將複雜統計結果轉成 reviewer 易讀的文字摘要。

---

## skill: clinical_evidence_grading
**說明：**  
整理並評等臨床證據品質：

- 區分臨床研究設計型態（RCT, observational, retrospective, 等）。
- 評估樣本數、偏差可能性、是否支撐預期用途。

---

## skill: risk_control_linking
**說明：**  
協助將風險分析中的「風險控制措施」連結到實際驗證 / 確認活動：

- 例如：風險控制為「材料更換」則對應的生物相容性試驗。
- 以表格呈現：風險項目、風險控制、對應測試 / 文件。

---

## skill: labeling_consistency
**說明：**  
檢查標籤與 IFU 在關鍵資訊上的一致性：

- 包含：適應症、禁忌症、警語、使用方式、預期使用者與環境。
- 標示任何內部矛盾或與主體文件不一致之處。

---

## skill: usability_issues
**說明：**  
從文本中找出可用性 / 人因工程相關風險：

- 例如：操作步驟複雜、警示不清楚、使用者訓練需求高。
- 建議是否需要進一步 summative / formative usability study。

---

## skill: sw_cybersecurity
**說明：**  
聚焦於軟體及網路安全議題：

- 確認是否涵蓋認證管理、存取控制、資料完整性、更新機制等。
- 參考 FDA 網路安全指引中的常見控制項。

---

## skill: biocomp_matrix
**說明：**  
整理生物相容性測試矩陣：

- 按照 ISO 10993 的一般分類，整理接觸類型 / 接觸時間對應之必需試驗。
- 檢查是否有「未提供但應有」的試驗項目。

---

## skill: sterilization_review
**說明：**  
審查滅菌方法與相關驗證：

- 確認滅菌方式（如 EO, Gamma, Steam）與 SAL。
- 檢查滅菌驗證報告、殘留物 (residuals) 評估是否敘述明確。

---

## skill: shelf_life_review
**說明：**  
整理與評估保存期限 / 穩定性試驗：

- 區分加速與實時間試驗結果。
- 關聯至標籤上宣稱之有效期限。

---

## skill: rta_screening
**說明：**  
依「Refuse to Accept」(RTA) 準則檢核 510(k) 文件完整性：

- 建立行政項目檢查清單（簽名、表格、費用、必要附件）。
- 明確標註「缺少/疑似缺少」的項目，便於快速修正。

---

## skill: se_argumentation
**說明：**  
協助撰寫「Substantial Equivalence (SE) 論述」：

- 清楚對比 subject device 與 predicate device 在預期用途與技術特性上的異同。
- 強調相同處、合理說明差異是否引入新疑慮。
- 避免誇大或缺乏支撐證據的結論。
