import os
import re
import json
import random
from io import StringIO
from datetime import datetime
from typing import Dict, Any, Optional, List

import streamlit as st
import yaml
import pandas as pd
from pypdf import PdfReader
import plotly.graph_objects as go

# --- Optional / provider SDKs -------------------------------------------------
try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover
    genai = None

try:
    from anthropic import Anthropic
except ImportError:  # pragma: no cover
    Anthropic = None

# --- Constants ----------------------------------------------------------------

APP_VERSION = "1.0.0"
DEFAULT_MAX_TOKENS = 12000

DEFAULT_AGENTS_YAML = """
fda_summarizer:
  name: "FDA Summary Expert"
  description: "Specialist in generating 510(k) summaries."
  model_provider: "google"
  model_name: "gemini-2.5-flash"
  temperature: 0.2
  max_tokens: 8000
  system_prompt: |
    You are an expert Regulatory Affairs Specialist for the FDA.
    Your task is to generate a comprehensive 510(k) summary compliant with 21 CFR 807.92.
    You must extract facts accurately from the provided document.
    You must output exactly 5 tables as requested.
  skills:
    - "pdf_extraction"
    - "table_formatting"
    - "regulatory_citation"

fda_guidance_expert:
  name: "Guidance Synthesizer"
  description: "Converts guidance docs into checklists."
  model_provider: "google"
  model_name: "gemini-3-pro-preview"
  temperature: 0.3
  max_tokens: 8000
  system_prompt: |
    You are a Senior FDA Reviewer.
    Analyze the provided guidance document.
    Create actionable checklists for a new reviewer.
  skills:
    - "checklist_generation"
    - "risk_analysis"

note_keeper:
  name: "Regulatory Scribe"
  description: "Organizes notes and applies AI Magics."
  model_provider: "openai"
  model_name: "gpt-4o-mini"
  temperature: 0.5
  max_tokens: 4000
  system_prompt: |
    You are a helpful assistant organizing regulatory notes into clear, well-structured Markdown.
"""

DEFAULT_SKILL_MD = """
# SKILL.md

## skill: table_formatting
**Instruction:** Ensure all tables are formatted using standard Markdown syntax.
Header rows must be bold.
If data is missing, write "Not Provided" instead of leaving blank.

## skill: regulatory_citation
**Instruction:** When making a claim about a requirement, cite the specific section of the
uploaded guidance or 21 CFR reference if applicable.

## skill: checklist_generation
**Instruction:** Convert requirements into concise, actionable checklist items with clear pass/fail criteria.

## skill: risk_analysis
**Instruction:** Where appropriate, briefly describe regulatory risk associated with missing or weak data.
"""

MODEL_CONFIG: Dict[str, Dict[str, str]] = {
    "gpt-4o-mini": {"provider": "openai"},
    "gpt-4.1-mini": {"provider": "openai"},
    "gemini-2.5-flash": {"provider": "google"},
    "gemini-2.5-flash-lite": {"provider": "google"},
    "gemini-3-pro-preview": {"provider": "google"},
    "claude-3-5-sonnet-latest": {"provider": "anthropic"},
    "claude-3-haiku-20240307": {"provider": "anthropic"},
    "grok-4-fast-reasoning": {"provider": "grok"},
    "grok-3-mini": {"provider": "grok"},
}

PAINTER_STYLES = [
    "Leonardo da Vinci",
    "Vincent van Gogh",
    "Pablo Picasso",
    "Claude Monet",
    "Rembrandt",
    "Frida Kahlo",
    "Henri Matisse",
    "Salvador Dalí",
    "Katsushika Hokusai",
    "Mark Rothko",
    "Jackson Pollock",
    "Marc Chagall",
    "Edgar Degas",
    "Paul Cézanne",
    "Caravaggio",
    "J. M. W. Turner",
    "Francisco Goya",
    "Gustav Klimt",
    "Johannes Vermeer",
    "Jean-Michel Basquiat",
]

# --- Simple i18n --------------------------------------------------------------

TEXT = {
    "en": {
        "app_title": "FDA 510(k) Agentic AI Review System – WOW UI",
        "dashboard": "Dashboard",
        "summary_tool": "510(k) Summary Tool",
        "guidance_assistant": "Guidance Assistant",
        "note_keeper": "Note Keeper (The Lab)",
        "settings": "Settings (Agentsmith)",
        "theme": "Theme",
        "theme_auto": "Auto",
        "theme_light": "Light",
        "theme_dark": "Dark",
        "language": "Language",
        "style": "Painter Style",
        "style_jackpot": "Jackpot!",
        "api_keys": "API Keys",
        "openai_key": "OpenAI API Key",
        "gemini_key": "Gemini API Key",
        "anthropic_key": "Anthropic API Key",
        "grok_key": "Grok API Key",
        "api_loaded_env": "Loaded from environment / secrets",
        "api_input_needed": "Please enter your API key.",
        "model_select": "Select Model",
        "agent_select": "Active Agent Persona",
        "max_tokens": "Max tokens",
        "temperature": "Temperature",
        "extra_prompt": "Optional extra instructions / focus",
        "file_uploader_510k": "Upload 510(k) submission file (.pdf, .txt, .md)",
        "file_uploader_guidance": "Upload FDA Guidance file (.pdf, .txt, .md)",
        "generate_summary": "Generate 510(k) Summary",
        "generate_guidance": "Create Guidance Checklists",
        "summary_output": "510(k) Summary Output",
        "guidance_output": "Guidance Output",
        "chained_input": "Chained input for next agent",
        "set_as_chained": "Use this output as input for next agent",
        "note_input": "Raw / Working Notes",
        "note_output": "Transformed Notes",
        "run_magic": "Run Magic",
        "magic_select": "AI Magics",
        "magic_ai_transform": "AI Transformation",
        "magic_keywords": "AI Keywords (Colorizer)",
        "magic_pattern": "Pattern Spotter",
        "magic_narrative": "Narrative Weaver",
        "magic_trend": "Trend Forecaster",
        "magic_socratic": "Socratic Mirror",
        "magic_mood": "Mood Scape",
        "keywords_input": "Keywords to highlight (comma-separated)",
        "keywords_color": "Highlight color",
        "skills_source": "SKILL.md (read-only template in this build)",
        "yaml_editor": "agents.yaml Editor",
        "yaml_validate": "Validate & Load YAML",
        "yaml_download": "Download current agents.yaml",
        "yaml_upload": "Upload agents.yaml",
        "append_prompt": "Append prompts to output (for audit trail)",
        "stats_tokens": "Total tokens used",
        "stats_docs": "Documents analyzed",
        "stats_runs": "Agent runs",
        "wow_status": "WOW Status Indicators",
        "mood_confidence": "Review Confidence",
        "dashboard_intro": "Interactive overview of your current review session.",
        "no_text": "No text available.",
        "needs_api": "At least one API key is required to run agents.",
        "processing": "Processing with selected agent and model...",
        "pdf_warning": "If this PDF is scanned images only, text extraction may be incomplete.",
        "download_md": "Download as Markdown",
        "download_txt": "Download as Text",
        "output_view_mode": "Output view mode",
        "view_markdown": "Markdown",
        "view_text": "Plain text",
    },
    "zh": {
        "app_title": "FDA 510(k) 智能審查系統 – WOW 介面",
        "dashboard": "儀表板",
        "summary_tool": "510(k) 摘要工具",
        "guidance_assistant": "指引助理",
        "note_keeper": "筆記管理（實驗室）",
        "settings": "設定（Agentsmith）",
        "theme": "主題",
        "theme_auto": "自動",
        "theme_light": "亮色",
        "theme_dark": "暗色",
        "language": "語言",
        "style": "畫家風格",
        "style_jackpot": "幸運轉盤！",
        "api_keys": "API 金鑰",
        "openai_key": "OpenAI API 金鑰",
        "gemini_key": "Gemini API 金鑰",
        "anthropic_key": "Anthropic API 金鑰",
        "grok_key": "Grok API 金鑰",
        "api_loaded_env": "已由環境 / secrets 載入",
        "api_input_needed": "請輸入您的 API 金鑰。",
        "model_select": "選擇模型",
        "agent_select": "選擇代理人格",
        "max_tokens": "最大 tokens",
        "temperature": "溫度（隨機性）",
        "extra_prompt": "額外說明 / 著重重點（選填）",
        "file_uploader_510k": "上傳 510(k) 文件（.pdf, .txt, .md）",
        "file_uploader_guidance": "上傳 FDA 指引文件（.pdf, .txt, .md）",
        "generate_summary": "產生 510(k) 摘要",
        "generate_guidance": "建立指引檢核表",
        "summary_output": "510(k) 摘要輸出",
        "guidance_output": "指引輸出",
        "chained_input": "下一個代理的輸入內容",
        "set_as_chained": "將此輸出作為下一個代理的輸入",
        "note_input": "原始 / 工作筆記",
        "note_output": "轉換後的筆記",
        "run_magic": "執行魔法",
        "magic_select": "AI 魔法",
        "magic_ai_transform": "AI 整理轉換",
        "magic_keywords": "AI 關鍵字著色",
        "magic_pattern": "模式偵測",
        "magic_narrative": "敘事編織",
        "magic_trend": "趨勢預測",
        "magic_socratic": "蘇格拉底反思",
        "magic_mood": "情緒景觀",
        "keywords_input": "欲標示之關鍵字（以逗號分隔）",
        "keywords_color": "標示顏色",
        "skills_source": "SKILL.md（本版本為唯讀樣板）",
        "yaml_editor": "agents.yaml 編輯器",
        "yaml_validate": "驗證並載入 YAML",
        "yaml_download": "下載目前 agents.yaml",
        "yaml_upload": "上傳 agents.yaml",
        "append_prompt": "在輸出中附上提示內容（稽核用）",
        "stats_tokens": "累計 tokens",
        "stats_docs": "已分析文件數",
        "stats_runs": "代理執行次數",
        "wow_status": "WOW 狀態指標",
        "mood_confidence": "審查信心",
        "dashboard_intro": "檢視本次審查工作的整體概況。",
        "no_text": "目前沒有可用文字。",
        "needs_api": "執行代理前，至少需設定一組 API 金鑰。",
        "processing": "正在使用選定代理與模型處理中…",
        "pdf_warning": "若本 PDF 為純影像掃描，文字擷取可能不完整。",
        "download_md": "下載為 Markdown",
        "download_txt": "下載為純文字",
        "output_view_mode": "輸出檢視模式",
        "view_markdown": "Markdown",
        "view_text": "純文字",
    },
}


def t(key: str) -> str:
    lang = st.session_state.get("language", "en")
    return TEXT.get(lang, TEXT["en"]).get(key, key)


# --- LLM Client ---------------------------------------------------------------

class LLMClient:
    def __init__(
        self,
        openai_key: Optional[str] = None,
        gemini_key: Optional[str] = None,
        anthropic_key: Optional[str] = None,
        grok_key: Optional[str] = None,
    ):
        self.openai_client = None
        self.gemini_configured = False
        self.anthropic_client = None
        self.grok_client = None

        if openai_key and OpenAI is not None:
            self.openai_client = OpenAI(api_key=openai_key)

        if gemini_key and genai is not None:
            genai.configure(api_key=gemini_key)
            self.gemini_configured = True

        if anthropic_key and Anthropic is not None:
            self.anthropic_client = Anthropic(api_key=anthropic_key)

        # Grok via OpenAI-compatible client
        if grok_key and OpenAI is not None:
            try:
                self.grok_client = OpenAI(api_key=grok_key, base_url="https://api.x.ai/v1")
            except Exception:
                self.grok_client = None

    def generate(
        self,
        provider: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        provider = provider.lower()
        if provider == "openai":
            return self._generate_openai(model, system_prompt, user_prompt, max_tokens, temperature)
        elif provider == "google":
            return self._generate_gemini(model, system_prompt, user_prompt, max_tokens, temperature)
        elif provider == "anthropic":
            return self._generate_anthropic(model, system_prompt, user_prompt, max_tokens, temperature)
        elif provider == "grok":
            return self._generate_grok(model, system_prompt, user_prompt, max_tokens, temperature)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _generate_openai(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Dict[str, Any]:
        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized or API key missing.")

        resp = self.openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = resp.choices[0].message.content
        usage = getattr(resp, "usage", None)
        tokens = 0
        if usage is not None:
            tokens = getattr(usage, "total_tokens", 0) or (
                getattr(usage, "prompt_tokens", 0) + getattr(usage, "completion_tokens", 0)
            )
        return {"text": text, "tokens": tokens}

    def _generate_gemini(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Dict[str, Any]:
        if not self.gemini_configured or genai is None:
            raise RuntimeError("Gemini client not initialized or API key missing.")
        gm = genai.GenerativeModel(model)
        prompt = f"{system_prompt.strip()}\n\nUser:\n{user_prompt.strip()}"
        resp = gm.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        text = resp.text
        usage = getattr(resp, "usage_metadata", None)
        tokens = 0
        if usage is not None:
            tokens = getattr(usage, "total_token_count", 0)
        return {"text": text, "tokens": tokens}

    def _generate_anthropic(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Dict[str, Any]:
        if not self.anthropic_client:
            raise RuntimeError("Anthropic client not initialized or API key missing.")
        msg = self.anthropic_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        # Concatenate text blocks
        parts = []
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", ""))
        text = "".join(parts) if parts else ""
        usage = getattr(msg, "usage", None)
        tokens = 0
        if usage is not None:
            tokens = getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0)
        return {"text": text, "tokens": tokens}

    def _generate_grok(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Dict[str, Any]:
        if not self.grok_client:
            raise RuntimeError("Grok client not initialized or API key missing.")
        resp = self.grok_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = resp.choices[0].message.content
        usage = getattr(resp, "usage", None)
        tokens = 0
        if usage is not None:
            tokens = getattr(usage, "total_tokens", 0) or (
                getattr(usage, "prompt_tokens", 0) + getattr(usage, "completion_tokens", 0)
            )
        return {"text": text, "tokens": tokens}


# --- Utilities ----------------------------------------------------------------

def load_agents() -> Dict[str, Any]:
    if "agents_config" in st.session_state:
        return st.session_state["agents_config"]

    path = "agents.yaml"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        data = yaml.safe_load(DEFAULT_AGENTS_YAML)
    st.session_state["agents_config"] = data
    return data


def load_agents_raw_text() -> str:
    path = "agents.yaml"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return DEFAULT_AGENTS_YAML.strip()


def load_skill_md() -> str:
    path = "SKILL.md"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return DEFAULT_SKILL_MD.strip()


def extract_text_from_file(uploaded_file) -> str:
    suffix = uploaded_file.name.lower()
    if suffix.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        texts = []
        for page in reader.pages:
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                texts.append("")
        return "\n\n".join(texts)
    else:
        # txt / md
        data = uploaded_file.read()
        try:
            return data.decode("utf-8")
        except Exception:
            return data.decode("latin-1", errors="ignore")


def apply_keywords_colorizer(text: str, keywords: List[str], color: str) -> str:
    if not text or not keywords:
        return text
    pattern = r"(" + "|".join(re.escape(k.strip()) for k in keywords if k.strip()) + r")"
    if pattern == "()":
        return text
    # Case-insensitive, preserve original case
    def repl(m):
        word = m.group(1)
        return f"<span style='color:{color}; font-weight:bold;'>{word}</span>"

    return re.sub(pattern, repl, text, flags=re.IGNORECASE)


def render_markdown_or_text(text: str, mode: str):
    if mode == "Markdown":
        st.markdown(text, unsafe_allow_html=True)
    else:
        st.text(text)


def ensure_session_defaults():
    ss = st.session_state
    ss.setdefault("language", "en")
    ss.setdefault("theme_mode", "Auto")
    ss.setdefault("painter_style", PAINTER_STYLES[0])
    ss.setdefault("wow_tokens", 0)
    ss.setdefault("wow_docs", 0)
    ss.setdefault("wow_runs", 0)
    ss.setdefault("chained_input", "")
    ss.setdefault("append_prompt", True)
    ss.setdefault("output_view_mode", "Markdown")
    ss.setdefault("openai_api_key", None)
    ss.setdefault("gemini_api_key", None)
    ss.setdefault("anthropic_api_key", None)
    ss.setdefault("grok_api_key", None)


def detect_env_key(secret_name: str) -> Optional[str]:
    # Prefer st.secrets, then environment
    try:
        val = st.secrets.get(secret_name)
        if val:
            return val
    except Exception:
        pass
    return os.getenv(secret_name)


def get_llm_client() -> LLMClient:
    return LLMClient(
        openai_key=st.session_state.get("openai_api_key"),
        gemini_key=st.session_state.get("gemini_api_key"),
        anthropic_key=st.session_state.get("anthropic_api_key"),
        grok_key=st.session_state.get("grok_api_key"),
    )


def painter_css(style: str) -> str:
    # Simple background gradients inspired by painters
    style = style.lower()
    if "van gogh" in style:
        bg = "linear-gradient(135deg, #1d3557 0%, #457b9d 40%, #f1fa8c 100%)"
    elif "picasso" in style:
        bg = "linear-gradient(135deg, #2b2d42 0%, #8d99ae 40%, #edf2f4 100%)"
    elif "monet" in style:
        bg = "linear-gradient(135deg, #a8dadc 0%, #f1faee 50%, #457b9d 100%)"
    elif "kahlo" in style:
        bg = "linear-gradient(135deg, #264653 0%, #e76f51 50%, #f4a261 100%)"
    elif "matisse" in style:
        bg = "linear-gradient(135deg, #0d1b2a 0%, #1b263b 40%, #e0e1dd 100%)"
    elif "dali" in style:
        bg = "linear-gradient(135deg, #f8edeb 0%, #ffb5a7 50%, #9d8189 100%)"
    elif "hokusai" in style:
        bg = "linear-gradient(135deg, #0f4c75 0%, #3282b8 40%, #bbe1fa 100%)"
    elif "rothko" in style:
        bg = "linear-gradient(180deg, #d62828 0%, #f77f00 50%, #fcbf49 100%)"
    elif "pollock" in style:
        bg = "radial-gradient(circle, #f1faee 0%, #a8dadc 40%, #1d3557 100%)"
    elif "chagall" in style:
        bg = "linear-gradient(135deg, #7209b7 0%, #f72585 40%, #4cc9f0 100%)"
    elif "basquiat" in style:
        bg = "linear-gradient(135deg, #000000 0%, #fca311 50%, #e5e5e5 100%)"
    else:
        bg = "linear-gradient(135deg, #ffffff 0%, #e9f5ff 50%, #cfe0ff 100%)"
    return f"""
        <style>
        .stApp {{
            background: {bg};
        }}
        .wow-card {{
            background-color: rgba(255,255,255,0.85);
            padding: 1rem;
            border-radius: 0.75rem;
            box-shadow: 0 0 10px rgba(0,0,0,0.08);
        }}
        </style>
    """


def render_dashboard():
    st.subheader(t("dashboard_intro"))
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(t("stats_tokens"), f"{st.session_state.get('wow_tokens', 0):,}")
    with col2:
        st.metric(t("stats_docs"), f"{st.session_state.get('wow_docs', 0):,}")
    with col3:
        st.metric(t("stats_runs"), f"{st.session_state.get('wow_runs', 0):,}")

    st.markdown("### " + t("wow_status"))
    col_a, col_b = st.columns([2, 3])
    with col_a:
        # Simple status badges
        api_ok = any(
            [
                st.session_state.get("openai_api_key"),
                st.session_state.get("gemini_api_key"),
                st.session_state.get("anthropic_api_key"),
                st.session_state.get("grok_api_key"),
            ]
        )
        docs = st.session_state.get("wow_docs", 0)
        runs = st.session_state.get("wow_runs", 0)

        st.markdown(
            f"""
            <div class='wow-card'>
            <ul>
              <li><b>API Ready:</b> {"✅" if api_ok else "⚠️"}</li>
              <li><b>Documents in Session:</b> {docs}</li>
              <li><b>Agent Runs this Session:</b> {runs}</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        # Simple interactive chart: tokens over time (mocked by session counters)
        tokens = st.session_state.get("wow_tokens", 0)
        runs = max(st.session_state.get("wow_runs", 1), 1)
        avg = tokens / runs if runs else 0
        fig = go.Figure(
            data=[
                go.Bar(
                    x=["Total Tokens", "Avg Tokens / Run"],
                    y=[tokens, avg],
                    marker_color=["#005596", "#1d3557"],
                )
            ]
        )
        fig.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)


def run_agent_for_text(
    llm: LLMClient,
    agent_key: str,
    agent_cfg: Dict[str, Any],
    base_prompt: str,
    document_text: str,
    user_extra_prompt: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    provider = MODEL_CONFIG.get(model_name, {}).get("provider", agent_cfg.get("model_provider", "google"))
    system_prompt = agent_cfg.get("system_prompt", "")
    full_user_prompt = base_prompt.strip()
    if user_extra_prompt.strip():
        full_user_prompt += "\n\nUser additional focus:\n" + user_extra_prompt.strip()
    full_user_prompt += "\n\n---\nSOURCE DOCUMENT:\n" + document_text[:200000]  # safety clip

    result = llm.generate(
        provider=provider,
        model=model_name,
        system_prompt=system_prompt,
        user_prompt=full_user_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    # Stats
    st.session_state["wow_tokens"] += int(result.get("tokens", 0))
    st.session_state["wow_runs"] += 1
    return result


def render_mood_gauge(score: int, label: str):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score,
            title={"text": f"{t('mood_confidence')} – {label}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#005596"},
                "steps": [
                    {"range": [0, 40], "color": "#e63946"},
                    {"range": [40, 70], "color": "#f1fa8c"},
                    {"range": [70, 100], "color": "#2a9d8f"},
                ],
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)


def run_note_magic(
    llm: LLMClient,
    magic: str,
    text: str,
    agent_cfg: Dict[str, Any],
    model_name: str,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    provider = MODEL_CONFIG.get(model_name, {}).get("provider", agent_cfg.get("model_provider", "openai"))
    system_prompt = agent_cfg.get("system_prompt", "You are a helpful assistant for regulatory notes.")
    if magic == "AI Transformation":
        user_prompt = (
            "Clean, structure, and professionalize the following regulatory notes. "
            "Use Markdown headings, bullet lists, and tables where appropriate. "
            "Correct grammar and preserve technical meaning.\n\n"
            f"NOTES:\n{text}"
        )
    elif magic == "Pattern Spotter":
        user_prompt = (
            "Analyze the following regulatory notes for recurring themes, systemic issues, "
            "and risk patterns. Output a Markdown section titled 'Patterns Detected' with "
            "sub-bullets explaining each pattern.\n\n"
            f"NOTES:\n{text}"
        )
    elif magic == "Narrative Weaver":
        user_prompt = (
            "You are drafting an executive-level narrative summary based on these notes. "
            "Write a cohesive story suitable for an FDA review memo, in Markdown.\n\n"
            f"NOTES:\n{text}"
        )
    elif magic == "Trend Forecaster":
        user_prompt = (
            "Based on these notes, forecast likely regulatory outcomes and potential "
            "Additional Information (AI) requests. Output a short 'Predicted Reviewer Concerns' "
            "section in Markdown. Include a clear disclaimer that this is AI prediction, not legal advice.\n\n"
            f"NOTES:\n{text}"
        )
    elif magic == "Socratic Mirror":
        user_prompt = (
            "Review the following regulatory strategy notes. Adopt the persona of a skeptical FDA Lead Reviewer. "
            "Do not rewrite the notes. Instead, output 3–10 challenging questions starting with "
            "'Have you considered...' or 'Where is the evidence for...'.\n\n"
            f"NOTES:\n{text}"
        )
    elif magic == "Mood Scape":
        user_prompt = (
            "Analyze the following regulatory notes and assess overall review confidence.\n"
            "Respond ONLY with a compact JSON object like:\n"
            '{"score": 0-100, "label": "string", "rationale": "short text"}\n\n'
            f"NOTES:\n{text}"
        )
    else:
        user_prompt = f"Organize the following notes into clear Markdown:\n\n{text}"

    result = llm.generate(
        provider=provider,
        model=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    st.session_state["wow_tokens"] += int(result.get("tokens", 0))
    st.session_state["wow_runs"] += 1
    return result


# --- Main UI -----------------------------------------------------------------


def main():
    ensure_session_defaults()

    # Basic page config
    st.set_page_config(
        page_title=t("app_title"),
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Top-level language, theme, style controls
    with st.sidebar:
        st.markdown(f"### {t('app_title')}")
        lang = st.radio(t("language"), ["en", "zh"], index=0 if st.session_state["language"] == "en" else 1)
        st.session_state["language"] = lang

        theme_mode = st.radio(
            t("theme"), [t("theme_auto"), t("theme_light"), t("theme_dark")],
            index={"Auto": 0, "Light": 1, "Dark": 2}.get(st.session_state["theme_mode"], 0),
        )
        # Map back to canonical
        if theme_mode == t("theme_auto"):
            st.session_state["theme_mode"] = "Auto"
        elif theme_mode == t("theme_light"):
            st.session_state["theme_mode"] = "Light"
        else:
            st.session_state["theme_mode"] = "Dark"

        painter = st.selectbox(t("style"), PAINTER_STYLES, index=PAINTER_STYLES.index(st.session_state["painter_style"]))
        st.session_state["painter_style"] = painter
        if st.button(t("style_jackpot")):
            st.session_state["painter_style"] = random.choice(PAINTER_STYLES)
            st.balloons()

        st.markdown(painter_css(st.session_state["painter_style"]), unsafe_allow_html=True)

        # API keys
        st.markdown(f"### {t('api_keys')}")
        for key_name, label in [
            ("OPENAI_API_KEY", t("openai_key")),
            ("GEMINI_API_KEY", t("gemini_key")),
            ("ANTHROPIC_API_KEY", t("anthropic_key")),
            ("GROK_API_KEY", t("grok_key")),
        ]:
            env_val = detect_env_key(key_name)
            if env_val and not st.session_state.get(key_name.lower()):
                # Store once; we don't display it
                st.session_state[key_name.lower().replace("api_key", "api_key")] = env_val

            short_key = key_name.split("_")[0].lower()
            current_val = st.session_state.get(f"{short_key.lower()}_api_key", None)
            if env_val:
                st.text_input(label, value="•••••••••• (from environment)", disabled=True)
                st.caption(t("api_loaded_env"))
                # Ensure session state key is set
                st.session_state[f"{short_key.lower()}_api_key"] = env_val
            else:
                new_val = st.text_input(
                    label,
                    type="password",
                    value=current_val or "",
                    placeholder=t("api_input_needed"),
                )
                if new_val:
                    st.session_state[f"{short_key.lower()}_api_key"] = new_val

        # Model and agent selection
        agents_cfg = load_agents()
        st.markdown("---")
        model_list = list(MODEL_CONFIG.keys())
        default_model = "gemini-2.5-flash" if "gemini-2.5-flash" in model_list else model_list[0]
        selected_model = st.selectbox(t("model_select"), model_list, index=model_list.index(default_model))
        st.session_state["global_model"] = selected_model

        agent_names = list(agents_cfg.keys())
        selected_agent_key = st.selectbox(t("agent_select"), agent_names, index=0)
        st.session_state["active_agent_key"] = selected_agent_key

        max_tokens = st.number_input(t("max_tokens"), min_value=512, max_value=200000, value=DEFAULT_MAX_TOKENS, step=512)
        st.session_state["global_max_tokens"] = int(max_tokens)
        temp = st.slider(t("temperature"), min_value=0.0, max_value=1.0, value=0.2, step=0.05)
        st.session_state["global_temperature"] = float(temp)

    # Main header
    st.title(t("app_title"))
    st.caption(f"Version {APP_VERSION} – Streamlit on Hugging Face Spaces")

    tabs = st.tabs(
        [
            t("dashboard"),
            t("summary_tool"),
            t("guidance_assistant"),
            t("note_keeper"),
            t("settings"),
        ]
    )

    # --- Dashboard ------------------------------------------------------------
    with tabs[0]:
        render_dashboard()

    # --- 510(k) Summary Tool --------------------------------------------------
    with tabs[1]:
        st.header(t("summary_tool"))
        col_left, col_right = st.columns([3, 2])

        with col_left:
            uploaded = st.file_uploader(
                t("file_uploader_510k"),
                type=["pdf", "txt", "md"],
                key="file_510k",
            )
            if uploaded is not None:
                st.info(t("pdf_warning"))
                text = extract_text_from_file(uploaded)
                st.session_state["wow_docs"] += 1
            else:
                text = ""

            extra_prompt = st.text_area(t("extra_prompt"), height=100, key="summary_extra_prompt")

        with col_right:
            agent_key = "fda_summarizer" if "fda_summarizer" in load_agents() else st.session_state["active_agent_key"]
            st.write(f"**Agent:** {agent_key}")
            local_model = st.selectbox(
                t("model_select"),
                list(MODEL_CONFIG.keys()),
                index=list(MODEL_CONFIG.keys()).index(st.session_state["global_model"]),
                key="summary_model_select",
            )
            local_max_tokens = st.number_input(
                t("max_tokens"),
                min_value=1024,
                max_value=200000,
                value=min(st.session_state["global_max_tokens"], 200000),
                step=512,
                key="summary_max_tokens",
            )
            local_temp = st.slider(
                t("temperature"),
                min_value=0.0,
                max_value=1.0,
                value=st.session_state["global_temperature"],
                step=0.05,
                key="summary_temp",
            )

            run_summary = st.button(t("generate_summary"), use_container_width=True)

        st.markdown("---")
        st.subheader(t("summary_output"))
        view_mode = st.radio(
            t("output_view_mode"),
            [t("view_markdown"), t("view_text")],
            index=0,
            horizontal=True,
            key="summary_view_mode",
        )
        output_mode = "Markdown" if view_mode == t("view_markdown") else "Text"

        if run_summary:
            if not text:
                st.warning(t("no_text"))
            elif not any(
                [
                    st.session_state.get("openai_api_key"),
                    st.session_state.get("gemini_api_key"),
                    st.session_state.get("anthropic_api_key"),
                    st.session_state.get("grok_api_key"),
                ]
            ):
                st.error(t("needs_api"))
            else:
                agents_cfg = load_agents()
                agent_cfg = agents_cfg.get(agent_key, {})
                llm = get_llm_client()
                base_prompt = (
                    "You are an expert FDA Regulatory Consultant. "
                    "I have provided the full text of a medical device 510(k) submission. "
                    "Your goal is to write a '510(k) Summary' document that fully complies with 21 CFR 807.92.\n\n"
                    "You MUST generate the following 5 tables using Markdown:\n"
                    "1. Submitter Info\n"
                    "2. Device Name & Classification\n"
                    "3. Predicate Device(s)\n"
                    "4. Device Description Comparison (Subject vs Predicate)\n"
                    "5. Summary of Non-Clinical Performance Data\n\n"
                    "After the tables, write a 500–2000 word narrative conclusion arguing why the device is "
                    "Substantially Equivalent (SE). Ensure professional, objective, regulatory tone."
                )
                with st.spinner(t("processing")):
                    result = run_agent_for_text(
                        llm=llm,
                        agent_key=agent_key,
                        agent_cfg=agent_cfg,
                        base_prompt=base_prompt,
                        document_text=text,
                        user_extra_prompt=extra_prompt,
                        model_name=local_model,
                        max_tokens=int(local_max_tokens),
                        temperature=float(local_temp),
                    )
                summary_text = result.get("text", "")
                if st.session_state.get("append_prompt", True):
                    summary_text += (
                        "\n\n---\n### Prompt Record\n"
                        "System Prompt:\n"
                        "```text\n"
                        f"{agent_cfg.get('system_prompt', '').strip()}\n"
                        "```\n\nUser Instruction:\n"
                        "```text\n"
                        f"{base_prompt}\n\n{extra_prompt}\n"
                        "```"
                    )
                st.session_state["summary_output"] = summary_text

        if "summary_output" in st.session_state:
            text_to_show = st.session_state["summary_output"]
            if output_mode == "Markdown":
                render_markdown_or_text(text_to_show, "Markdown")
            else:
                render_markdown_or_text(text_to_show, "Text")

            st.download_button(
                t("download_md"),
                data=text_to_show,
                file_name="510k_summary.md",
                mime="text/markdown",
            )
            st.download_button(
                t("download_txt"),
                data=text_to_show,
                file_name="510k_summary.txt",
                mime="text/plain",
            )

            if st.button(t("set_as_chained"), key="summary_set_chained"):
                st.session_state["chained_input"] = text_to_show
                st.success("Chained output stored for next agent.")

    # --- Guidance Assistant ----------------------------------------------------
    with tabs[2]:
        st.header(t("guidance_assistant"))
        col_left, col_right = st.columns([3, 2])

        with col_left:
            uploaded = st.file_uploader(
                t("file_uploader_guidance"),
                type=["pdf", "txt", "md"],
                key="file_guidance",
            )
            if uploaded is not None:
                st.info(t("pdf_warning"))
                text_g = extract_text_from_file(uploaded)
                st.session_state["wow_docs"] += 1
            else:
                text_g = ""

            device_type = st.text_input("Device type / context (optional)", "")

        with col_right:
            agent_key_g = "fda_guidance_expert" if "fda_guidance_expert" in load_agents() else st.session_state["active_agent_key"]
            st.write(f"**Agent:** {agent_key_g}")
            local_model_g = st.selectbox(
                t("model_select"),
                list(MODEL_CONFIG.keys()),
                index=list(MODEL_CONFIG.keys()).index(st.session_state["global_model"]),
                key="guidance_model_select",
            )
            local_max_tokens_g = st.number_input(
                t("max_tokens"),
                min_value=1024,
                max_value=200000,
                value=min(st.session_state["global_max_tokens"], 200000),
                step=512,
                key="guidance_max_tokens",
            )
            local_temp_g = st.slider(
                t("temperature"),
                min_value=0.0,
                max_value=1.0,
                value=st.session_state["global_temperature"],
                step=0.05,
                key="guidance_temp",
            )

            run_guidance = st.button(t("generate_guidance"), use_container_width=True)

        st.markdown("---")
        st.subheader(t("guidance_output"))
        view_mode_g = st.radio(
            t("output_view_mode"),
            [t("view_markdown"), t("view_text")],
            index=0,
            horizontal=True,
            key="guidance_view_mode",
        )
        output_mode_g = "Markdown" if view_mode_g == t("view_markdown") else "Text"

        if run_guidance:
            if not text_g:
                st.warning(t("no_text"))
            elif not any(
                [
                    st.session_state.get("openai_api_key"),
                    st.session_state.get("gemini_api_key"),
                    st.session_state.get("anthropic_api_key"),
                    st.session_state.get("grok_api_key"),
                ]
            ):
                st.error(t("needs_api"))
            else:
                agents_cfg = load_agents()
                agent_cfg_g = agents_cfg.get(agent_key_g, {})
                llm = get_llm_client()
                base_prompt_g = (
                    "You are a Senior FDA Reviewer. Analyze the provided FDA guidance document.\n\n"
                    "Your task is to:\n"
                    "1. Extract 'Refuse to Accept (RTA)' or administrative acceptance criteria.\n"
                    "2. Extract key scientific/technical review requirements and performance testing expectations.\n"
                    "3. Extract labeling expectations (warnings, contraindications, IFU elements).\n\n"
                    "Output in Markdown with:\n"
                    "- Checklist 1: Administrative Checklist\n"
                    "- Checklist 2: Scientific/Technical Review Checklist\n"
                    "- Checklist 3: Labeling Checklist\n"
                    "- A 2000–3000 word synthesis explaining how to apply this guidance to a review.\n"
                )
                if device_type.strip():
                    base_prompt_g += f"\nDevice context: {device_type.strip()}.\n"

                with st.spinner(t("processing")):
                    result_g = run_agent_for_text(
                        llm=llm,
                        agent_key=agent_key_g,
                        agent_cfg=agent_cfg_g,
                        base_prompt=base_prompt_g,
                        document_text=text_g,
                        user_extra_prompt="",
                        model_name=local_model_g,
                        max_tokens=int(local_max_tokens_g),
                        temperature=float(local_temp_g),
                    )
                out_g = result_g.get("text", "")
                if st.session_state.get("append_prompt", True):
                    out_g += (
                        "\n\n---\n### Prompt Record\n"
                        "System Prompt:\n```text\n"
                        f"{agent_cfg_g.get('system_prompt', '').strip()}\n"
                        "```\n\nUser Instruction:\n```text\n"
                        f"{base_prompt_g}\n"
                        "```"
                    )
                st.session_state["guidance_output"] = out_g

        if "guidance_output" in st.session_state:
            text_to_show = st.session_state["guidance_output"]
            render_markdown_or_text(text_to_show, output_mode_g)
            st.download_button(
                t("download_md"),
                data=text_to_show,
                file_name="guidance_checklists.md",
                mime="text/markdown",
            )
            st.download_button(
                t("download_txt"),
                data=text_to_show,
                file_name="guidance_checklists.txt",
                mime="text/plain",
            )
            if st.button(t("set_as_chained"), key="guidance_set_chained"):
                st.session_state["chained_input"] = text_to_show
                st.success("Chained output stored for next agent.")

    # --- Note Keeper & AI Magics ----------------------------------------------
    with tabs[3]:
        st.header(t("note_keeper"))
        col_left, col_right = st.columns([3, 2])

        with col_left:
            default_note = st.session_state.get("chained_input", "")
            raw_notes = st.text_area(t("note_input"), value=default_note, height=260, key="note_input_area")

            magic = st.selectbox(
                t("magic_select"),
                [
                    t("magic_ai_transform"),
                    t("magic_keywords"),
                    t("magic_pattern"),
                    t("magic_narrative"),
                    t("magic_trend"),
                    t("magic_socratic"),
                    t("magic_mood"),
                ],
                index=0,
            )

            keywords = ""
            color = "#d62828"
            if magic == t("magic_keywords"):
                keywords = st.text_input(t("keywords_input"), "")
                color = st.color_picker(t("keywords_color"), "#d62828")

            run_magic_btn = st.button(t("run_magic"))

        with col_right:
            agent_key_n = "note_keeper" if "note_keeper" in load_agents() else st.session_state["active_agent_key"]
            st.write(f"**Agent:** {agent_key_n}")
            local_model_n = st.selectbox(
                t("model_select"),
                list(MODEL_CONFIG.keys()),
                index=list(MODEL_CONFIG.keys()).index(st.session_state["global_model"]),
                key="note_model_select",
            )
            local_max_tokens_n = st.number_input(
                t("max_tokens"),
                min_value=512,
                max_value=64000,
                value=min(st.session_state["global_max_tokens"], 64000),
                step=512,
                key="note_max_tokens",
            )
            local_temp_n = st.slider(
                t("temperature"),
                min_value=0.0,
                max_value=1.0,
                value=st.session_state["global_temperature"],
                step=0.05,
                key="note_temp",
            )

        st.markdown("---")
        st.subheader(t("note_output"))
        view_mode_n = st.radio(
            t("output_view_mode"),
            [t("view_markdown"), t("view_text")],
            index=0,
            horizontal=True,
            key="note_view_mode",
        )
        output_mode_n = "Markdown" if view_mode_n == t("view_markdown") else "Text"

        if run_magic_btn:
            if not raw_notes.strip():
                st.warning(t("no_text"))
            elif magic == t("magic_keywords"):
                if not keywords.strip():
                    st.warning(t("no_text"))
                else:
                    # Local keyword highlighter
                    kws = [k.strip() for k in keywords.split(",") if k.strip()]
                    colored = apply_keywords_colorizer(raw_notes, kws, color)
                    st.session_state["note_output"] = colored
            elif not any(
                [
                    st.session_state.get("openai_api_key"),
                    st.session_state.get("gemini_api_key"),
                    st.session_state.get("anthropic_api_key"),
                    st.session_state.get("grok_api_key"),
                ]
            ):
                st.error(t("needs_api"))
            else:
                agents_cfg = load_agents()
                agent_cfg_n = agents_cfg.get(agent_key_n, {})
                llm = get_llm_client()
                # Map label to internal magic key
                magic_map = {
                    t("magic_ai_transform"): "AI Transformation",
                    t("magic_pattern"): "Pattern Spotter",
                    t("magic_narrative"): "Narrative Weaver",
                    t("magic_trend"): "Trend Forecaster",
                    t("magic_socratic"): "Socratic Mirror",
                    t("magic_mood"): "Mood Scape",
                }
                internal_magic = magic_map.get(magic, "AI Transformation")

                with st.spinner(t("processing")):
                    result_n = run_note_magic(
                        llm=llm,
                        magic=internal_magic,
                        text=raw_notes,
                        agent_cfg=agent_cfg_n,
                        model_name=local_model_n,
                        max_tokens=int(local_max_tokens_n),
                        temperature=float(local_temp_n),
                    )
                out_n = result_n.get("text", "")

                # If Mood Scape, parse JSON and draw gauge, but also store rationale
                if internal_magic == "Mood Scape":
                    try:
                        data = json.loads(out_n.strip())
                        score = int(data.get("score", 50))
                        label = str(data.get("label", "Neutral"))
                        rationale = str(data.get("rationale", ""))
                    except Exception:
                        score = 50
                        label = "Uncertain"
                        rationale = out_n[:500]
                    st.session_state["mood_score"] = score
                    st.session_state["mood_label"] = label
                    st.session_state["mood_rationale"] = rationale
                    # Also create a Markdown block
                    md = f"**{t('mood_confidence')}: {score}/100 – {label}**\n\n{rationale}"
                    out_n = md

                if st.session_state.get("append_prompt", True):
                    out_n += (
                        "\n\n---\n### Prompt Record\n"
                        "System Prompt:\n```text\n"
                        f"{agent_cfg_n.get('system_prompt', '').strip()}\n"
                        "```\n"
                    )
                st.session_state["note_output"] = out_n

        if "note_output" in st.session_state:
            note_out = st.session_state["note_output"]
            render_markdown_or_text(note_out, output_mode_n)
            st.download_button(
                t("download_md"),
                data=note_out,
                file_name="note_transformed.md",
                mime="text/markdown",
            )
            st.download_button(
                t("download_txt"),
                data=note_out,
                file_name="note_transformed.txt",
                mime="text/plain",
            )
            if st.button(t("set_as_chained"), key="note_set_chained"):
                st.session_state["chained_input"] = note_out
                st.success("Chained output stored for next agent.")

        # Mood gauge, if available
        if "mood_score" in st.session_state:
            render_mood_gauge(
                st.session_state.get("mood_score", 50),
                st.session_state.get("mood_label", "Neutral"),
            )
            st.markdown("**Rationale:** " + st.session_state.get("mood_rationale", ""))

    # --- Settings / Agentsmith -------------------------------------------------
    with tabs[4]:
        st.header(t("settings"))

        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.subheader(t("yaml_editor"))
            raw_yaml = st.text_area(
                "agents.yaml",
                value=load_agents_raw_text(),
                height=360,
                key="agents_yaml_editor",
            )
            if st.button(t("yaml_validate")):
                try:
                    parsed = yaml.safe_load(raw_yaml) or {}
                    st.session_state["agents_config"] = parsed
                    st.success("agents.yaml loaded into session.")
                except Exception as e:
                    st.error(f"YAML error: {e}")

            st.download_button(
                t("yaml_download"),
                data=raw_yaml,
                file_name="agents.yaml",
                mime="text/yaml",
            )

            uploaded_yaml = st.file_uploader(t("yaml_upload"), type=["yaml", "yml"])
            if uploaded_yaml is not None:
                try:
                    text_u = uploaded_yaml.read().decode("utf-8")
                    parsed_u = yaml.safe_load(text_u) or {}
                    st.session_state["agents_config"] = parsed_u
                    st.session_state["agents_yaml_editor"] = text_u
                    st.success("Uploaded agents.yaml loaded into session.")
                except Exception as e:
                    st.error(f"Failed to load uploaded YAML: {e}")

        with col_b:
            st.subheader("SKILL.md")
            st.caption(t("skills_source"))
            st.text_area("SKILL.md", value=load_skill_md(), height=360, disabled=True)

            st.markdown("---")
            append_prompt = st.checkbox(
                t("append_prompt"),
                value=st.session_state.get("append_prompt", True),
            )
            st.session_state["append_prompt"] = append_prompt

    # Footer small
    st.markdown(
        "<hr/><small>Session is stateless; download your work to preserve it. "
        "Use enterprise / zero-data-retention API endpoints for PHI if required.</small>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
