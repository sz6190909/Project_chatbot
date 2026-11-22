"""
AI Chatbot powered by Groq API and Streamlit.

A professional, dark-themed, fully functional chatbot interface with a
configurable sidebar (API key management, model selection, system roles,
generation parameters, streaming/history toggles, export utilities) and a
main chat surface built on Streamlit's native chat components.

Run:
    pip install -r requirements.txt
    streamlit run chatbot.py
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st
from groq import Groq, APIError, APIConnectionError, RateLimitError, AuthenticationError

# --------------------------------------------------------------------------- #
# Logging configuration — WARNING level only; API keys are NEVER logged.      #
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("chatbot")


# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #
APP_TITLE: str = "Groq AI Chatbot"
APP_ICON: str = "💬"
APP_VERSION: str = "1.0.0"

ENV_API_KEY_VAR: str = "GROQ_API_KEY"

DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_MAX_TOKENS: int = 1024
DEFAULT_TOP_P: float = 1.0
DEFAULT_FREQUENCY_PENALTY: float = 0.0
DEFAULT_PRESENCE_PENALTY: float = 0.0
DEFAULT_STREAMING: bool = True
DEFAULT_HISTORY_ENABLED: bool = True

MIN_TEMPERATURE: float = 0.0
MAX_TEMPERATURE: float = 2.0
MIN_MAX_TOKENS: int = 1
MAX_MAX_TOKENS: int = 8192
MIN_TOP_P: float = 0.0
MAX_TOP_P: float = 1.0
MIN_PENALTY: float = -2.0
MAX_PENALTY: float = 2.0


# --------------------------------------------------------------------------- #
# Fallback Groq models (used before API validation / if fetch fails).          #
# The authoritative list is fetched dynamically via client.models.list()      #
# after the API key is validated, so the dropdown only shows models the key    #
# can actually access.                                                         #
# --------------------------------------------------------------------------- #
FALLBACK_MODELS: List[Dict[str, str]] = [
    {"id": "openai/gpt-oss-20b",         "label": "GPT OSS 20B"},
    {"id": "openai/gpt-oss-120b",        "label": "GPT OSS 120B"},
    {"id": "qwen/qwen3.8-27b",           "label": "Qwen 3.8 27B"},
    {"id": "llama-3.3-70b-versatile",    "label": "Llama 3.3 70B Versatile"},
    {"id": "llama-3.1-8b-instant",       "label": "Llama 3.1 8B Instant"},
]
DEFAULT_MODEL_ID: str = "openai/gpt-oss-20b"
CUSTOM_MODEL_LABEL: str = "Custom Model ID"


# --------------------------------------------------------------------------- #
# Predefined system role templates.                                            #
# --------------------------------------------------------------------------- #
SYSTEM_ROLES: Dict[str, str] = {
    "General Assistant": (
        "You are a helpful, knowledgeable, and friendly general-purpose assistant. "
        "Answer questions clearly and accurately across any topic. "
        "When information is uncertain, say so honestly rather than guessing."
    ),
    "Python Tutor": (
        "You are an expert Python programming tutor. Explain concepts with clear, "
        "well-commented code examples. Adapt explanations to the learner's level, "
        "point out common pitfalls, and suggest best practices. "
        "Prefer modern Python (3.10+) syntax and idiomatic constructs."
    ),
    "Machine Learning Expert": (
        "You are a senior machine learning engineer. Provide rigorous, practical "
        "guidance on model selection, training, evaluation, and deployment. "
        "Use precise terminology, reference relevant algorithms and metrics, "
        "and include concise code snippets when helpful. "
        "Always consider data quality, overfitting, and reproducibility."
    ),
    "AI & Robotics Trainer": (
        "You are an AI and robotics trainer. Explain perception, planning, control, "
        "and learning for robotic systems. Cover sensors, actuators, ROS/ROS2, "
        "reinforcement learning, and sim-to-real transfer. "
        "Provide actionable guidance for building and training intelligent robots."
    ),
    "Data Analyst": (
        "You are an expert data analyst. Help with data cleaning, exploration, "
        "visualization, statistical inference, and storytelling with data. "
        "Recommend appropriate tools (pandas, numpy, SQL, matplotlib, plotly) and "
        "walk through reasoning step by step. Highlight assumptions and caveats."
    ),
    "Custom": "",
}
ROLE_LABELS: List[str] = list(SYSTEM_ROLES.keys())


# --------------------------------------------------------------------------- #
# Universal language directive — appended to every system prompt so the        #
# assistant always replies in the language the user writes in, across all     #
# supported languages (English, Arabic, French, Spanish, Chinese, etc.).      #
# --------------------------------------------------------------------------- #
LANGUAGE_DIRECTIVE: str = (
    "CRITICAL LANGUAGE RULE (highest priority): Respond in the EXACT same language "
    "the user writes in. Detect the language from the user's LATEST message and "
    "reply ONLY in that language. If the user writes English, reply in English. "
    "Never reply in Arabic unless the user wrote in Arabic. Never reply in any "
    "language the user did not use. For very short inputs like 'hi' or 'hello', "
    "treat them as English. Keep code, commands, and technical terms unchanged."
)

LANGUAGE_REMINDER: str = (
    "FINAL REMINDER — overrides any prior assistant messages: Look ONLY at the "
    "user's most recent message. Detect both its SCRIPT and LANGUAGE, then reply "
    "in that exact language and script.\n"
    "- Latin script with English words -> English.\n"
    "- Latin script with Urdu/Hindi words (e.g. kya, hai, mumkin, kaise, acha) -> "
    "Roman Urdu (Latin script), NOT Arabic.\n"
    "- Arabic script -> Arabic. Devanagari -> Hindi. Cyrillic -> Russian/etc.\n"
    "- NEVER reply in Arabic unless the user's latest message is in Arabic script.\n"
    "- Ignore the language of any previous assistant messages in history."
)


# --------------------------------------------------------------------------- #
# Session-state initialization                                                 #
# --------------------------------------------------------------------------- #
def init_session_state() -> None:
    """Initialize all session-state keys with sensible defaults."""
    defaults: Dict[str, Any] = {
        "messages": [],
        "api_key": "",
        "api_validated": False,
        "available_models": [],
        "theme_mode": "dark",
        "selected_model_label": FALLBACK_MODELS[0]["label"],
        "custom_model_id": "",
        "selected_role": ROLE_LABELS[0],
        "custom_system_prompt": "",
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "top_p": DEFAULT_TOP_P,
        "frequency_penalty": DEFAULT_FREQUENCY_PENALTY,
        "presence_penalty": DEFAULT_PRESENCE_PENALTY,
        "streaming_enabled": DEFAULT_STREAMING,
        "history_enabled": DEFAULT_HISTORY_ENABLED,
        "validation_status": None,
        "chat_counter": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# --------------------------------------------------------------------------- #
# API key handling                                                             #
# --------------------------------------------------------------------------- #
def resolve_api_key() -> str:
    """Return the active API key from session state, falling back to env var."""
    if st.session_state.api_key:
        return st.session_state.api_key.strip()
    env_key = os.getenv(ENV_API_KEY_VAR, "").strip()
    if env_key:
        return env_key
    return ""


def build_client(api_key: str) -> Optional[Groq]:
    """Construct a Groq client. Returns None if the key is empty."""
    if not api_key:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception as exc:
        logger.warning("Failed to construct Groq client: %s", type(exc).__name__)
        return None


def validate_api_connection(client: Groq) -> tuple[bool, str]:
    """
    Validate the API connection with a lightweight models.list() call.

    Returns (success, message). The API key is never included in the message.
    On success, populates st.session_state.available_models with the IDs the
    key can actually access.
    """
    try:
        models_page = client.models.list()
        model_ids: List[str] = []
        data = getattr(models_page, "data", None)
        if data is None and isinstance(models_page, dict):
            data = models_page.get("data", [])
        if data:
            for m in data:
                mid = getattr(m, "id", None) or (m.get("id") if isinstance(m, dict) else None)
                if mid:
                    model_ids.append(mid)
        model_ids = sorted(set(model_ids))
        st.session_state.available_models = model_ids
        if model_ids:
            return True, f"Connection successful. {len(model_ids)} model(s) available."
        return True, "Connection successful. API key is valid."
    except AuthenticationError:
        return False, "Authentication failed: invalid API key."
    except APIConnectionError:
        return False, "Connection error: unable to reach Groq servers."
    except RateLimitError:
        return False, "Rate limit reached. Please wait and try again."
    except APIError as exc:
        return False, f"API error ({exc.status_code}): {exc.message}"
    except Exception as exc:
        return False, f"Unexpected error: {type(exc).__name__}"


# --------------------------------------------------------------------------- #
# Model resolution                                                             #
# --------------------------------------------------------------------------- #
def get_model_options() -> List[str]:
    """Return the dropdown labels: dynamic model IDs (if fetched) plus Custom."""
    dynamic = st.session_state.get("available_models", [])
    if dynamic:
        return list(dynamic) + [CUSTOM_MODEL_LABEL]
    return [m["label"] for m in FALLBACK_MODELS] + [CUSTOM_MODEL_LABEL]


def resolve_model_id() -> str:
    """Resolve the final model ID from the dropdown or custom entry."""
    selected = st.session_state.selected_model_label
    if selected == CUSTOM_MODEL_LABEL:
        custom = st.session_state.custom_model_id.strip()
        return custom if custom else DEFAULT_MODEL_ID
    dynamic = st.session_state.get("available_models", [])
    if dynamic and selected in dynamic:
        return selected
    for model in FALLBACK_MODELS:
        if model["label"] == selected:
            return model["id"]
    return DEFAULT_MODEL_ID


def resolve_system_prompt() -> str:
    """Resolve the active system prompt with the language rule prepended (top priority)."""
    role = st.session_state.selected_role
    if role == "Custom":
        base = st.session_state.custom_system_prompt.strip()
    else:
        base = SYSTEM_ROLES.get(role, SYSTEM_ROLES[ROLE_LABELS[0]])
    if base:
        return LANGUAGE_DIRECTIVE + "\n\n" + base
    return LANGUAGE_DIRECTIVE


# --------------------------------------------------------------------------- #
# Chat history helpers                                                         #
# --------------------------------------------------------------------------- #
def build_api_messages(system_prompt: str) -> List[Dict[str, str]]:
    """Assemble the message list sent to the API (system + history + reminder)."""
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for msg in st.session_state.messages:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "system", "content": LANGUAGE_REMINDER})
    return messages


def add_message(role: str, content: str) -> None:
    """Append a message to session-state history (if history is enabled)."""
    if not st.session_state.history_enabled:
        return
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })


# --------------------------------------------------------------------------- #
# Response generation                                                          #
# --------------------------------------------------------------------------- #
def generate_response_streaming(client: Groq, payload: Dict[str, Any]) -> str:
    """Stream the response token-by-token, returning the full text."""
    stream = client.chat.completions.create(stream=True, **payload)
    full_text: List[str] = []
    response_placeholder = st.empty()
    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            full_text.append(delta.content)
            response_placeholder.markdown("".join(full_text))
    response_placeholder.markdown("".join(full_text))
    return "".join(full_text)


def generate_response_blocking(client: Groq, payload: Dict[str, Any]) -> str:
    """Request a full (non-streamed) response and display it."""
    completion = client.chat.completions.create(stream=False, **payload)
    content = completion.choices[0].message.content or ""
    st.markdown(content)
    return content


def send_user_message(user_input: str) -> None:
    """Handle a user message: display it, call the API, show the reply."""
    api_key = resolve_api_key()
    if not api_key:
        st.error("Please enter your Groq API key in the sidebar to begin.")
        return

    client = build_client(api_key)
    if client is None:
        st.error("Could not initialize the Groq client. Check your API key.")
        return

    add_message("user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    model_id = resolve_model_id()
    system_prompt = resolve_system_prompt()
    api_messages = build_api_messages(system_prompt)

    payload: Dict[str, Any] = {
        "model": model_id,
        "messages": api_messages,
        "temperature": st.session_state.temperature,
        "max_tokens": st.session_state.max_tokens,
        "top_p": st.session_state.top_p,
        "frequency_penalty": st.session_state.frequency_penalty,
        "presence_penalty": st.session_state.presence_penalty,
    }

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if st.session_state.streaming_enabled:
                    reply = generate_response_streaming(client, payload)
                else:
                    reply = generate_response_blocking(client, payload)
            except AuthenticationError:
                reply = ""
                st.error("Authentication failed: invalid API key.")
            except RateLimitError:
                reply = ""
                st.error("Rate limit reached. Please wait and try again.")
            except APIConnectionError:
                reply = ""
                st.error("Connection error: unable to reach Groq servers.")
            except APIError as exc:
                reply = ""
                st.error(f"API error ({exc.status_code}): {exc.message}")
            except Exception as exc:
                reply = ""
                st.error(f"Unexpected error: {type(exc).__name__}: {exc}")

    if reply:
        add_message("assistant", reply)


# --------------------------------------------------------------------------- #
# Export utilities                                                             #
# --------------------------------------------------------------------------- #
def export_as_txt() -> str:
    """Render the conversation as plain text."""
    lines: List[str] = [
        f"Groq AI Chatbot - Conversation Export",
        f"Exported: {datetime.now().isoformat(timespec='seconds')}",
        f"Model: {resolve_model_id()}",
        f"System role: {st.session_state.selected_role}",
        "-" * 60,
        "",
    ]
    for msg in st.session_state.messages:
        speaker = "You" if msg["role"] == "user" else "Assistant"
        ts = msg.get("timestamp", "")
        lines.append(f"[{ts}] {speaker}:")
        lines.append(msg["content"])
        lines.append("")
    return "\n".join(lines)


def export_as_json() -> str:
    """Render the conversation as a JSON string."""
    payload = {
        "app": APP_TITLE,
        "version": APP_VERSION,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "model": resolve_model_id(),
        "system_role": st.session_state.selected_role,
        "system_prompt": resolve_system_prompt(),
        "parameters": {
            "temperature": st.session_state.temperature,
            "max_tokens": st.session_state.max_tokens,
            "top_p": st.session_state.top_p,
            "frequency_penalty": st.session_state.frequency_penalty,
            "presence_penalty": st.session_state.presence_penalty,
        },
        "messages": st.session_state.messages,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Sidebar configuration panel                                                  #
# --------------------------------------------------------------------------- #
def render_sidebar() -> None:
    """Render the full left-sidebar configuration panel."""
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding:6px 0 4px;">
                <div style="font-size:20px; font-weight:700; color:#e6edf3;">
                    {APP_ICON}&nbsp; {APP_TITLE}
                </div>
                <div style="font-size:12px; color:#8b98a9; margin-top:4px;">
                    v{APP_VERSION} · Powered by Groq
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # --- API key management ------------------------------------------- #
        st.subheader("API Configuration")

        env_hint = ""
        if os.getenv(ENV_API_KEY_VAR):
            env_hint = " (env var detected)"
        st.markdown(f"**Groq API Key**{env_hint}")
        st.caption("Your key is stored only in this browser session and is never logged.")

        api_key_input = st.text_input(
            label="Groq API Key",
            value=st.session_state.api_key,
            type="password",
            placeholder="gsk_...",
            key="api_key_input_widget",
            help="Get your key at https://console.groq.com/keys",
            label_visibility="collapsed",
        )
        if api_key_input != st.session_state.api_key:
            st.session_state.api_key = api_key_input.strip()
            st.session_state.api_validated = False
            st.session_state.validation_status = None
            st.session_state.available_models = []

        col_validate, col_clear = st.columns(2)
        with col_validate:
            if st.button("Validate", use_container_width=True):
                key = resolve_api_key()
                if not key:
                    st.session_state.validation_status = (False, "Enter an API key first.")
                else:
                    client = build_client(key)
                    if client is None:
                        st.session_state.validation_status = (False, "Invalid key format.")
                    else:
                        st.session_state.validation_status = validate_api_connection(client)
                st.session_state.api_validated = st.session_state.validation_status[0]
        with col_clear:
            if st.button("Clear Key", use_container_width=True):
                st.session_state.api_key = ""
                st.session_state.api_validated = False
                st.session_state.validation_status = None
                st.session_state.available_models = []
                st.rerun()

        if st.session_state.validation_status is not None:
            ok, msg = st.session_state.validation_status
            if ok:
                st.success(msg)
            else:
                st.error(msg)
        elif st.session_state.api_validated:
            st.info("API key validated.")

        st.divider()

        # --- Model selection ---------------------------------------------- #
        st.subheader("Model Selection")
        model_options = get_model_options()
        current_label = st.session_state.selected_model_label
        if current_label not in model_options:
            current_label = model_options[0]
            st.session_state.selected_model_label = current_label
        selected_label = st.selectbox(
            label="Model",
            options=model_options,
            index=model_options.index(current_label),
            key="model_select_widget",
        )
        st.session_state.selected_model_label = selected_label

        if selected_label == CUSTOM_MODEL_LABEL:
            custom_id = st.text_input(
                label="Custom Model ID",
                value=st.session_state.custom_model_id,
                placeholder="e.g. openai/gpt-oss-20b",
                key="custom_model_widget",
            )
            st.session_state.custom_model_id = custom_id.strip()

        if st.session_state.get("available_models"):
            st.caption(f"Active model: `{resolve_model_id()}`  (fetched from API)")
        else:
            st.caption(f"Active model: `{resolve_model_id()}`  (validate to refresh)")

        st.divider()

        # --- System role & prompt ----------------------------------------- #
        st.subheader("System Role")
        selected_role = st.selectbox(
            label="Role template",
            options=ROLE_LABELS,
            index=ROLE_LABELS.index(st.session_state.selected_role),
            key="role_select_widget",
        )
        st.session_state.selected_role = selected_role

        if selected_role == "Custom":
            custom_prompt = st.text_area(
                label="Custom system prompt",
                value=st.session_state.custom_system_prompt,
                height=160,
                placeholder="Describe the assistant's behavior, tone, and constraints...",
                key="custom_prompt_widget",
            )
            st.session_state.custom_system_prompt = custom_prompt
        else:
            st.caption(SYSTEM_ROLES[selected_role])

        st.divider()

        # --- Generation parameters ---------------------------------------- #
        st.subheader("Generation Parameters")

        st.session_state.temperature = st.slider(
            label="Temperature",
            min_value=MIN_TEMPERATURE,
            max_value=MAX_TEMPERATURE,
            value=float(st.session_state.temperature),
            step=0.05,
            help="Controls randomness. Lower is deterministic; higher is creative.",
        )
        st.session_state.max_tokens = st.slider(
            label="Max tokens",
            min_value=MIN_MAX_TOKENS,
            max_value=MAX_MAX_TOKENS,
            value=int(st.session_state.max_tokens),
            step=64,
            help="Maximum number of tokens in the response.",
        )
        st.session_state.top_p = st.slider(
            label="Top-P",
            min_value=MIN_TOP_P,
            max_value=MAX_TOP_P,
            value=float(st.session_state.top_p),
            step=0.01,
            help="Nucleus sampling probability mass.",
        )
        st.session_state.frequency_penalty = st.slider(
            label="Frequency penalty",
            min_value=MIN_PENALTY,
            max_value=MAX_PENALTY,
            value=float(st.session_state.frequency_penalty),
            step=0.1,
            help="Penalize repeated tokens (positive reduces repetition).",
        )
        st.session_state.presence_penalty = st.slider(
            label="Presence penalty",
            min_value=MIN_PENALTY,
            max_value=MAX_PENALTY,
            value=float(st.session_state.presence_penalty),
            step=0.1,
            help="Penalize tokens that have already appeared.",
        )

        st.divider()

        # --- Toggles ------------------------------------------------------ #
        st.subheader("Options")
        st.session_state.streaming_enabled = st.toggle(
            label="Streaming responses",
            value=bool(st.session_state.streaming_enabled),
            help="Stream tokens as they arrive.",
        )
        st.session_state.history_enabled = st.toggle(
            label="Conversation history",
            value=bool(st.session_state.history_enabled),
            help="Maintain context across messages in this session.",
        )

        st.divider()

        # --- Chat actions ------------------------------------------------- #
        st.subheader("Chat Actions")
        col_new, col_clear_chat = st.columns(2)
        with col_new:
            if st.button("New Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.chat_counter += 1
                st.rerun()
        with col_clear_chat:
            if st.button("Clear History", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        if st.button("Reset Settings", use_container_width=True):
            reset_settings()
            st.rerun()

        st.divider()

        # --- Export ------------------------------------------------------- #
        st.subheader("Export Conversation")
        col_txt, col_json = st.columns(2)
        with col_txt:
            if st.button("Export TXT", use_container_width=True):
                if st.session_state.messages:
                    st.download_button(
                        label="Download .txt",
                        data=export_as_txt(),
                        file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
                else:
                    st.warning("No messages to export.")
        with col_json:
            if st.button("Export JSON", use_container_width=True):
                if st.session_state.messages:
                    st.download_button(
                        label="Download .json",
                        data=export_as_json(),
                        file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                else:
                    st.warning("No messages to export.")


def reset_settings() -> None:
    """Reset all configurable settings to their defaults (keeps API key)."""
    st.session_state.selected_model_label = FALLBACK_MODELS[0]["label"]
    st.session_state.custom_model_id = ""
    st.session_state.selected_role = ROLE_LABELS[0]
    st.session_state.custom_system_prompt = ""
    st.session_state.temperature = DEFAULT_TEMPERATURE
    st.session_state.max_tokens = DEFAULT_MAX_TOKENS
    st.session_state.top_p = DEFAULT_TOP_P
    st.session_state.frequency_penalty = DEFAULT_FREQUENCY_PENALTY
    st.session_state.presence_penalty = DEFAULT_PRESENCE_PENALTY
    st.session_state.streaming_enabled = DEFAULT_STREAMING
    st.session_state.history_enabled = DEFAULT_HISTORY_ENABLED


# --------------------------------------------------------------------------- #
# Main chat surface                                                            #
# --------------------------------------------------------------------------- #
def render_chat_surface() -> None:
    """Render the main chat area: hero header, history, and input box."""
    active_model = resolve_model_id()
    active_role = st.session_state.selected_role
    is_connected = st.session_state.api_validated
    status_pill = (
        f"<span class='pill pill-ok'>● Connected</span>"
        if is_connected else
        f"<span class='pill pill-no'>○ Not validated</span>"
    )
    model_pill = f"<span class='pill pill-id'>{active_model}</span>"
    role_pill = f"<span class='pill pill-id'>{active_role}</span>"

    st.markdown(
        f"""
        <div class="hero">
            <h1>{APP_ICON}&nbsp; {APP_TITLE}</h1>
            <div class="sub">Powered by Groq · ultra-fast LLM inference · v{APP_VERSION}</div>
            <div class="meta">
                {status_pill} &nbsp; Model {model_pill} &nbsp; Role {role_pill}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not resolve_api_key():
        st.markdown(
            """
            <div class="hint">
                🔑 Enter your <b>Groq API key</b> in the sidebar to start chatting.<br>
                Get a free key at
                <a href="https://console.groq.com/keys" target="_blank">console.groq.com/keys</a>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _render_welcome_cards()
        return

    if not st.session_state.messages:
        _render_welcome_cards()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Type your message in any language…")
    if user_input and user_input.strip():
        send_user_message(user_input.strip())


def _render_welcome_cards() -> None:
    """Render feature hint cards shown on the empty/welcome state."""
    cards = [
        ("🌍", "Multilingual", "Chat in any language — English, Arabic, French, "
         "Spanish, Chinese, Urdu, Hindi, and more. The assistant replies in your language."),
        ("⚡", "Streaming", "Responses stream token-by-token for instant feedback. "
         "Toggle streaming on/off in the sidebar options."),
        ("🧠", "Roles & Prompts", "Switch between General Assistant, Python Tutor, "
         "ML Expert, Robotics Trainer, Data Analyst, or a custom system prompt."),
        ("💾", "Export", "Save your conversation as TXT or JSON with full metadata "
         "and parameters for reproducibility."),
    ]
    cells = "".join(
        f"<div class='welcome-card'>"
        f"<div class='ico'>{ico}</div>"
        f"<div class='ttl'>{ttl}</div>"
        f"<div class='desc'>{desc}</div>"
        f"</div>"
        for ico, ttl, desc in cards
    )
    st.markdown(
        f"<div class='welcome-grid'>{cells}</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Page configuration & entrypoint                                              #
# --------------------------------------------------------------------------- #
def configure_page() -> None:
    """Configure Streamlit page metadata and a readable dark/light-themed UI."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    is_dark = st.session_state.get("theme_mode", "dark") == "dark"

    if is_dark:
        theme = {
            "bg": "#0b0f17", "surface": "#121826", "surface2": "#1a2234",
            "border": "#2a3650", "text": "#f0f6fc", "muted": "#aebac9",
            "accent": "#2dd4bf", "accent2": "#6366f1",
            "danger": "#f87171", "ok": "#34d399",
            "app_grad_a": "#0b0f17", "app_grad_b": "#0d1320",
            "hero_grad_a": "#1a2234", "hero_grad_b": "#121826", "hero_grad_c": "#0b0f17",
            "scroll_thumb": "#334155", "scroll_thumb_h": "#475569", "scroll_track": "#0b0f17",
            "btn_hover_bg": "#0f1726",
        }
    else:
        theme = {
            "bg": "#f6f8fa", "surface": "#ffffff", "surface2": "#eef1f5",
            "border": "#d0d7de", "text": "#1f2328", "muted": "#57606a",
            "accent": "#0d9488", "accent2": "#4f46e5",
            "danger": "#cf222e", "ok": "#1a7f37",
            "app_grad_a": "#f6f8fa", "app_grad_b": "#eaeef2",
            "hero_grad_a": "#ffffff", "hero_grad_b": "#f0f4f8", "hero_grad_c": "#e6ebf1",
            "scroll_thumb": "#9aa4b2", "scroll_thumb_h": "#6e7781", "scroll_track": "#eaeef2",
            "btn_hover_bg": "#e6ebf1",
        }

    t = theme
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

            :root {{
                --bg: {t['bg']}; --surface: {t['surface']}; --surface-2: {t['surface2']};
                --border: {t['border']}; --text: {t['text']}; --muted: {t['muted']};
                --accent: {t['accent']}; --accent-2: {t['accent2']};
                --danger: {t['danger']}; --ok: {t['ok']};
            }}

            html, body, [data-testid="stAppViewContainer"] {{
                color: var(--text);
                font-family: 'Inter', -apple-system, sans-serif;
                font-size: 15px;
            }}
            [data-testid="stAppViewContainer"] {{
                background: linear-gradient(180deg, {t['app_grad_a']} 0%, {t['app_grad_b']} 100%);
            }}

            /* Sidebar */
            [data-testid="stSidebar"] {{
                background: var(--surface);
                border-right: 1px solid var(--border);
            }}
            [data-testid="stSidebar"] * {{ font-family: 'Inter', sans-serif; }}

            /* Section headers */
            h1, h2, h3 {{ letter-spacing: 0.3px; font-weight: 600; color: var(--text); }}
            .stSubheader, h3 {{
                border-bottom: 1px solid var(--border);
                padding-bottom: 6px;
            }}

            /* Captions / muted text — boosted contrast for readability */
            .stSidebar .stCaption, [data-testid="stCaptionContainer"] {{
                color: var(--muted) !important;
                font-size: 13px !important;
            }}

            /* Buttons */
            .stButton > button {{
                border-radius: 8px;
                border: 1px solid var(--border);
                background: var(--surface-2);
                color: var(--text);
                font-weight: 500;
                transition: all .15s ease;
            }}
            .stButton > button:hover {{
                border-color: var(--accent);
                color: var(--accent);
                background: {t['btn_hover_bg']};
            }}

            /* Text inputs / selectboxes / textareas */
            .stTextInput > div > div > input,
            .stTextArea > div > textarea,
            .stSelectbox > div > div > div {{
                background: var(--surface-2) !important;
                color: var(--text) !important;
                border-color: var(--border) !important;
            }}

            /* Chat input — high-contrast placeholder */
            [data-testid="stChatInput"] textarea {{
                color: var(--text) !important;
                background: var(--surface-2) !important;
                border-radius: 12px !important;
                border: 1px solid var(--border) !important;
                font-size: 15px !important;
            }}
            [data-testid="stChatInput"] textarea::placeholder {{
                color: var(--muted) !important;
                opacity: 1 !important;
            }}

            /* Chat message bubbles */
            [data-testid="stChatMessage"] {{
                border-radius: 14px;
                border: 1px solid var(--border);
                background: var(--surface);
                color: var(--text);
            }}
            [data-testid="stChatMessage"] p,
            [data-testid="stChatMessage"] li {{
                color: var(--text) !important;
                font-size: 15px !important;
                line-height: 1.6 !important;
            }}

            /* Sliders / toggles accent */
            .stSlider > div > div > div > div {{ background: var(--accent) !important; }}

            /* Status pills */
            .pill {{
                display: inline-block;
                padding: 4px 12px;
                border-radius: 999px;
                font-size: 13px;
                font-weight: 600;
                letter-spacing: .3px;
            }}
            .pill-ok  {{ background: rgba(52,211,153,.18);  color: var(--ok);     border:1px solid rgba(52,211,153,.5); }}
            .pill-no  {{ background: rgba(248,113,113,.18);  color: var(--danger); border:1px solid rgba(248,113,113,.5); }}
            .pill-id  {{ background: rgba(45,212,191,.18);   color: var(--accent); border:1px solid rgba(45,212,191,.5); }}

            /* Header banner */
            .hero {{
                background: linear-gradient(135deg, {t['hero_grad_a']} 0%, {t['hero_grad_b']} 60%, {t['hero_grad_c']} 100%);
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 22px 28px;
                margin-bottom: 18px;
            }}
            .hero h1 {{ margin:0; font-size: 26px; color: var(--text); }}
            .hero .sub {{ color: var(--muted); font-size: 14px; margin-top:6px; }}
            .hero .meta {{ margin-top: 14px; font-size: 14px; color: var(--muted); }}
            .hero .meta b {{ color: var(--text); }}
            .hero code {{
                background: var(--surface-2);
                border: 1px solid var(--border);
                border-radius: 6px;
                padding: 2px 8px;
                color: var(--accent);
            }}

            /* Welcome cards — wrap on small screens, full text visible */
            .welcome-grid {{
                display: flex; flex-wrap: wrap; gap: 16px; margin-top: 22px;
            }}
            .welcome-card {{
                flex: 1 1 240px;
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 14px;
                padding: 20px;
                transition: transform .15s ease, border-color .15s ease;
            }}
            .welcome-card:hover {{ transform: translateY(-3px); border-color: var(--accent); }}
            .welcome-card .ico {{ font-size: 28px; }}
            .welcome-card .ttl {{ font-weight: 600; margin: 10px 0 6px; color: var(--text); font-size: 16px; }}
            .welcome-card .desc {{ font-size: 14px; color: var(--muted); line-height: 1.55; }}

            .hint {{
                text-align: center; margin-top: 28px; color: var(--muted);
                font-size: 15px; line-height: 1.7;
            }}
            .hint b {{ color: var(--text); }}
            .hint a {{ color: var(--accent); text-decoration: none; font-weight: 600; }}
            .hint a:hover {{ text-decoration: underline; }}

            /* Theme toggle button */
            .theme-toggle {{
                position: fixed; top: 14px; right: 18px; z-index: 9999;
                background: var(--surface-2); border: 1px solid var(--border);
                border-radius: 999px; padding: 8px 16px; cursor: pointer;
                font-size: 14px; font-weight: 600; color: var(--text);
                box-shadow: 0 2px 8px rgba(0,0,0,.15);
                transition: all .15s ease;
            }}
            .theme-toggle:hover {{ border-color: var(--accent); color: var(--accent); }}

            /* Hide clutter */
            #MainMenu {{ visibility: hidden; }}
            footer {{ visibility: hidden; }}

            /* Scrollbar */
            ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
            ::-webkit-scrollbar-thumb {{ background: {t['scroll_thumb']}; border-radius: 6px; }}
            ::-webkit-scrollbar-thumb:hover {{ background: {t['scroll_thumb_h']}; }}
            ::-webkit-scrollbar-track {{ background: {t['scroll_track']}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_theme_toggle() -> None:
    """Render the dark/light mode toggle button (top-right of main area)."""
    is_dark = st.session_state.get("theme_mode", "dark") == "dark"
    icon = "🌙" if is_dark else "☀️"
    label = "Dark" if is_dark else "Light"
    cols = st.columns([1, 1, 1, 1, 1, 1])
    with cols[5]:
        if st.button(f"{icon} {label}", key="theme_toggle_btn", use_container_width=True):
            st.session_state.theme_mode = "light" if is_dark else "dark"
            st.rerun()


def main() -> None:
    """Application entrypoint."""
    init_session_state()
    configure_page()
    render_theme_toggle()
    render_sidebar()
    render_chat_surface()


if __name__ == "__main__":
    main()