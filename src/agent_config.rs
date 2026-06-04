use anyhow::{Result, bail};
use serde_json::{Value, json};

use crate::prompt::AGENT_PREAMBLE;

pub(crate) const DEFAULT_OLLAMA_NUM_CTX: u64 = 9068;
const THINK_BOOL_MODEL_MARKERS: &[&str] = &[
    "qwen3",
    "qwen3.5",
    "qwen3.6",
    "deepseek-r1",
    "deepseek-v3.1",
    "deepseek-v3.2",
    "deepseek-v4",
    "glm-4.7",
    "glm-5",
    "glm-5.1",
    "minimax-m2.5",
    "minimax-m2.7",
    "minimax-m3",
    "lfm2.5",
    "nemotron3",
    "nemotron-3",
    "kimi-k2.5",
    "kimi-k2.6",
    "gemini-3-flash-preview",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ThinkingMode {
    OllamaBool,
    OllamaLevel(&'static str),
    PromptToken,
    Disabled,
}

pub(crate) struct AgentPromptConfig {
    pub(crate) preamble: String,
    pub(crate) params: Value,
    pub(crate) thinking_mode: ThinkingMode,
}
pub(crate) fn parse_ollama_num_ctx(value: Option<String>) -> Result<u64> {
    let Some(value) = value else {
        return Ok(DEFAULT_OLLAMA_NUM_CTX);
    };

    let trimmed = value.trim();
    let parsed = trimmed
        .parse::<u64>()
        .map_err(|_| anyhow::anyhow!("OLLAMA_NUM_CTX must be a positive integer: {trimmed}"))?;
    if parsed == 0 {
        bail!("OLLAMA_NUM_CTX must be greater than 0");
    }

    Ok(parsed)
}

pub(crate) fn build_agent_prompt_config(model: &str, num_ctx: u64) -> AgentPromptConfig {
    match thinking_mode_for_model(model) {
        ThinkingMode::OllamaBool => AgentPromptConfig {
            preamble: AGENT_PREAMBLE.to_string(),
            params: json!({ "think": true, "num_ctx": num_ctx }),
            thinking_mode: ThinkingMode::OllamaBool,
        },
        ThinkingMode::OllamaLevel(level) => AgentPromptConfig {
            preamble: AGENT_PREAMBLE.to_string(),
            params: json!({ "think": level, "num_ctx": num_ctx }),
            thinking_mode: ThinkingMode::OllamaLevel(level),
        },
        ThinkingMode::PromptToken => AgentPromptConfig {
            preamble: format!("<|think|>\n{AGENT_PREAMBLE}"),
            params: json!({ "num_ctx": num_ctx }),
            thinking_mode: ThinkingMode::PromptToken,
        },
        ThinkingMode::Disabled => AgentPromptConfig {
            preamble: AGENT_PREAMBLE.to_string(),
            params: json!({ "num_ctx": num_ctx }),
            thinking_mode: ThinkingMode::Disabled,
        },
    }
}

pub(crate) fn thinking_mode_for_model(model: &str) -> ThinkingMode {
    let model = model.to_ascii_lowercase();
    match model.as_str() {
        m if m.contains("gemma4") || m.contains("gemma-4") => ThinkingMode::PromptToken,
        m if m.contains("gpt-oss") => ThinkingMode::OllamaLevel("medium"),
        m if THINK_BOOL_MODEL_MARKERS
            .iter()
            .any(|marker| m.contains(marker)) =>
        {
            ThinkingMode::OllamaBool
        }
        _ => ThinkingMode::Disabled,
    }
}
