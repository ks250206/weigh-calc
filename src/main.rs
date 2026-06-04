use anyhow::{Result, bail};
use futures::StreamExt;
use rig::agent::{MultiTurnStreamItem, StreamingResult};
use rig::client::CompletionClient;
use rig::completion::{Message, ToolDefinition};
use rig::message::{ReasoningContent, ToolResult, ToolResultContent};
use rig::providers::ollama;
use rig::streaming::{StreamedAssistantContent, StreamedUserContent, StreamingChat};
use rig::tool::Tool;
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::io::{self, Write};
use tokio::time::{Duration, Instant, timeout};

const EPS: f64 = 1.0e-8;
const STREAM_CHUNK_TIMEOUT_SECS: u64 = 90;
const STREAM_TOTAL_TIMEOUT_SECS: u64 = 240;
const DEFAULT_OLLAMA_NUM_CTX: u64 = 9068;
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
enum ThinkingMode {
    OllamaBool,
    OllamaLevel(&'static str),
    PromptToken,
    Disabled,
}

struct AgentPromptConfig {
    preamble: String,
    params: Value,
    thinking_mode: ThinkingMode,
}

#[derive(Debug, Clone)]
struct Compound {
    formula: String,
    atoms: BTreeMap<String, f64>,
    molar_mass: f64,
}

#[derive(Debug, Clone, Serialize)]
struct WeighingResult {
    formula: String,
    moles: f64,
    grams: f64,
}

#[derive(Debug, Clone)]
struct WeighingCalculation {
    results: Vec<WeighingResult>,
    gas_reactants: Vec<WeighingResult>,
    volatile_byproducts: Vec<WeighingResult>,
    reaction_equation: String,
}

#[derive(Debug, Clone, PartialEq)]
enum CalculationError {
    Inconsistent,
    NotUnique,
    NegativeAmount { formula: String, moles: f64 },
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut app = WeighingAgentApp::new()?;
    app.run().await
}

struct WeighingAgentApp {
    ollama_base_url: String,
    model: String,
    ollama_num_ctx: u64,
    line_editor: DefaultEditor,
    state: ConversationState,
    last_suggested_formulas: Vec<String>,
}

impl WeighingAgentApp {
    fn new() -> Result<Self> {
        Ok(Self {
            ollama_base_url: std::env::var("OLLAMA_BASE_URL")
                .unwrap_or_else(|_| "http://localhost:11434".to_string()),
            model: std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen3.6:35b".to_string()),
            ollama_num_ctx: parse_ollama_num_ctx(std::env::var("OLLAMA_NUM_CTX").ok())?,
            line_editor: DefaultEditor::new()?,
            state: ConversationState::default(),
            last_suggested_formulas: Vec::new(),
        })
    }

    async fn run(&mut self) -> Result<()> {
        let client = ollama::Client::new(self.ollama_base_url.as_str())?;
        let agent_config = build_agent_prompt_config(&self.model, self.ollama_num_ctx);
        if agent_config.thinking_mode == ThinkingMode::Disabled {
            println!(
                "注意: このモデルでは thinking 出力を有効化しません。Thinking... 表示は出ない場合があります。model: {}",
                self.model
            );
        }

        let agent = client
            .agent(&self.model)
            .preamble(&agent_config.preamble)
            .additional_params(agent_config.params)
            .default_max_turns(8)
            .tool(ValidateFormulaTool)
            .tool(CalculateWeighingTool)
            .build();

        let mut chat_history = Vec::<Message>::new();
        let stream = agent
            .stream_chat("ワークフローを開始してください。", chat_history.clone())
            .await;
        let (_, updated_history, response_text) = print_streaming_agent_response(stream).await?;
        self.update_last_suggestions(&response_text);
        if let Some(updated_history) = updated_history {
            chat_history = updated_history;
        }

        loop {
            let Some(input) = self.read_line("> ")? else {
                break;
            };
            if input.trim().is_empty() {
                continue;
            }
            if let Some(should_continue) = self.handle_local_command(&input)? {
                if should_continue {
                    continue;
                }
                break;
            }

            self.update_state_from_user_input(&input);
            let prompt = self.state.wrap_user_input(&input);
            let stream = agent.stream_chat(prompt, chat_history.clone()).await;
            let (should_end, updated_history, response_text) =
                print_streaming_agent_response(stream).await?;
            self.update_last_suggestions(&response_text);
            if let Some(updated_history) = updated_history {
                chat_history = updated_history;
            }
            if should_end {
                break;
            }
        }

        Ok(())
    }

    fn read_line(&mut self, prompt: &str) -> Result<Option<String>> {
        match self.line_editor.readline(prompt) {
            Ok(input) => {
                if !input.trim().is_empty() {
                    let _ = self.line_editor.add_history_entry(input.as_str());
                }
                Ok(Some(input.trim().to_string()))
            }
            Err(ReadlineError::Interrupted) => Ok(None),
            Err(ReadlineError::Eof) => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    fn handle_local_command(&self, input: &str) -> Result<Option<bool>> {
        match input.trim() {
            "/quit" | "/exit" => Ok(Some(false)),
            "/clear" => {
                print!("\x1b[2J\x1b[H");
                io::stdout().flush()?;
                Ok(Some(true))
            }
            _ => Ok(None),
        }
    }

    fn update_state_from_user_input(&mut self, input: &str) {
        if is_confirmation_only(input)
            && self.state.precursor_formulas.is_empty()
            && !self.last_suggested_formulas.is_empty()
        {
            self.state.precursor_formulas = self.last_suggested_formulas.clone();
            return;
        }

        let formulas = extract_valid_formulas(input);
        if self.state.target_formula.is_none() {
            if let Some(formula) = formulas.first() {
                self.state.target_formula = Some(formula.clone());
            } else if let Some(formula) = guess_material_formula(input) {
                self.last_suggested_formulas = vec![formula];
            }
        } else if self.state.target_mass_g.is_some() && !formulas.is_empty() {
            let target = self.state.target_formula.as_deref();
            let precursor_formulas = formulas
                .into_iter()
                .filter(|formula| Some(formula.as_str()) != target)
                .collect::<Vec<_>>();
            if !precursor_formulas.is_empty() {
                self.state.precursor_formulas = precursor_formulas;
            }
        }

        if let Some(mass_g) = extract_mass_g(input) {
            self.state.target_mass_g = Some(mass_g);
        }
    }

    fn update_last_suggestions(&mut self, response_text: &str) {
        let formulas = extract_valid_formulas(response_text);
        if formulas.len() >= 2 {
            self.last_suggested_formulas = formulas;
        }
    }
}

fn parse_ollama_num_ctx(value: Option<String>) -> Result<u64> {
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

fn build_agent_prompt_config(model: &str, num_ctx: u64) -> AgentPromptConfig {
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

fn thinking_mode_for_model(model: &str) -> ThinkingMode {
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

const END_MARKER: &str = "[[END_WORKFLOW]]";

#[derive(Debug, Default)]
struct ConversationState {
    target_formula: Option<String>,
    target_mass_g: Option<f64>,
    precursor_formulas: Vec<String>,
}

impl ConversationState {
    fn wrap_user_input(&self, input: &str) -> String {
        format!(
            "[現在の状態]\n目的組成: {}\n目標質量_g: {}\n原料組成: {}\n\n[ユーザー入力]\n{}",
            self.target_formula.as_deref().unwrap_or("未確定"),
            self.target_mass_g
                .map(|mass| format!("{mass}"))
                .unwrap_or_else(|| "未確定".to_string()),
            if self.precursor_formulas.is_empty() {
                "未確定".to_string()
            } else {
                self.precursor_formulas.join(", ")
            },
            input
        )
    }
}

fn is_confirmation_only(input: &str) -> bool {
    let normalized = input
        .trim()
        .trim_matches(|ch: char| ch.is_ascii_punctuation() || matches!(ch, '。' | '、'))
        .to_ascii_lowercase();
    matches!(
        normalized.as_str(),
        "ok" | "yes" | "y" | "はい" | "それでok" | "それでよい" | "それで良い" | "それでいい"
    )
}

fn extract_valid_formulas(input: &str) -> Vec<String> {
    let mut formulas = Vec::new();
    for token in input.split(|ch: char| {
        ch.is_whitespace()
            || matches!(
                ch,
                ',' | '、' | ';' | '；' | ':' | '：' | '"' | '\'' | '`' | '「' | '」'
            )
    }) {
        let candidate = token
            .trim()
            .trim_matches(|ch: char| {
                ch.is_ascii_punctuation()
                    && ch != ')'
                    && ch != ']'
                    && ch != '}'
                    && ch != '/'
                    && ch != '.'
            })
            .trim();
        if candidate.is_empty() {
            continue;
        }
        if parse_compound(candidate).is_ok() && !formulas.iter().any(|formula| formula == candidate)
        {
            formulas.push(candidate.to_string());
        }
    }
    formulas
}

fn guess_material_formula(input: &str) -> Option<String> {
    let normalized = input
        .trim()
        .to_ascii_lowercase()
        .replace([' ', '-', '_'], "");
    let formula = match normalized.as_str() {
        "lfp" => "LiFePO4",
        "nmc111" | "lini1/3co1/3mn1/3o2" => "LiNi1/3Co1/3Mn1/3O2",
        "nmc532" => "LiNi0.5Co0.3Mn0.2O2",
        "nmc622" => "LiNi0.6Co0.2Mn0.2O2",
        "nmc811" => "LiNi0.8Co0.1Mn0.1O2",
        "latp" => "Li1.3Al0.3Ti1.7(PO4)3",
        "llzo" => "Li7La3Zr2O12",
        "lgps" => "Li10GeP2S12",
        _ => return None,
    };
    Some(formula.to_string())
}

fn extract_mass_g(input: &str) -> Option<f64> {
    for token in input.split_whitespace() {
        let normalized = token
            .trim()
            .trim_matches(|ch: char| ch.is_ascii_punctuation())
            .to_ascii_lowercase();
        let Some(number) = normalized.strip_suffix('g') else {
            continue;
        };
        if number.is_empty() || normalized.ends_with("mg") {
            continue;
        }
        if let Ok(value) = number.parse::<f64>()
            && value.is_finite()
            && value > 0.0
        {
            return Some(value);
        }
    }

    input
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|value| *value > 0.0)
}

const AGENT_PREAMBLE: &str = r#"
あなたは無機材料の秤量計算を支援する日本語のAIエージェントです。

必ず次のワークフローで進行してください。
1. はじめに自己紹介し、ユーザーに「作りたい組成」と「何g欲しいか」を聞く。
2. 目的組成を受け取ったら validate_formula tool で確認する。不正または無機材料らしくない場合は再入力を依頼する。
2a. ユーザーの入力が化学式ではなく材料名・略称・曖昧な表現に見える場合は、推測できる化学式候補を示して「これですか？」と確認する。例: LFP -> LiFePO4, NMC111 -> LiNi1/3Co1/3Mn1/3O2, LATP -> Li1.3Al0.3Ti1.7(PO4)3。確認が取れるまで計算に進まない。
3. 目標質量は0より大きいg単位の数値として扱う。不正なら再入力を依頼する。
4. 次に原料組成を複数の組成のリストとして聞く。例: Li2S, GeS2, P2S5。
5. 原料組成は各項目を validate_formula tool で確認する。不正または無機材料らしくないものがあれば、原料リスト全体の再入力を依頼する。
5a. 原料組成の一部が化学式ではなく曖昧な材料名に見える場合も、推測できる化学式候補を示して「この原料はこれですか？」と確認する。確認が取れるまで calculate_weighing tool を呼ばない。
5b. 原料組成を聞いている場面で、あなたが直前に具体的な原料組成例や候補を提示しており、ユーザーが「OK」「はい」「それでOK」「それでよい」のような確認だけを返した場合は、その直前に提示した原料組成リストを採用する。直前に具体的な原料組成リストがない場合だけ、原料組成を化学式のリストで再入力してもらう。
5c. 目的組成と目標質量がすでに会話履歴にある状態で原料組成を再入力してもらった場合、目的組成や目標質量を再度聞いてはいけない。そのまま既存の目的組成と目標質量と新しい原料リストで calculate_weighing tool を呼ぶ。
6. 目的組成、目標質量、原料リストがそろったら、まず volatile_byproduct_formulas を指定せずに calculate_weighing tool を必ず呼ぶ。原料リストに O2, N2, H2 などのガスが含まれる場合も除外せず、そのまま precursor_formulas に含める。秤量値を自分で暗算・推測してはいけない。
7. tool が ok=false で、LiOH, Li2CO3, NH4H2PO4 などから CO2/H2O/NH3 等の揮発が考えられる場合は、あなたが妥当な揮発成分候補を考え、volatile_byproduct_formulas に入れて calculate_weighing tool を再度呼び、反応式の収支を確認する。
8. 揮発成分を含む tool 結果が ok=true の場合は、必ず reaction_equation をユーザーに見せ、「この反応式では ... が揮発すると仮定します。それでもこの前提で秤量計算してよいですか？」と確認する。この時点では results の秤量値を表示しない。反応式がシステム表示済みの場合は同じ反応式を長く繰り返さず、確認文を続ける。
9. ユーザーが揮発前提に同意した場合だけ、直前と同じ volatile_byproduct_formulas で calculate_weighing tool を呼び、reaction_equation と results に基づいて固体・液体原料ごとの g, mg, mol をリスト表示する。gas_reactants が空でない場合は、ガス原料は秤量対象外だが元素収支に含めたことを一文で示す。volatile_byproducts も一文で示す。その後「再度計算しますか？」と聞く。
10. 揮発なしで tool が ok=true の場合も、必ず reaction_equation を表示してから、tool の results だけを根拠に固体・液体原料ごとの g, mg, mol をリスト表示する。gas_reactants が空でない場合は、ガス原料は秤量対象外だが元素収支に含めたことを一文で示す。その後「再度計算しますか？」と聞く。
11. 再計算する場合は目的組成と何g欲しいかを聞くところへ戻る。再計算しない場合は短く終了し、最後に [[END_WORKFLOW]] を付ける。

応答は簡潔にしてください。ただし計算結果を出すときは「計算が完了しました。」のような短い完了文を入れてから結果を表示してください。計算根拠や説明は必要な場面だけにしてください。
Markdown 記法は使わないでください。太字の `**`、箇条書き以外の `*`、見出しの `#`、表の `|`、LaTeX、コードフェンスを使ってはいけません。たとえば **Li2S** のように `**` を付けず、`- Li2S: 0.390 g ...` のようなプレーンテキストの箇条書きで表示してください。
秤量結果の数値には `...` の省略表記を使わないでください。g は小数4桁、mg は小数1桁、mol は有効数字4桁程度で表示してください。
"#;

async fn print_streaming_agent_response<R>(
    mut stream: StreamingResult<R>,
) -> Result<(bool, Option<Vec<Message>>, String)>
where
    R: Clone + Unpin,
{
    let mut response_text = String::new();
    let mut pending_text = String::new();
    let mut should_end = false;
    let mut printed_visible_text = false;
    let mut printed_reaction_equations = BTreeSet::new();
    let mut showing_thinking = false;
    let mut thinking_char_count = 0usize;
    let mut suppress_duplicate_reaction_line = false;
    let mut duplicate_reaction_line_buffer = String::new();
    let mut updated_history = None;
    let started_at = Instant::now();

    loop {
        if started_at.elapsed() > Duration::from_secs(STREAM_TOTAL_TIMEOUT_SECS) {
            clear_thinking_indicator(&mut showing_thinking)?;
            println!(
                "応答が長時間完了しなかったため中断しました。もう一度、化学式のリストを具体的に入力してください。"
            );
            return Ok((false, updated_history, response_text));
        }

        let Some(chunk) = (match timeout(
            Duration::from_secs(STREAM_CHUNK_TIMEOUT_SECS),
            stream.next(),
        )
        .await
        {
            Ok(chunk) => chunk,
            Err(_) => {
                clear_thinking_indicator(&mut showing_thinking)?;
                println!(
                    "応答が停止したため中断しました。もう一度、化学式のリストを具体的に入力してください。"
                );
                return Ok((false, updated_history, response_text));
            }
        }) else {
            break;
        };

        let item = chunk?;
        match item {
            MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(text)) => {
                clear_thinking_indicator(&mut showing_thinking)?;
                response_text.push_str(&text.text);
                let sanitized = sanitize_assistant_text(&text.text);
                let filtered = filter_duplicate_reaction_line(
                    &sanitized,
                    !printed_reaction_equations.is_empty(),
                    &mut suppress_duplicate_reaction_line,
                    &mut duplicate_reaction_line_buffer,
                );
                pending_text.push_str(&filtered);
                printed_visible_text |= flush_visible_stream_text(&mut pending_text, false)?;
            }
            MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Reasoning(
                reasoning,
            )) => {
                update_thinking_indicator(
                    reasoning_text_len(&reasoning.content),
                    &mut thinking_char_count,
                    &mut showing_thinking,
                )?;
            }
            MultiTurnStreamItem::StreamAssistantItem(
                StreamedAssistantContent::ReasoningDelta { reasoning, .. },
            ) => {
                update_thinking_indicator(
                    reasoning.chars().count(),
                    &mut thinking_char_count,
                    &mut showing_thinking,
                )?;
            }
            MultiTurnStreamItem::StreamAssistantItem(
                StreamedAssistantContent::ToolCall { .. }
                | StreamedAssistantContent::ToolCallDelta { .. }
                | StreamedAssistantContent::Final(_),
            )
            | MultiTurnStreamItem::CompletionCall(_) => {}
            MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                tool_result,
                ..
            }) => {
                clear_thinking_indicator(&mut showing_thinking)?;
                let flushed_text = flush_visible_stream_text(&mut pending_text, true)?;
                if flushed_text {
                    println!();
                }
                printed_visible_text |= flushed_text;
                printed_visible_text |= print_reaction_equation_from_tool_result(
                    &tool_result,
                    &mut printed_reaction_equations,
                )?;
            }
            MultiTurnStreamItem::FinalResponse(response) => {
                clear_thinking_indicator(&mut showing_thinking)?;
                should_end = response.response().contains(END_MARKER);
                updated_history = response.history().map(|history| history.to_vec());
            }
            _ => {}
        }
    }

    should_end |= pending_text.contains(END_MARKER);
    clear_thinking_indicator(&mut showing_thinking)?;
    pending_text.push_str(&flush_duplicate_reaction_line_buffer(
        &mut duplicate_reaction_line_buffer,
    ));
    printed_visible_text |= flush_visible_stream_text(&mut pending_text, true)?;
    if should_end && !printed_visible_text {
        println!("終了します。");
    } else if printed_visible_text {
        println!();
    }
    Ok((should_end, updated_history, response_text))
}

fn sanitize_assistant_text(text: &str) -> String {
    text.replace("**", "").replace("__", "").replace("```", "")
}

fn filter_duplicate_reaction_line(
    text: &str,
    reaction_was_printed_by_tool: bool,
    suppress_duplicate_reaction_line: &mut bool,
    duplicate_reaction_line_buffer: &mut String,
) -> String {
    if !reaction_was_printed_by_tool {
        return text.to_string();
    }

    let mut output = String::new();

    for ch in text.chars() {
        if *suppress_duplicate_reaction_line {
            if ch == '\n' {
                *suppress_duplicate_reaction_line = false;
            }
            continue;
        }

        duplicate_reaction_line_buffer.push(ch);
        if duplicate_reaction_line_buffer.contains("反応式") {
            *suppress_duplicate_reaction_line = true;
            duplicate_reaction_line_buffer.clear();
            if ch == '\n' {
                *suppress_duplicate_reaction_line = false;
            }
            continue;
        }

        if ch == '\n' {
            output.push_str(duplicate_reaction_line_buffer);
            duplicate_reaction_line_buffer.clear();
        }
    }

    output
}

fn flush_duplicate_reaction_line_buffer(duplicate_reaction_line_buffer: &mut String) -> String {
    if duplicate_reaction_line_buffer.contains("反応式") {
        duplicate_reaction_line_buffer.clear();
        return String::new();
    }
    std::mem::take(duplicate_reaction_line_buffer)
}

fn reasoning_text_len(content: &[ReasoningContent]) -> usize {
    content
        .iter()
        .map(|content| match content {
            ReasoningContent::Text { text, .. } => text.chars().count(),
            ReasoningContent::Summary(text) => text.chars().count(),
            ReasoningContent::Encrypted(text) => text.chars().count(),
            ReasoningContent::Redacted { data } => data.chars().count(),
            _ => 0,
        })
        .sum()
}

fn update_thinking_indicator(
    chunk_len: usize,
    thinking_char_count: &mut usize,
    showing_thinking: &mut bool,
) -> Result<()> {
    *thinking_char_count += chunk_len;
    print!(
        "\r\x1b[2K\x1b[5mThinking... ({} chars)\x1b[0m",
        *thinking_char_count
    );
    io::stdout().flush()?;
    *showing_thinking = true;
    Ok(())
}

fn clear_thinking_indicator(showing_thinking: &mut bool) -> Result<()> {
    if *showing_thinking {
        print!("\r\x1b[2K");
        io::stdout().flush()?;
        *showing_thinking = false;
    }
    Ok(())
}

fn print_reaction_equation_from_tool_result(
    tool_result: &ToolResult,
    printed_reaction_equations: &mut BTreeSet<String>,
) -> Result<bool> {
    for content in tool_result.content.iter() {
        let ToolResultContent::Text(text) = content else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&text.text) else {
            continue;
        };
        if value.get("ok").and_then(|ok| ok.as_bool()) != Some(true) {
            continue;
        }
        let Some(equation) = value
            .get("reaction_equation")
            .and_then(|equation| equation.as_str())
            .filter(|equation| !equation.is_empty())
        else {
            continue;
        };
        if printed_reaction_equations.insert(equation.to_string()) {
            println!("反応式: {equation}");
            return Ok(true);
        }
    }

    Ok(false)
}

fn flush_visible_stream_text(pending_text: &mut String, force: bool) -> Result<bool> {
    if pending_text.is_empty() {
        return Ok(false);
    }

    if let Some(marker_start) = pending_text.find(END_MARKER) {
        let visible = pending_text[..marker_start].to_string();
        let printed = !visible.is_empty();
        if !visible.is_empty() {
            print!("{visible}");
            io::stdout().flush()?;
        }
        pending_text.clear();
        return Ok(printed);
    }

    if force {
        let visible = pending_text.replace(END_MARKER, "");
        let printed = !visible.is_empty();
        if !visible.is_empty() {
            print!("{visible}");
            io::stdout().flush()?;
        }
        pending_text.clear();
        return Ok(printed);
    }

    let keep_tail = END_MARKER.len().saturating_sub(1);
    if pending_text.len() <= keep_tail {
        return Ok(false);
    }

    let split_at = pending_text
        .char_indices()
        .map(|(index, _)| index)
        .filter(|index| pending_text.len() - *index <= keep_tail)
        .next()
        .unwrap_or(pending_text.len());
    let visible = pending_text[..split_at].to_string();
    let tail = pending_text[split_at..].to_string();
    print!("{visible}");
    io::stdout().flush()?;
    *pending_text = tail;
    Ok(!visible.is_empty())
}

#[derive(Debug)]
struct ToolCallFailure;

impl fmt::Display for ToolCallFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("tool call failed")
    }
}

impl Error for ToolCallFailure {}

#[derive(Debug, Deserialize)]
struct ValidateFormulaArgs {
    formula: String,
}

#[derive(Debug, Serialize)]
struct ValidateFormulaOutput {
    valid: bool,
    formula: String,
    message: String,
    elements: BTreeMap<String, f64>,
    molar_mass_g_per_mol: Option<f64>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ValidateFormulaTool;

impl Tool for ValidateFormulaTool {
    const NAME: &'static str = "validate_formula";

    type Error = ToolCallFailure;
    type Args = ValidateFormulaArgs;
    type Output = ValidateFormulaOutput;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "化学式を検証し、無機材料の組成として扱えるか、元素数とモル質量を返す。"
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "formula": {
                        "type": "string",
                        "description": "検証する化学式。例: Li10GeP2S12"
                    }
                },
                "required": ["formula"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(match parse_compound(&args.formula) {
            Ok(compound) if is_inorganic_material_like(&compound) => ValidateFormulaOutput {
                valid: true,
                formula: compound.formula,
                message: "valid".to_string(),
                elements: compound.atoms,
                molar_mass_g_per_mol: Some(compound.molar_mass),
            },
            Ok(compound) => ValidateFormulaOutput {
                valid: false,
                formula: compound.formula,
                message: "無機材料の組成として扱いにくい入力です。".to_string(),
                elements: compound.atoms,
                molar_mass_g_per_mol: Some(compound.molar_mass),
            },
            Err(error) => ValidateFormulaOutput {
                valid: false,
                formula: args.formula,
                message: error.to_string(),
                elements: BTreeMap::new(),
                molar_mass_g_per_mol: None,
            },
        })
    }
}

#[derive(Debug, Deserialize)]
struct CalculateWeighingArgs {
    target_formula: String,
    target_mass_g: f64,
    precursor_formulas: Vec<String>,
    #[serde(default)]
    volatile_byproduct_formulas: Vec<String>,
}

#[derive(Debug, Serialize)]
struct CalculateWeighingOutput {
    ok: bool,
    target_formula: String,
    target_mass_g: f64,
    results: Vec<WeighingResult>,
    gas_reactants: Vec<WeighingResult>,
    volatile_byproducts: Vec<WeighingResult>,
    reaction_equation: Option<String>,
    error: Option<String>,
    suggestions: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct CalculateWeighingTool;

impl Tool for CalculateWeighingTool {
    const NAME: &'static str = "calculate_weighing";

    type Error = ToolCallFailure;
    type Args = CalculateWeighingArgs;
    type Output = CalculateWeighingOutput;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description:
                "目的組成、目標質量[g]、原料組成リストから元素収支に基づく秤量値を計算する。"
                    .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "target_formula": {
                        "type": "string",
                        "description": "目的組成の化学式。例: Li10GeP2S12"
                    },
                    "target_mass_g": {
                        "type": "number",
                        "description": "作製したい目的物の質量[g]"
                    },
                    "precursor_formulas": {
                        "type": "array",
                        "description": "原料組成の化学式リスト。O2などのガス原料も除外せず含める。",
                        "items": { "type": "string" }
                    },
                    "volatile_byproduct_formulas": {
                        "type": "array",
                        "description": "LLMが提案した揮発副生成物の化学式リスト。揮発を仮定しない初回計算では省略または空配列にする。例: [\"CO2\", \"H2O\", \"NH3\"]",
                        "items": { "type": "string" }
                    }
                },
                "required": ["target_formula", "target_mass_g", "precursor_formulas"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(calculate_weighing_tool_output(args))
    }
}

fn calculate_weighing_tool_output(args: CalculateWeighingArgs) -> CalculateWeighingOutput {
    let target = match parse_compound(&args.target_formula) {
        Ok(compound) if is_inorganic_material_like(&compound) => compound,
        Ok(_) => {
            return CalculateWeighingOutput {
                ok: false,
                target_formula: args.target_formula,
                target_mass_g: args.target_mass_g,
                results: Vec::new(),
                gas_reactants: Vec::new(),
                volatile_byproducts: Vec::new(),
                reaction_equation: None,
                error: Some("目的組成が無機材料の組成として扱いにくい入力です。".to_string()),
                suggestions: Vec::new(),
            };
        }
        Err(error) => {
            return CalculateWeighingOutput {
                ok: false,
                target_formula: args.target_formula,
                target_mass_g: args.target_mass_g,
                results: Vec::new(),
                gas_reactants: Vec::new(),
                volatile_byproducts: Vec::new(),
                reaction_equation: None,
                error: Some(format!("目的組成が不正です: {error}")),
                suggestions: Vec::new(),
            };
        }
    };

    if !args.target_mass_g.is_finite() || args.target_mass_g <= 0.0 {
        return CalculateWeighingOutput {
            ok: false,
            target_formula: target.formula,
            target_mass_g: args.target_mass_g,
            results: Vec::new(),
            gas_reactants: Vec::new(),
            volatile_byproducts: Vec::new(),
            reaction_equation: None,
            error: Some("目標質量は0より大きいg単位の数値で指定してください。".to_string()),
            suggestions: Vec::new(),
        };
    }

    let mut precursors = Vec::new();
    for formula in &args.precursor_formulas {
        match parse_compound(formula) {
            Ok(compound) if is_inorganic_material_like(&compound) => precursors.push(compound),
            Ok(_) => {
                return CalculateWeighingOutput {
                    ok: false,
                    target_formula: target.formula,
                    target_mass_g: args.target_mass_g,
                    results: Vec::new(),
                    gas_reactants: Vec::new(),
                    volatile_byproducts: Vec::new(),
                    reaction_equation: None,
                    error: Some(format!(
                        "原料 {formula} は無機材料の組成として扱いにくい入力です。"
                    )),
                    suggestions: Vec::new(),
                };
            }
            Err(error) => {
                return CalculateWeighingOutput {
                    ok: false,
                    target_formula: target.formula,
                    target_mass_g: args.target_mass_g,
                    results: Vec::new(),
                    gas_reactants: Vec::new(),
                    volatile_byproducts: Vec::new(),
                    reaction_equation: None,
                    error: Some(format!("原料 {formula} が不正です: {error}")),
                    suggestions: Vec::new(),
                };
            }
        }
    }

    let mut volatile_byproducts = Vec::new();
    for formula in &args.volatile_byproduct_formulas {
        match parse_compound(formula) {
            Ok(compound) => volatile_byproducts.push(compound),
            Err(error) => {
                return CalculateWeighingOutput {
                    ok: false,
                    target_formula: target.formula,
                    target_mass_g: args.target_mass_g,
                    results: Vec::new(),
                    gas_reactants: Vec::new(),
                    volatile_byproducts: Vec::new(),
                    reaction_equation: None,
                    error: Some(format!("揮発副生成物 {formula} が不正です: {error}")),
                    suggestions: Vec::new(),
                };
            }
        }
    }

    match calculate_weighing_with_optional_volatiles(
        &target,
        args.target_mass_g,
        &precursors,
        volatile_byproducts,
    ) {
        Ok(calculation) => CalculateWeighingOutput {
            ok: true,
            target_formula: target.formula,
            target_mass_g: args.target_mass_g,
            results: calculation.results,
            gas_reactants: calculation.gas_reactants,
            volatile_byproducts: calculation.volatile_byproducts,
            reaction_equation: Some(calculation.reaction_equation),
            error: None,
            suggestions: Vec::new(),
        },
        Err(error) => CalculateWeighingOutput {
            ok: false,
            target_formula: target.formula.clone(),
            target_mass_g: args.target_mass_g,
            results: Vec::new(),
            gas_reactants: Vec::new(),
            volatile_byproducts: Vec::new(),
            reaction_equation: None,
            error: Some(explain_calculation_error(&error, &target, &precursors)),
            suggestions: suggest_precursor_fixes(&target, &precursors),
        },
    }
}

fn parse_compound(formula: &str) -> Result<Compound> {
    let formula = formula.trim();
    if formula.is_empty() {
        bail!("空の化学式です");
    }
    if formula.chars().any(is_hydrate_separator) {
        bail!("水和物やドット区切りの式は未対応です");
    }

    let chars: Vec<char> = formula.chars().collect();
    let mut parser = FormulaParser {
        chars: &chars,
        index: 0,
    };
    let atoms = parser.parse_group(None)?;
    if parser.index != chars.len() {
        bail!("読み取れない文字があります");
    }
    if atoms.is_empty() {
        bail!("元素が含まれていません");
    }

    let molar_mass = molar_mass(&atoms)?;
    Ok(Compound {
        formula: formula.to_string(),
        atoms,
        molar_mass,
    })
}

struct FormulaParser<'a> {
    chars: &'a [char],
    index: usize,
}

impl FormulaParser<'_> {
    fn parse_group(&mut self, stop_on_close: Option<char>) -> Result<BTreeMap<String, f64>> {
        let mut atoms = BTreeMap::new();

        while self.index < self.chars.len() {
            match self.chars[self.index] {
                '(' | '[' | '{' => {
                    let open = self.chars[self.index];
                    let close = matching_close_bracket(open);
                    self.index += 1;
                    let group = self.parse_group(Some(close))?;
                    let multiplier = self.parse_number()?;
                    add_scaled(&mut atoms, &group, multiplier);
                }
                ')' | ']' | '}' => {
                    if Some(self.chars[self.index]) == stop_on_close {
                        self.index += 1;
                        return Ok(atoms);
                    }
                    bail!("閉じ括弧の対応が不正です");
                }
                ch if ch.is_ascii_uppercase() => {
                    let element = self.parse_element();
                    if atomic_weight(&element).is_none() {
                        bail!("未知の元素記号です: {element}");
                    }
                    let count = self.parse_number()?;
                    *atoms.entry(element).or_insert(0.0) += count;
                }
                ch => bail!("化学式として読めない文字です: {ch}"),
            }
        }

        if stop_on_close.is_some() {
            bail!("括弧が閉じていません");
        }
        Ok(atoms)
    }

    fn parse_element(&mut self) -> String {
        let mut element = String::new();
        element.push(self.chars[self.index]);
        self.index += 1;

        if self.index < self.chars.len() && self.chars[self.index].is_ascii_lowercase() {
            element.push(self.chars[self.index]);
            self.index += 1;
        }

        element
    }

    fn parse_number(&mut self) -> Result<f64> {
        let numerator = self.parse_decimal_number()?;
        if self.index < self.chars.len() && self.chars[self.index] == '/' {
            let Some(numerator) = numerator else {
                return Ok(1.0);
            };
            self.index += 1;
            let Some(denominator) = self.parse_decimal_number()? else {
                bail!("分数の分母がありません");
            };
            if denominator <= 0.0 {
                bail!("分数の分母は0より大きい必要があります");
            }
            return Ok(numerator / denominator);
        }

        Ok(numerator.unwrap_or(1.0))
    }

    fn parse_decimal_number(&mut self) -> Result<Option<f64>> {
        let start = self.index;
        let mut dots = 0;

        while self.index < self.chars.len() {
            let ch = self.chars[self.index];
            if ch.is_ascii_digit() {
                self.index += 1;
            } else if ch == '.' {
                let previous_is_digit =
                    self.index > start && self.chars[self.index - 1].is_ascii_digit();
                let next_is_digit = self
                    .chars
                    .get(self.index + 1)
                    .is_some_and(|next| next.is_ascii_digit());
                if !previous_is_digit || !next_is_digit {
                    break;
                }
                dots += 1;
                if dots > 1 {
                    bail!("数値の小数点が多すぎます");
                }
                self.index += 1;
            } else {
                break;
            }
        }

        if start == self.index {
            return Ok(None);
        }

        let value: f64 = self.chars[start..self.index]
            .iter()
            .collect::<String>()
            .parse()?;
        if !value.is_finite() || value <= 0.0 {
            bail!("元素数は0より大きい必要があります");
        }
        Ok(Some(value))
    }
}

fn matching_close_bracket(open: char) -> char {
    match open {
        '(' => ')',
        '[' => ']',
        '{' => '}',
        _ => unreachable!("open bracket must be one of (, [, {{"),
    }
}

fn is_hydrate_separator(ch: char) -> bool {
    matches!(ch, '・' | '·' | '•')
}

fn add_scaled(target: &mut BTreeMap<String, f64>, source: &BTreeMap<String, f64>, scale: f64) {
    for (element, count) in source {
        *target.entry(element.clone()).or_insert(0.0) += count * scale;
    }
}

fn is_inorganic_material_like(compound: &Compound) -> bool {
    !(compound.atoms.contains_key("C")
        && compound.atoms.contains_key("H")
        && compound.atoms.keys().all(|element| {
            matches!(
                element.as_str(),
                "C" | "H" | "O" | "N" | "S" | "P" | "Cl" | "Br" | "I"
            )
        }))
}

fn is_gas_reactant(compound: &Compound) -> bool {
    matches!(
        compound.formula.as_str(),
        "H2" | "N2" | "O2" | "F2" | "Cl2" | "He" | "Ne" | "Ar" | "Kr" | "Xe" | "Rn"
    )
}

fn calculate_weighing(
    target: &Compound,
    target_mass_g: f64,
    precursors: &[Compound],
) -> Result<Vec<WeighingResult>, CalculationError> {
    let target_moles = target_mass_g / target.molar_mass;
    let mut elements = BTreeSet::new();
    for element in target.atoms.keys() {
        elements.insert(element.clone());
    }
    for precursor in precursors {
        for element in precursor.atoms.keys() {
            elements.insert(element.clone());
        }
    }

    let mut matrix = Vec::new();
    for element in elements {
        let mut row = Vec::new();
        for precursor in precursors {
            row.push(*precursor.atoms.get(&element).unwrap_or(&0.0));
        }
        row.push(target.atoms.get(&element).copied().unwrap_or(0.0) * target_moles);
        matrix.push(row);
    }

    let solution = solve_unique_linear_system(matrix, precursors.len())?;
    let mut results = Vec::new();
    for (compound, moles) in precursors.iter().zip(solution) {
        if moles < -EPS {
            return Err(CalculationError::NegativeAmount {
                formula: compound.formula.clone(),
                moles,
            });
        }
        let moles = if moles.abs() < EPS { 0.0 } else { moles };
        results.push(WeighingResult {
            formula: compound.formula.clone(),
            moles,
            grams: moles * compound.molar_mass,
        });
    }

    Ok(results)
}

fn split_gas_reactants(
    precursors: &[Compound],
    precursor_results: Vec<WeighingResult>,
) -> (Vec<WeighingResult>, Vec<WeighingResult>) {
    let mut weighed_results = Vec::new();
    let mut gas_reactants = Vec::new();

    for (compound, result) in precursors.iter().zip(precursor_results) {
        if is_gas_reactant(compound) && result.moles > 0.0 {
            gas_reactants.push(result);
        } else {
            weighed_results.push(result);
        }
    }

    (weighed_results, gas_reactants)
}

fn calculate_weighing_with_optional_volatiles(
    target: &Compound,
    target_mass_g: f64,
    precursors: &[Compound],
    volatile_byproducts: Vec<Compound>,
) -> Result<WeighingCalculation, CalculationError> {
    if volatile_byproducts.is_empty() {
        let precursor_results = calculate_weighing(target, target_mass_g, precursors)?;
        let target_moles = target_mass_g / target.molar_mass;
        let reaction_equation = format_reaction_equation(
            precursors,
            &precursor_results,
            target,
            target_moles,
            &[],
            &[],
        );
        let (results, gas_reactants) = split_gas_reactants(precursors, precursor_results);
        Ok(WeighingCalculation {
            results,
            gas_reactants,
            volatile_byproducts: Vec::new(),
            reaction_equation,
        })
    } else {
        calculate_weighing_with_volatile_byproducts(
            target,
            target_mass_g,
            precursors,
            volatile_byproducts,
        )
    }
}

fn calculate_weighing_with_volatile_byproducts(
    target: &Compound,
    target_mass_g: f64,
    precursors: &[Compound],
    volatile_byproducts: Vec<Compound>,
) -> Result<WeighingCalculation, CalculationError> {
    let target_moles = target_mass_g / target.molar_mass;
    let unknown_count = precursors.len() + volatile_byproducts.len();
    let mut elements = BTreeSet::new();

    for element in target.atoms.keys() {
        elements.insert(element.clone());
    }
    for precursor in precursors {
        for element in precursor.atoms.keys() {
            elements.insert(element.clone());
        }
    }
    for byproduct in &volatile_byproducts {
        for element in byproduct.atoms.keys() {
            elements.insert(element.clone());
        }
    }

    let mut matrix = Vec::new();
    for element in elements {
        let mut row = Vec::new();
        for precursor in precursors {
            row.push(*precursor.atoms.get(&element).unwrap_or(&0.0));
        }
        for byproduct in &volatile_byproducts {
            row.push(-byproduct.atoms.get(&element).copied().unwrap_or(0.0));
        }
        row.push(target.atoms.get(&element).copied().unwrap_or(0.0) * target_moles);
        matrix.push(row);
    }

    let solution = solve_unique_linear_system(matrix, unknown_count)?;
    let (precursor_moles, byproduct_moles) = solution.split_at(precursors.len());

    let mut precursor_results = Vec::new();
    for (compound, moles) in precursors.iter().zip(precursor_moles.iter().copied()) {
        if moles < -EPS {
            return Err(CalculationError::NegativeAmount {
                formula: compound.formula.clone(),
                moles,
            });
        }
        let moles = if moles.abs() < EPS { 0.0 } else { moles };
        precursor_results.push(WeighingResult {
            formula: compound.formula.clone(),
            moles,
            grams: moles * compound.molar_mass,
        });
    }

    let mut volatile_results = Vec::new();
    for (compound, moles) in volatile_byproducts
        .iter()
        .zip(byproduct_moles.iter().copied())
    {
        if moles < -EPS {
            return Err(CalculationError::NegativeAmount {
                formula: compound.formula.clone(),
                moles,
            });
        }
        let moles = if moles.abs() < EPS { 0.0 } else { moles };
        if moles > 0.0 {
            volatile_results.push(WeighingResult {
                formula: compound.formula.clone(),
                moles,
                grams: moles * compound.molar_mass,
            });
        }
    }

    let reaction_equation = format_reaction_equation(
        precursors,
        &precursor_results,
        target,
        target_moles,
        &volatile_byproducts,
        &volatile_results,
    );
    let (results, gas_reactants) = split_gas_reactants(precursors, precursor_results);

    Ok(WeighingCalculation {
        reaction_equation,
        results,
        gas_reactants,
        volatile_byproducts: volatile_results,
    })
}

fn format_reaction_equation(
    precursors: &[Compound],
    precursor_results: &[WeighingResult],
    target: &Compound,
    target_moles: f64,
    volatile_byproducts: &[Compound],
    volatile_results: &[WeighingResult],
) -> String {
    let reactants = precursors
        .iter()
        .zip(precursor_results)
        .map(|(compound, result)| {
            format_reaction_term(result.moles / target_moles, &compound.formula)
        })
        .collect::<Vec<_>>()
        .join(" + ");

    let mut products = vec![format_reaction_term(1.0, &target.formula)];
    products.extend(
        volatile_byproducts
            .iter()
            .zip(volatile_results)
            .filter(|(_, result)| result.moles > 0.0)
            .map(|(compound, result)| {
                format_reaction_term(result.moles / target_moles, &compound.formula)
            }),
    );

    format!("{reactants} -> {}", products.join(" + "))
}

fn format_reaction_term(coefficient: f64, formula: &str) -> String {
    if (coefficient - 1.0).abs() < 1.0e-7 {
        formula.to_string()
    } else {
        format!("{} {}", format_coefficient(coefficient), formula)
    }
}

fn format_coefficient(value: f64) -> String {
    let rounded_integer = value.round();
    if (value - rounded_integer).abs() < 1.0e-7 {
        return format!("{rounded_integer:.0}");
    }

    let text = format!("{value:.6}");
    text.trim_end_matches('0').trim_end_matches('.').to_string()
}

fn solve_unique_linear_system(
    mut matrix: Vec<Vec<f64>>,
    unknown_count: usize,
) -> Result<Vec<f64>, CalculationError> {
    let row_count = matrix.len();
    let mut pivot_cols = Vec::new();
    let mut row = 0;

    for col in 0..unknown_count {
        let pivot = (row..row_count)
            .max_by(|&a, &b| matrix[a][col].abs().total_cmp(&matrix[b][col].abs()))
            .filter(|&candidate| matrix[candidate][col].abs() > EPS);

        let Some(pivot) = pivot else {
            continue;
        };

        matrix.swap(row, pivot);
        let divisor = matrix[row][col];
        for value in &mut matrix[row][col..=unknown_count] {
            *value /= divisor;
        }

        for other_row in 0..row_count {
            if other_row == row {
                continue;
            }
            let factor = matrix[other_row][col];
            if factor.abs() <= EPS {
                continue;
            }
            for current_col in col..=unknown_count {
                matrix[other_row][current_col] -= factor * matrix[row][current_col];
            }
        }

        pivot_cols.push(col);
        row += 1;
        if row == row_count {
            break;
        }
    }

    for values in &matrix {
        let all_zero = values[..unknown_count]
            .iter()
            .all(|value| value.abs() <= EPS);
        if all_zero && values[unknown_count].abs() > EPS {
            return Err(CalculationError::Inconsistent);
        }
    }

    if pivot_cols.len() < unknown_count {
        return Err(CalculationError::NotUnique);
    }

    let mut solution = vec![0.0; unknown_count];
    for (row_index, col) in pivot_cols.into_iter().enumerate() {
        solution[col] = matrix[row_index][unknown_count];
    }

    Ok(solution)
}

fn explain_calculation_error(
    error: &CalculationError,
    target: &Compound,
    precursors: &[Compound],
) -> String {
    let base = match error {
        CalculationError::Inconsistent => {
            "元素収支が一致しません。目的組成に必要な元素が不足しているか、余分な元素が原料に含まれています。"
                .to_string()
        }
        CalculationError::NotUnique => {
            "原料の組み合わせから一意な秤量値を決められません。原料を減らすか、独立した元素源を追加してください。"
                .to_string()
        }
        CalculationError::NegativeAmount { formula, moles } => format!(
            "{formula} が負の使用量 ({moles:.6} mol) になりました。原料の組み合わせが目的組成に合っていません。"
        ),
    };

    let suggestions = suggest_precursor_fixes(target, precursors);
    if suggestions.is_empty() {
        base
    } else {
        format!("{base}\nもしかして {} では？", suggestions.join(", "))
    }
}

fn suggest_precursor_fixes(target: &Compound, precursors: &[Compound]) -> Vec<String> {
    let precursor_elements: BTreeSet<&str> = precursors
        .iter()
        .flat_map(|compound| compound.atoms.keys().map(String::as_str))
        .collect();
    let mut suggestions = Vec::new();

    for element in target.atoms.keys() {
        if !precursor_elements.contains(element.as_str()) {
            suggestions.extend(suggest_sources_for_element(element, target));
        }
    }

    if target.atoms.contains_key("S")
        && precursors
            .iter()
            .any(|compound| compound.atoms.contains_key("O"))
    {
        suggestions.push("酸化物ではなく硫化物原料".to_string());
    }
    if target.atoms.contains_key("O")
        && precursors
            .iter()
            .any(|compound| compound.atoms.contains_key("S"))
    {
        suggestions.push("硫化物ではなく酸化物原料".to_string());
    }

    suggestions.sort();
    suggestions.dedup();
    suggestions
}

fn suggest_sources_for_element(element: &str, target: &Compound) -> Vec<String> {
    let has_s = target.atoms.contains_key("S");
    let has_o = target.atoms.contains_key("O");
    match element {
        "Li" if has_s => vec!["Li2S".to_string()],
        "Li" if has_o => vec!["Li2O".to_string(), "Li2CO3".to_string()],
        "Na" if has_s => vec!["Na2S".to_string()],
        "Na" if has_o => vec!["Na2O".to_string(), "Na2CO3".to_string()],
        "K" if has_s => vec!["K2S".to_string()],
        "K" if has_o => vec!["K2O".to_string(), "K2CO3".to_string()],
        "P" if has_s => vec!["P2S5".to_string()],
        "P" if has_o => vec!["P2O5".to_string(), "NH4H2PO4".to_string()],
        "Si" if has_o => vec!["SiO2".to_string()],
        "Ge" if has_s => vec!["GeS2".to_string()],
        "Ge" if has_o => vec!["GeO2".to_string()],
        "S" => vec!["硫化物原料".to_string()],
        "O" => vec!["酸化物原料".to_string()],
        _ => vec![format!("{element}を含む原料")],
    }
}

fn molar_mass(atoms: &BTreeMap<String, f64>) -> Result<f64> {
    let mut mass = 0.0;
    for (element, count) in atoms {
        let Some(weight) = atomic_weight(element) else {
            bail!("未知の元素記号です: {element}");
        };
        mass += weight * count;
    }
    Ok(mass)
}

fn atomic_weight(element: &str) -> Option<f64> {
    Some(match element {
        "H" => 1.008,
        "He" => 4.002602,
        "Li" => 6.94,
        "Be" => 9.0121831,
        "B" => 10.81,
        "C" => 12.011,
        "N" => 14.007,
        "O" => 15.999,
        "F" => 18.998403163,
        "Ne" => 20.1797,
        "Na" => 22.98976928,
        "Mg" => 24.305,
        "Al" => 26.9815385,
        "Si" => 28.085,
        "P" => 30.973761998,
        "S" => 32.06,
        "Cl" => 35.45,
        "Ar" => 39.948,
        "K" => 39.0983,
        "Ca" => 40.078,
        "Sc" => 44.955908,
        "Ti" => 47.867,
        "V" => 50.9415,
        "Cr" => 51.9961,
        "Mn" => 54.938044,
        "Fe" => 55.845,
        "Co" => 58.933194,
        "Ni" => 58.6934,
        "Cu" => 63.546,
        "Zn" => 65.38,
        "Ga" => 69.723,
        "Ge" => 72.63,
        "As" => 74.921595,
        "Se" => 78.971,
        "Br" => 79.904,
        "Kr" => 83.798,
        "Rb" => 85.4678,
        "Sr" => 87.62,
        "Y" => 88.90584,
        "Zr" => 91.224,
        "Nb" => 92.90637,
        "Mo" => 95.95,
        "Ru" => 101.07,
        "Rh" => 102.9055,
        "Pd" => 106.42,
        "Ag" => 107.8682,
        "Cd" => 112.414,
        "In" => 114.818,
        "Sn" => 118.71,
        "Sb" => 121.76,
        "Te" => 127.6,
        "I" => 126.90447,
        "Cs" => 132.90545196,
        "Ba" => 137.327,
        "La" => 138.90547,
        "Ce" => 140.116,
        "Pr" => 140.90766,
        "Nd" => 144.242,
        "Sm" => 150.36,
        "Eu" => 151.964,
        "Gd" => 157.25,
        "Tb" => 158.92535,
        "Dy" => 162.5,
        "Ho" => 164.93033,
        "Er" => 167.259,
        "Tm" => 168.93422,
        "Yb" => 173.045,
        "Lu" => 174.9668,
        "Hf" => 178.49,
        "Ta" => 180.94788,
        "W" => 183.84,
        "Re" => 186.207,
        "Os" => 190.23,
        "Ir" => 192.217,
        "Pt" => 195.084,
        "Au" => 196.966569,
        "Hg" => 200.592,
        "Tl" => 204.38,
        "Pb" => 207.2,
        "Bi" => 208.9804,
        "Th" => 232.0377,
        "U" => 238.02891,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_formula_with_parentheses() {
        let compound = parse_compound("Ca(OH)2").unwrap();

        assert_eq!(compound.atoms["Ca"], 1.0);
        assert_eq!(compound.atoms["O"], 2.0);
        assert_eq!(compound.atoms["H"], 2.0);
    }

    #[test]
    fn parses_ollama_num_ctx_setting() {
        assert_eq!(parse_ollama_num_ctx(None).unwrap(), DEFAULT_OLLAMA_NUM_CTX);
        assert_eq!(
            parse_ollama_num_ctx(Some(" 16384 ".to_string())).unwrap(),
            16384
        );
        assert!(parse_ollama_num_ctx(Some("0".to_string())).is_err());
        assert!(parse_ollama_num_ctx(Some("large".to_string())).is_err());
    }

    #[test]
    fn detects_model_thinking_modes() {
        for model in [
            "qwen3.6:35b",
            "QWEN3:8b",
            "deepseek-r1:8b",
            "deepseek-v3.2:latest",
            "glm-5.1:latest",
            "minimax-m3:latest",
            "kimi-k2.6:latest",
        ] {
            assert_eq!(thinking_mode_for_model(model), ThinkingMode::OllamaBool);
        }

        assert_eq!(
            thinking_mode_for_model("gpt-oss:20b"),
            ThinkingMode::OllamaLevel("medium")
        );
        assert_eq!(
            thinking_mode_for_model("gemma4:26b"),
            ThinkingMode::PromptToken
        );
        assert_eq!(
            thinking_mode_for_model("gemma-4:26b"),
            ThinkingMode::PromptToken
        );
        assert_eq!(
            thinking_mode_for_model("llama3.2:latest"),
            ThinkingMode::Disabled
        );
    }

    #[test]
    fn builds_model_specific_agent_prompt_config() {
        let qwen = build_agent_prompt_config("qwen3.6:35b", 4096);
        assert_eq!(qwen.thinking_mode, ThinkingMode::OllamaBool);
        assert_eq!(qwen.params, json!({ "think": true, "num_ctx": 4096 }));
        assert_eq!(qwen.preamble, AGENT_PREAMBLE);

        let gpt_oss = build_agent_prompt_config("gpt-oss:20b", 4096);
        assert_eq!(gpt_oss.thinking_mode, ThinkingMode::OllamaLevel("medium"));
        assert_eq!(
            gpt_oss.params,
            json!({ "think": "medium", "num_ctx": 4096 })
        );
        assert_eq!(gpt_oss.preamble, AGENT_PREAMBLE);

        let gemma = build_agent_prompt_config("gemma-4:26b", 4096);
        assert_eq!(gemma.thinking_mode, ThinkingMode::PromptToken);
        assert_eq!(gemma.params, json!({ "num_ctx": 4096 }));
        assert!(gemma.preamble.starts_with("<|think|>\n"));
        assert!(gemma.preamble.ends_with(AGENT_PREAMBLE));

        let disabled = build_agent_prompt_config("llama3.2:latest", 4096);
        assert_eq!(disabled.thinking_mode, ThinkingMode::Disabled);
        assert_eq!(disabled.params, json!({ "num_ctx": 4096 }));
        assert_eq!(disabled.preamble, AGENT_PREAMBLE);
    }

    #[test]
    fn parses_formula_with_decimal_stoichiometry_and_brackets() {
        let compound = parse_compound("(Li0.5La0.5)TiO3").unwrap();

        assert_eq!(compound.atoms["Li"], 0.5);
        assert_eq!(compound.atoms["La"], 0.5);
        assert_eq!(compound.atoms["Ti"], 1.0);
        assert_eq!(compound.atoms["O"], 3.0);

        let bracketed = parse_compound("Ca[OH]2").unwrap();
        assert_eq!(bracketed.atoms["Ca"], 1.0);
        assert_eq!(bracketed.atoms["O"], 2.0);
        assert_eq!(bracketed.atoms["H"], 2.0);
    }

    #[test]
    fn parses_formula_with_fractional_stoichiometry() {
        let compound = parse_compound("LiNi1/3Co1/3Mn1/3O2").unwrap();

        assert_eq!(compound.atoms["Li"], 1.0);
        assert!((compound.atoms["Ni"] - 1.0 / 3.0).abs() < 1.0e-12);
        assert!((compound.atoms["Co"] - 1.0 / 3.0).abs() < 1.0e-12);
        assert!((compound.atoms["Mn"] - 1.0 / 3.0).abs() < 1.0e-12);
        assert_eq!(compound.atoms["O"], 2.0);
    }

    #[test]
    fn guesses_common_ambiguous_material_names() {
        assert_eq!(guess_material_formula("LFP").as_deref(), Some("LiFePO4"));
        assert_eq!(
            guess_material_formula("NMC111").as_deref(),
            Some("LiNi1/3Co1/3Mn1/3O2")
        );
        assert_eq!(
            guess_material_formula("LATP").as_deref(),
            Some("Li1.3Al0.3Ti1.7(PO4)3")
        );
        assert_eq!(guess_material_formula("unknown"), None);
    }

    #[test]
    fn calculates_decimal_composition_precursor_masses() {
        let target = parse_compound("Li0.5La0.5TiO3").unwrap();
        let precursors = ["Li2O", "La2O3", "TiO2"]
            .into_iter()
            .map(parse_compound)
            .collect::<Result<Vec<_>>>()
            .unwrap();

        let results = calculate_weighing(&target, 1.0, &precursors).unwrap();

        assert_eq!(results.len(), 3);
        assert!((results[0].grams - 0.044256).abs() < 1.0e-5);
        assert!((results[1].grams - 0.482573).abs() < 1.0e-5);
        assert!((results[2].grams - 0.473171).abs() < 1.0e-5);
    }

    #[test]
    fn calculates_with_common_volatile_byproducts() {
        let target = parse_compound("Li1.3Al0.3Ti1.7(PO4)3").unwrap();
        let precursors = ["Li2CO3", "Al2O3", "TiO2", "NH4H2PO4"]
            .into_iter()
            .map(parse_compound)
            .collect::<Result<Vec<_>>>()
            .unwrap();

        assert_eq!(
            calculate_weighing(&target, 1.0, &precursors).unwrap_err(),
            CalculationError::Inconsistent
        );

        assert!(
            calculate_weighing_with_optional_volatiles(&target, 1.0, &precursors, Vec::new())
                .is_err()
        );

        let volatile_byproducts = ["CO2", "H2O", "NH3"]
            .into_iter()
            .map(parse_compound)
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let calculation = calculate_weighing_with_optional_volatiles(
            &target,
            1.0,
            &precursors,
            volatile_byproducts,
        )
        .unwrap();

        assert_eq!(calculation.results.len(), 4);
        assert!((calculation.results[0].grams - 0.125267).abs() < 1.0e-5);
        assert!((calculation.results[1].grams - 0.039891).abs() < 1.0e-5);
        assert!((calculation.results[2].grams - 0.354123).abs() < 1.0e-5);
        assert!((calculation.results[3].grams - 0.900038).abs() < 1.0e-5);

        assert_eq!(calculation.volatile_byproducts.len(), 3);
        assert_eq!(calculation.volatile_byproducts[0].formula, "CO2");
        assert_eq!(calculation.volatile_byproducts[1].formula, "H2O");
        assert_eq!(calculation.volatile_byproducts[2].formula, "NH3");
        assert_eq!(
            calculation.reaction_equation,
            "0.65 Li2CO3 + 0.15 Al2O3 + 1.7 TiO2 + 3 NH4H2PO4 -> Li1.3Al0.3Ti1.7(PO4)3 + 0.65 CO2 + 4.5 H2O + 3 NH3"
        );
    }

    #[test]
    fn calculates_with_gas_reactant_excluded_from_weighing_results() {
        let target = parse_compound("LaCoO3").unwrap();
        let precursors = ["La2O3", "Co3O4", "O2"]
            .into_iter()
            .map(parse_compound)
            .collect::<Result<Vec<_>>>()
            .unwrap();

        let calculation =
            calculate_weighing_with_optional_volatiles(&target, 10.0, &precursors, Vec::new())
                .unwrap();

        assert_eq!(calculation.results.len(), 2);
        assert_eq!(calculation.results[0].formula, "La2O3");
        assert_eq!(calculation.results[1].formula, "Co3O4");
        assert!((calculation.results[0].grams - 6.6265).abs() < 1.0e-4);
        assert!((calculation.results[1].grams - 3.2650).abs() < 1.0e-4);

        assert_eq!(calculation.gas_reactants.len(), 1);
        assert_eq!(calculation.gas_reactants[0].formula, "O2");
        assert!((calculation.gas_reactants[0].moles - 0.0033898).abs() < 1.0e-7);
        assert_eq!(
            calculation.reaction_equation,
            "0.5 La2O3 + 0.333333 Co3O4 + 0.083333 O2 -> LaCoO3"
        );
    }

    #[test]
    fn calculates_lgps_precursor_masses() {
        let target = parse_compound("Li10GeP2S12").unwrap();
        let precursors = ["Li2S", "GeS2", "P2S5"]
            .into_iter()
            .map(parse_compound)
            .collect::<Result<Vec<_>>>()
            .unwrap();

        let results = calculate_weighing(&target, 1.0, &precursors).unwrap();

        assert_eq!(results.len(), 3);
        assert!((results[0].grams - 0.390183).abs() < 1.0e-5);
        assert!((results[1].grams - 0.232292).abs() < 1.0e-5);
        assert!((results[2].grams - 0.377524).abs() < 1.0e-5);
    }

    #[test]
    fn rejects_inconsistent_precursors() {
        let target = parse_compound("Li10GeP2S12").unwrap();
        let precursors = ["Li2O", "GeO2", "P2O5"]
            .into_iter()
            .map(parse_compound)
            .collect::<Result<Vec<_>>>()
            .unwrap();

        let error = calculate_weighing(&target, 1.0, &precursors).unwrap_err();

        assert_eq!(error, CalculationError::Inconsistent);
    }

    #[test]
    fn detects_non_unique_precursor_set() {
        let target = parse_compound("Li2O").unwrap();
        let precursors = ["Li2O", "Li2O"]
            .into_iter()
            .map(parse_compound)
            .collect::<Result<Vec<_>>>()
            .unwrap();

        let error = calculate_weighing(&target, 1.0, &precursors).unwrap_err();

        assert_eq!(error, CalculationError::NotUnique);
    }

    #[test]
    fn accepts_elemental_precursors() {
        let target = parse_compound("FeS").unwrap();
        let precursors = ["Fe", "S"]
            .into_iter()
            .map(parse_compound)
            .collect::<Result<Vec<_>>>()
            .unwrap();

        let results = calculate_weighing(&target, 1.0, &precursors).unwrap();

        assert!((results[0].grams - 0.635288).abs() < 1.0e-5);
        assert!((results[1].grams - 0.364712).abs() < 1.0e-5);
    }
}
