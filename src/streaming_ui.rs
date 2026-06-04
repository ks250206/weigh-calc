use anyhow::Result;
use crossterm::event::{self, Event, KeyCode};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use futures::StreamExt;
use rig::agent::{MultiTurnStreamItem, StreamingResult};
use rig::completion::Message;
use rig::message::{ReasoningContent, ToolResult, ToolResultContent};
use rig::streaming::{StreamedAssistantContent, StreamedUserContent};
use serde::Deserialize;
use std::collections::BTreeSet;
use std::io::{self, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use tokio::select;
use tokio::sync::watch;
use tokio::time::{Duration, Instant, sleep, timeout};

use crate::prompt::END_MARKER;

const STREAM_CHUNK_TIMEOUT_SECS: u64 = 30;
const STREAM_TOTAL_TIMEOUT_SECS: u64 = 120;
const DETERMINISTIC_OUTPUT_CHUNK_CHARS: usize = 4;
const DETERMINISTIC_OUTPUT_CHUNK_DELAY_MS: u64 = 8;

pub(crate) struct AgentStreamPrintResult {
    pub(crate) should_end: bool,
    pub(crate) updated_history: Option<Vec<Message>>,
    pub(crate) response_text: String,
    pub(crate) printed_deterministic_result: bool,
    pub(crate) pending_volatile_result: Option<String>,
    pub(crate) pending_volatile_calculation_record: Option<String>,
    pub(crate) completed_calculation_record: Option<String>,
}

pub(crate) async fn print_streaming_agent_response<R>(
    mut stream: StreamingResult<R>,
) -> Result<AgentStreamPrintResult>
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
    let debug_thinking = debug_thinking_enabled(std::env::var("WEIGH_CALC_DEBUG_THINKING").ok());
    let mut suppress_duplicate_reaction_line = false;
    let mut duplicate_reaction_line_buffer = String::new();
    let mut updated_history = None;
    let started_at = Instant::now();
    let escape_listener = EscapeListener::start();
    let raw_output = escape_listener.raw_mode_enabled();
    let mut escape_rx = escape_listener.subscribe();

    loop {
        if started_at.elapsed() > Duration::from_secs(STREAM_TOTAL_TIMEOUT_SECS) {
            clear_thinking_indicator(&mut showing_thinking, debug_thinking, raw_output)?;
            print_terminal(
                "応答が長時間完了しなかったため中断しました。もう一度、化学式のリストを具体的に入力してください。\n",
                raw_output,
            )?;
            return Ok(AgentStreamPrintResult {
                should_end: false,
                updated_history,
                response_text,
                printed_deterministic_result: false,
                pending_volatile_result: None,
                pending_volatile_calculation_record: None,
                completed_calculation_record: None,
            });
        }

        let Some(chunk) = (select! {
            chunk = timeout(Duration::from_secs(STREAM_CHUNK_TIMEOUT_SECS), stream.next()) => {
                match chunk {
                    Ok(chunk) => chunk,
                    Err(_) => {
                        clear_thinking_indicator(&mut showing_thinking, debug_thinking, raw_output)?;
                        print_terminal(
                            "応答が停止したため中断しました。もう一度、化学式のリストを具体的に入力してください。\n",
                            raw_output,
                        )?;
                        return Ok(AgentStreamPrintResult {
                            should_end: false,
                            updated_history,
                            response_text,
                            printed_deterministic_result: false,
                            pending_volatile_result: None,
                            pending_volatile_calculation_record: None,
                            completed_calculation_record: None,
                        });
                    }
                }
            }
            _ = escape_rx.changed() => {
                clear_thinking_indicator(&mut showing_thinking, debug_thinking, raw_output)?;
                print_terminal(
                    "Esc が押されたため thinking/応答生成を中断しました。\n",
                    raw_output,
                )?;
                return Ok(AgentStreamPrintResult {
                    should_end: false,
                    updated_history,
                    response_text,
                    printed_deterministic_result: false,
                    pending_volatile_result: None,
                    pending_volatile_calculation_record: None,
                    completed_calculation_record: None,
                });
            }
        }) else {
            break;
        };

        let item = match chunk {
            Ok(item) => item,
            Err(error) => {
                clear_thinking_indicator(&mut showing_thinking, debug_thinking, raw_output)?;
                print_terminal(
                    &format!("{}\n", stream_error_message(&error.to_string())),
                    raw_output,
                )?;
                return Ok(AgentStreamPrintResult {
                    should_end: false,
                    updated_history,
                    response_text,
                    printed_deterministic_result: false,
                    pending_volatile_result: None,
                    pending_volatile_calculation_record: None,
                    completed_calculation_record: None,
                });
            }
        };
        match item {
            MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(text)) => {
                clear_thinking_indicator(&mut showing_thinking, debug_thinking, raw_output)?;
                response_text.push_str(&text.text);
                let sanitized = sanitize_assistant_text(&text.text);
                let filtered = filter_duplicate_reaction_line(
                    &sanitized,
                    !printed_reaction_equations.is_empty(),
                    &mut suppress_duplicate_reaction_line,
                    &mut duplicate_reaction_line_buffer,
                );
                pending_text.push_str(&filtered);
                printed_visible_text |=
                    flush_visible_stream_text(&mut pending_text, false, raw_output)?;
            }
            MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Reasoning(
                reasoning,
            )) => {
                update_thinking_output(
                    &reasoning_content_text(&reasoning.content),
                    debug_thinking,
                    raw_output,
                    &mut thinking_char_count,
                    &mut showing_thinking,
                )?;
            }
            MultiTurnStreamItem::StreamAssistantItem(
                StreamedAssistantContent::ReasoningDelta { reasoning, .. },
            ) => {
                update_thinking_output(
                    &reasoning,
                    debug_thinking,
                    raw_output,
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
                clear_thinking_indicator(&mut showing_thinking, debug_thinking, raw_output)?;
                let flushed_text = flush_visible_stream_text(&mut pending_text, true, raw_output)?;
                if flushed_text {
                    print_terminal("\n", raw_output)?;
                }
                printed_visible_text |= flushed_text;
                let tool_print_result = print_calculation_from_tool_result(
                    &tool_result,
                    &mut printed_reaction_equations,
                    raw_output,
                )
                .await?;
                printed_visible_text |= tool_print_result.printed;
                let pending_volatile_result = tool_print_result.pending_volatile_result;
                let pending_volatile_calculation_record =
                    tool_print_result.pending_volatile_calculation_record;
                if tool_print_result.printed_deterministic_result {
                    clear_thinking_indicator(&mut showing_thinking, debug_thinking, raw_output)?;
                    return Ok(AgentStreamPrintResult {
                        should_end: false,
                        updated_history,
                        response_text,
                        printed_deterministic_result: true,
                        pending_volatile_result: None,
                        pending_volatile_calculation_record: None,
                        completed_calculation_record: tool_print_result
                            .completed_calculation_record,
                    });
                }
                if pending_volatile_result.is_some() {
                    clear_thinking_indicator(&mut showing_thinking, debug_thinking, raw_output)?;
                    return Ok(AgentStreamPrintResult {
                        should_end: false,
                        updated_history,
                        response_text,
                        printed_deterministic_result: false,
                        pending_volatile_result,
                        pending_volatile_calculation_record,
                        completed_calculation_record: None,
                    });
                }
            }
            MultiTurnStreamItem::FinalResponse(response) => {
                clear_thinking_indicator(&mut showing_thinking, debug_thinking, raw_output)?;
                should_end = response.response().contains(END_MARKER);
                updated_history = response.history().map(|history| history.to_vec());
            }
            _ => {}
        }
    }

    should_end |= pending_text.contains(END_MARKER);
    clear_thinking_indicator(&mut showing_thinking, debug_thinking, raw_output)?;
    pending_text.push_str(&flush_duplicate_reaction_line_buffer(
        &mut duplicate_reaction_line_buffer,
    ));
    printed_visible_text |= flush_visible_stream_text(&mut pending_text, true, raw_output)?;
    if should_end && !printed_visible_text {
        print_terminal("終了します。\n", raw_output)?;
    } else if printed_visible_text {
        print_terminal("\n", raw_output)?;
    }
    Ok(AgentStreamPrintResult {
        should_end,
        updated_history,
        response_text,
        printed_deterministic_result: false,
        pending_volatile_result: None,
        pending_volatile_calculation_record: None,
        completed_calculation_record: None,
    })
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

struct EscapeListener {
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
    raw_mode_enabled: bool,
    receiver: watch::Receiver<bool>,
}

impl EscapeListener {
    fn start() -> Self {
        let (sender, receiver) = watch::channel(false);
        let raw_mode_enabled = enable_raw_mode().is_ok();
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        let handle = raw_mode_enabled.then(|| {
            thread::spawn(move || {
                while !thread_stop.load(Ordering::Relaxed) {
                    if poll_escape_key() {
                        let _ = sender.send(true);
                        return;
                    }
                    thread::sleep(Duration::from_millis(20));
                }
            })
        });

        Self {
            stop,
            handle,
            raw_mode_enabled,
            receiver,
        }
    }

    fn raw_mode_enabled(&self) -> bool {
        self.raw_mode_enabled
    }

    fn subscribe(&self) -> watch::Receiver<bool> {
        self.receiver.clone()
    }
}

impl Drop for EscapeListener {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        if self.raw_mode_enabled {
            let _ = disable_raw_mode();
        }
    }
}

fn poll_escape_key() -> bool {
    if !event::poll(Duration::from_millis(0)).unwrap_or(false) {
        return false;
    }

    matches!(
        event::read(),
        Ok(Event::Key(key_event)) if key_event.code == KeyCode::Esc
    )
}

fn print_terminal(text: &str, raw_output: bool) -> Result<()> {
    if raw_output {
        print!("{}", text.replace('\n', "\r\n"));
    } else {
        print!("{text}");
    }
    io::stdout().flush()?;
    Ok(())
}

fn reasoning_content_text(content: &[ReasoningContent]) -> String {
    let mut text = String::new();
    for content in content {
        match content {
            ReasoningContent::Text { text: chunk, .. }
            | ReasoningContent::Summary(chunk)
            | ReasoningContent::Encrypted(chunk) => text.push_str(chunk),
            ReasoningContent::Redacted { data } => text.push_str(data),
            _ => {}
        }
    }
    text
}

fn debug_thinking_enabled(value: Option<String>) -> bool {
    value
        .as_deref()
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .iter()
        .any(|value| matches!(value.as_str(), "1" | "true" | "yes" | "on"))
}

fn stream_error_message(error: &str) -> &'static str {
    if error.contains("MaxTurnError") || error.contains("MaxTurnsError") {
        "tool 呼び出しが規定回数内に完了しなかったため中断しました。もう一度、目的組成・目標質量・原料組成を具体的に入力してください。"
    } else if error.contains("ToolCallError") || error.contains("UnknownToolCall") {
        "tool 呼び出しに失敗したため中断しました。もう一度、化学式のリストを具体的に入力してください。"
    } else {
        "応答処理中にエラーが発生したため中断しました。もう一度入力してください。"
    }
}

fn update_thinking_output(
    chunk: &str,
    debug_thinking: bool,
    raw_output: bool,
    thinking_char_count: &mut usize,
    showing_thinking: &mut bool,
) -> Result<()> {
    *thinking_char_count += chunk.chars().count();
    if debug_thinking {
        if !*showing_thinking {
            print_terminal("\x1b[2m[thinking]\x1b[0m\n", raw_output)?;
        }
        print_terminal(chunk, raw_output)?;
    } else {
        print_terminal(
            &format!(
                "\r\x1b[2K\x1b[5mThinking... ({} chars)\x1b[0m",
                *thinking_char_count
            ),
            raw_output,
        )?;
    }
    *showing_thinking = true;
    Ok(())
}

fn clear_thinking_indicator(
    showing_thinking: &mut bool,
    debug_thinking: bool,
    raw_output: bool,
) -> Result<()> {
    if *showing_thinking {
        if debug_thinking {
            print_terminal("\n", raw_output)?;
        } else {
            print_terminal("\r\x1b[2K", raw_output)?;
        }
        *showing_thinking = false;
    }
    Ok(())
}

#[derive(Debug, Default)]
struct ToolPrintResult {
    printed: bool,
    printed_deterministic_result: bool,
    pending_volatile_result: Option<String>,
    pending_volatile_calculation_record: Option<String>,
    completed_calculation_record: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CalculationToolResult {
    ok: bool,
    target_formula: String,
    target_mass_g: f64,
    results: Vec<CalculationItem>,
    gas_reactants: Vec<CalculationItem>,
    volatile_byproducts: Vec<CalculationItem>,
    reaction_equation: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CalculationItem {
    formula: String,
    moles: f64,
    grams: f64,
}

async fn print_calculation_from_tool_result(
    tool_result: &ToolResult,
    printed_reaction_equations: &mut BTreeSet<String>,
    raw_output: bool,
) -> Result<ToolPrintResult> {
    for content in tool_result.content.iter() {
        let ToolResultContent::Text(text) = content else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<CalculationToolResult>(&text.text) else {
            continue;
        };
        if !value.ok {
            continue;
        }
        let Some(equation) = value
            .reaction_equation
            .as_deref()
            .filter(|equation| !equation.is_empty())
        else {
            continue;
        };

        if !value.volatile_byproducts.is_empty() {
            if printed_reaction_equations.insert(equation.to_string()) {
                let volatile_formulas = format_formula_list(&value.volatile_byproducts);
                print_terminal(
                    &format!(
                        "反応式: {equation}\n\nこの反応式では {volatile_formulas} が揮発すると仮定します。それでもこの前提で秤量計算してよいですか？\n"
                    ),
                    raw_output,
                )?;
                return Ok(ToolPrintResult {
                    printed: true,
                    printed_deterministic_result: false,
                    pending_volatile_result: Some(format_deterministic_calculation_result(
                        equation, &value,
                    )),
                    pending_volatile_calculation_record: Some(format_calculation_record(
                        equation, &value,
                    )),
                    completed_calculation_record: None,
                });
            }
            return Ok(ToolPrintResult::default());
        }

        if !printed_reaction_equations.insert(equation.to_string()) {
            return Ok(ToolPrintResult::default());
        }

        stream_text_with_mode(
            &format_deterministic_calculation_result(equation, &value),
            raw_output,
        )
        .await?;
        return Ok(ToolPrintResult {
            printed: true,
            printed_deterministic_result: true,
            pending_volatile_result: None,
            pending_volatile_calculation_record: None,
            completed_calculation_record: Some(format_calculation_record(equation, &value)),
        });
    }

    Ok(ToolPrintResult::default())
}

pub(crate) async fn stream_text(text: &str) -> Result<()> {
    stream_text_with_mode(text, false).await
}

async fn stream_text_with_mode(text: &str, raw_output: bool) -> Result<()> {
    let mut chunk = String::new();
    let mut chunk_chars = 0usize;

    for ch in text.chars() {
        chunk.push(ch);
        chunk_chars += 1;
        if chunk_chars >= DETERMINISTIC_OUTPUT_CHUNK_CHARS || ch == '\n' {
            print_terminal(&chunk, raw_output)?;
            chunk.clear();
            chunk_chars = 0;
            sleep(Duration::from_millis(DETERMINISTIC_OUTPUT_CHUNK_DELAY_MS)).await;
        }
    }

    if !chunk.is_empty() {
        print_terminal(&chunk, raw_output)?;
    }
    Ok(())
}

fn format_deterministic_calculation_result(
    equation: &str,
    value: &CalculationToolResult,
) -> String {
    let mut output = format!("反応式: {equation}\n\n計算が完了しました。\n\n");
    for result in &value.results {
        output.push_str(&format!(
            "- {}: {:.4} g, {:.1} mg, {} mol\n",
            result.formula,
            result.grams,
            result.grams * 1000.0,
            format_significant(result.moles, 4)
        ));
    }
    if !value.gas_reactants.is_empty() {
        let gas_formulas = format_formula_list(&value.gas_reactants);
        output.push_str(&format!(
            "\nガス原料（{gas_formulas}）は秤量対象外ですが、元素収支に含めて計算しました。\n"
        ));
    }
    if !value.volatile_byproducts.is_empty() {
        let volatile_formulas = format_formula_list(&value.volatile_byproducts);
        output.push_str(&format!(
            "\n揮発副生成物（{volatile_formulas}）を仮定して計算しました。\n"
        ));
    }
    output.push_str("\n再度計算しますか？\n");
    output
}

fn format_calculation_record(equation: &str, value: &CalculationToolResult) -> String {
    format!(
        "作製材料: {}; 目標質量_g: {}; 反応式: {}",
        value.target_formula, value.target_mass_g, equation
    )
}

fn format_formula_list(items: &[CalculationItem]) -> String {
    items
        .iter()
        .map(|result| result.formula.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_significant(value: f64, significant_digits: i32) -> String {
    if value == 0.0 {
        return "0".to_string();
    }

    let abs = value.abs();
    let digits_before_decimal = abs.log10().floor() as i32 + 1;
    let decimals = (significant_digits - digits_before_decimal).max(0) as usize;
    let text = format!("{value:.decimals$}");
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        text
    }
}

fn flush_visible_stream_text(
    pending_text: &mut String,
    force: bool,
    raw_output: bool,
) -> Result<bool> {
    if pending_text.is_empty() {
        return Ok(false);
    }

    if let Some(marker_start) = pending_text.find(END_MARKER) {
        let visible = pending_text[..marker_start].to_string();
        let printed = !visible.is_empty();
        if !visible.is_empty() {
            print_terminal(&visible, raw_output)?;
        }
        pending_text.clear();
        return Ok(printed);
    }

    if force {
        let visible = pending_text.replace(END_MARKER, "");
        let printed = !visible.is_empty();
        if !visible.is_empty() {
            print_terminal(&visible, raw_output)?;
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
    print_terminal(&visible, raw_output)?;
    *pending_text = tail;
    Ok(!visible.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_nonvolatile_tool_result_with_weighing_values() {
        let value: CalculationToolResult = serde_json::from_value(serde_json::json!({
            "ok": true,
            "target_formula": "LiCoO2",
            "target_mass_g": 3.0,
            "results": [
                { "formula": "Li2O", "moles": 0.01533, "grams": 0.4579 },
                { "formula": "Co3O4", "moles": 0.01022, "grams": 2.4603 }
            ],
            "gas_reactants": [
                { "formula": "O2", "moles": 0.002555, "grams": 0.08176 }
            ],
            "volatile_byproducts": [],
            "reaction_equation": "0.5 Li2O + 0.333333 Co3O4 + 0.083333 O2 -> LiCoO2"
        }))
        .unwrap();

        let formatted = format_deterministic_calculation_result(
            value.reaction_equation.as_deref().unwrap(),
            &value,
        );

        assert!(formatted.contains("反応式: 0.5 Li2O + 0.333333 Co3O4 + 0.083333 O2 -> LiCoO2"));
        assert!(formatted.contains("計算が完了しました。"));
        assert!(formatted.contains("- Li2O: 0.4579 g, 457.9 mg, 0.01533 mol"));
        assert!(formatted.contains("- Co3O4: 2.4603 g, 2460.3 mg, 0.01022 mol"));
        assert!(
            formatted.contains("ガス原料（O2）は秤量対象外ですが、元素収支に含めて計算しました。")
        );
        assert!(formatted.ends_with("再度計算しますか？\n"));

        let record = format_calculation_record(value.reaction_equation.as_deref().unwrap(), &value);
        assert_eq!(
            record,
            "作製材料: LiCoO2; 目標質量_g: 3; 反応式: 0.5 Li2O + 0.333333 Co3O4 + 0.083333 O2 -> LiCoO2"
        );
    }

    #[test]
    fn parses_debug_thinking_env_values() {
        assert!(debug_thinking_enabled(Some("1".to_string())));
        assert!(debug_thinking_enabled(Some("TRUE".to_string())));
        assert!(debug_thinking_enabled(Some(" yes ".to_string())));
        assert!(debug_thinking_enabled(Some("on".to_string())));

        assert!(!debug_thinking_enabled(None));
        assert!(!debug_thinking_enabled(Some("0".to_string())));
        assert!(!debug_thinking_enabled(Some("false".to_string())));
        assert!(!debug_thinking_enabled(Some("debug".to_string())));
    }
}
