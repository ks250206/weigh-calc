use anyhow::Result;
use futures::StreamExt;
use rig::agent::{MultiTurnStreamItem, StreamingResult};
use rig::completion::Message;
use rig::message::{ReasoningContent, ToolResult, ToolResultContent};
use rig::streaming::{StreamedAssistantContent, StreamedUserContent};
use std::collections::BTreeSet;
use std::io::{self, Write};
use tokio::time::{Duration, Instant, timeout};

use crate::prompt::END_MARKER;

const STREAM_CHUNK_TIMEOUT_SECS: u64 = 90;
const STREAM_TOTAL_TIMEOUT_SECS: u64 = 240;

pub(crate) async fn print_streaming_agent_response<R>(
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
