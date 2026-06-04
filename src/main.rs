use anyhow::Result;
use rig::client::CompletionClient;
use rig::completion::Message;
use rig::providers::ollama;
use rig::streaming::StreamingChat;
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;
use std::io::{self, Write};

mod agent_config;
mod chemistry;
mod conversation;
mod prompt;
mod streaming_ui;
mod tools;

#[cfg(test)]
use agent_config::{DEFAULT_OLLAMA_NUM_CTX, thinking_mode_for_model};
use agent_config::{
    ThinkingMode, ThinkingSetting, build_agent_prompt_config, parse_ollama_num_ctx,
    parse_weigh_calc_thinking,
};
#[cfg(test)]
use chemistry::{
    CalculationError, calculate_weighing, calculate_weighing_with_optional_volatiles,
    parse_compound,
};
use conversation::{
    ConversationState, extract_mass_g, extract_valid_formulas, guess_material_formula,
    is_confirmation_only, references_suggested_formulas,
};
#[cfg(test)]
use prompt::AGENT_PREAMBLE;
#[cfg(test)]
use serde_json::json;
use streaming_ui::{AgentStreamPrintResult, print_streaming_agent_response, stream_text};
use tools::{CalculateWeighingTool, ValidateFormulaTool};

#[tokio::main]
async fn main() -> Result<()> {
    let mut app = WeighingAgentApp::new()?;
    app.run().await
}

struct WeighingAgentApp {
    ollama_base_url: String,
    model: String,
    ollama_num_ctx: u64,
    thinking_setting: ThinkingSetting,
    line_editor: DefaultEditor,
    state: ConversationState,
    last_suggested_formulas: Vec<String>,
    pending_volatile_result: Option<String>,
    pending_volatile_calculation_record: Option<String>,
    last_calculation_summary: Option<String>,
    calculation_records: Vec<String>,
}

impl WeighingAgentApp {
    fn new() -> Result<Self> {
        Ok(Self {
            ollama_base_url: std::env::var("OLLAMA_BASE_URL")
                .unwrap_or_else(|_| "http://localhost:11434".to_string()),
            model: std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen3.6:35b".to_string()),
            ollama_num_ctx: parse_ollama_num_ctx(std::env::var("OLLAMA_NUM_CTX").ok())?,
            thinking_setting: parse_weigh_calc_thinking(std::env::var("WEIGH_CALC_THINKING").ok())?,
            line_editor: DefaultEditor::new()?,
            state: ConversationState::default(),
            last_suggested_formulas: Vec::new(),
            pending_volatile_result: None,
            pending_volatile_calculation_record: None,
            last_calculation_summary: None,
            calculation_records: Vec::new(),
        })
    }

    async fn run(&mut self) -> Result<()> {
        let client = ollama::Client::new(self.ollama_base_url.as_str())?;
        let agent_config =
            build_agent_prompt_config(&self.model, self.ollama_num_ctx, self.thinking_setting);
        if self.thinking_setting == ThinkingSetting::Off {
            println!(
                "注意: WEIGH_CALC_THINKING=off のため thinking を無効化します。Thinking... 表示は出ません。model: {}",
                self.model
            );
        } else if agent_config.thinking_mode == ThinkingMode::Disabled {
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
        let stream_result = print_streaming_agent_response(stream).await?;
        self.update_last_suggestions(&stream_result.response_text);
        if let Some(updated_history) = stream_result.updated_history {
            chat_history = updated_history;
        }

        let mut awaiting_recalculate_answer = false;
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

            if self.pending_volatile_result.is_some() {
                if is_confirmation_only(&input) {
                    if let Some(result_text) = self.pending_volatile_result.take() {
                        stream_text(&result_text).await?;
                        self.update_last_calculation_summary();
                        if let Some(record) = self.pending_volatile_calculation_record.take() {
                            self.add_calculation_record(record);
                        }
                        awaiting_recalculate_answer = true;
                        continue;
                    }
                } else if is_rejection_only(&input) {
                    self.pending_volatile_result = None;
                    self.pending_volatile_calculation_record = None;
                    println!(
                        "揮発前提を変更する場合は、揮発成分または原料組成を具体的に入力してください。"
                    );
                    continue;
                }
            }

            if awaiting_recalculate_answer {
                match classify_recalculate_answer(&input) {
                    RecalculateAnswer::Restart => {
                        self.reset_calculation_state();
                        chat_history.clear();
                        chat_history.push(Message::assistant(
                            "再計算を開始します。自己紹介は繰り返さず、作りたい組成と目標質量を確認します。",
                        ));
                        awaiting_recalculate_answer = false;
                        println!("作りたい組成（化学式）と、何g欲しいか教えてください。");
                        continue;
                    }
                    RecalculateAnswer::End => break,
                    RecalculateAnswer::NewRequest => {
                        chat_history.clear();
                        chat_history.push(Message::assistant(self.recalculation_context_message()));
                        awaiting_recalculate_answer = false;
                    }
                }
            }

            self.update_state_from_user_input(&input);
            let prompt = self.state.wrap_user_input(
                &input,
                &self.last_suggested_formulas,
                self.last_calculation_summary.as_deref(),
                &self.calculation_records,
            );
            let stream = agent.stream_chat(prompt, chat_history.clone()).await;
            let stream_result = print_streaming_agent_response(stream).await?;
            let should_end = self.apply_stream_result(
                stream_result,
                &mut chat_history,
                &mut awaiting_recalculate_answer,
            );
            if awaiting_recalculate_answer {
                continue;
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
        let adopted_suggested_precursors = (is_confirmation_only(input)
            || references_suggested_formulas(input))
            && self.state.precursor_formulas.is_empty()
            && !self.last_suggested_formulas.is_empty();
        if (is_confirmation_only(input) || references_suggested_formulas(input))
            && self.state.precursor_formulas.is_empty()
            && !self.last_suggested_formulas.is_empty()
        {
            self.state.precursor_formulas = self.last_suggested_formulas.clone();
        }

        let formulas = extract_valid_formulas(input);
        if self.state.target_formula.is_none() {
            if let Some(formula) = formulas.first() {
                self.state.target_formula = Some(formula.clone());
            } else if let Some(formula) = guess_material_formula(input) {
                self.last_suggested_formulas = vec![formula];
            }
        } else if self.state.target_mass_g.is_some()
            && !formulas.is_empty()
            && !adopted_suggested_precursors
        {
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

    fn reset_calculation_state(&mut self) {
        self.state = ConversationState::default();
        self.last_suggested_formulas.clear();
        self.pending_volatile_result = None;
        self.pending_volatile_calculation_record = None;
        self.last_calculation_summary = None;
        self.calculation_records.clear();
    }

    fn update_last_calculation_summary(&mut self) {
        let Some(target_formula) = self.state.target_formula.as_deref() else {
            return;
        };
        let Some(target_mass_g) = self.state.target_mass_g else {
            return;
        };
        let precursor_formulas = if self.state.precursor_formulas.is_empty() {
            "未確定".to_string()
        } else {
            self.state.precursor_formulas.join(", ")
        };
        self.last_calculation_summary = Some(format!(
            "目的組成: {target_formula}; 目標質量_g: {target_mass_g}; 原料組成: {precursor_formulas}"
        ));
    }

    fn recalculation_context_message(&self) -> String {
        match self.last_calculation_summary.as_deref() {
            Some(summary) => format!(
                "前回計算のバリエーションとして処理します。自己紹介は繰り返さず、前回計算要約とこれまでの計算一覧を参照して、ユーザーの変更指示だけを反映します。前回計算要約: {summary}"
            ),
            None => "再計算を開始します。自己紹介は繰り返さず、ユーザーの新しい依頼を処理します。"
                .to_string(),
        }
    }

    fn add_calculation_record(&mut self, record: String) {
        self.calculation_records.push(record);
    }

    fn apply_stream_result(
        &mut self,
        stream_result: AgentStreamPrintResult,
        chat_history: &mut Vec<Message>,
        awaiting_recalculate_answer: &mut bool,
    ) -> bool {
        let should_end = stream_result.should_end;
        self.update_last_suggestions(&stream_result.response_text);
        if let Some(updated_history) = stream_result.updated_history {
            *chat_history = updated_history;
        }
        if stream_result.printed_deterministic_result {
            self.update_last_calculation_summary();
            if let Some(record) = stream_result.completed_calculation_record {
                self.add_calculation_record(record);
            }
            *awaiting_recalculate_answer = true;
        }
        if stream_result.pending_volatile_result.is_some() {
            self.pending_volatile_result = stream_result.pending_volatile_result;
            self.pending_volatile_calculation_record =
                stream_result.pending_volatile_calculation_record;
        }
        should_end
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecalculateAnswer {
    Restart,
    End,
    NewRequest,
}

fn classify_recalculate_answer(input: &str) -> RecalculateAnswer {
    let normalized = input
        .trim()
        .trim_matches(|ch: char| {
            ch.is_ascii_punctuation() || matches!(ch, '。' | '、' | '？' | '！')
        })
        .to_ascii_lowercase();
    match normalized.as_str() {
        "はい" | "yes" | "y" | "ok" | "再計算" | "もう一度" | "もう1回" | "続ける" => {
            RecalculateAnswer::Restart
        }
        "いいえ" | "no" | "n" | "終了" | "終わり" | "やめる" => RecalculateAnswer::End,
        _ => RecalculateAnswer::NewRequest,
    }
}

fn is_rejection_only(input: &str) -> bool {
    let normalized = input
        .trim()
        .trim_matches(|ch: char| {
            ch.is_ascii_punctuation() || matches!(ch, '。' | '、' | '？' | '！')
        })
        .to_ascii_lowercase();
    matches!(
        normalized.as_str(),
        "いいえ" | "no" | "n" | "違う" | "ちがう" | "だめ" | "ng"
    )
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
    fn parses_thinking_setting() {
        assert_eq!(
            parse_weigh_calc_thinking(None).unwrap(),
            ThinkingSetting::Auto
        );
        assert_eq!(
            parse_weigh_calc_thinking(Some(" auto ".to_string())).unwrap(),
            ThinkingSetting::Auto
        );

        for value in ["off", "false", "0", "no", " OFF "] {
            assert_eq!(
                parse_weigh_calc_thinking(Some(value.to_string())).unwrap(),
                ThinkingSetting::Off
            );
        }

        assert!(parse_weigh_calc_thinking(Some("low".to_string())).is_err());
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
        let qwen = build_agent_prompt_config("qwen3.6:35b", 4096, ThinkingSetting::Auto);
        assert_eq!(qwen.thinking_mode, ThinkingMode::OllamaBool);
        assert_eq!(qwen.params, json!({ "think": true, "num_ctx": 4096 }));
        assert_eq!(qwen.preamble, AGENT_PREAMBLE);

        let gpt_oss = build_agent_prompt_config("gpt-oss:20b", 4096, ThinkingSetting::Auto);
        assert_eq!(gpt_oss.thinking_mode, ThinkingMode::OllamaLevel("medium"));
        assert_eq!(
            gpt_oss.params,
            json!({ "think": "medium", "num_ctx": 4096 })
        );
        assert_eq!(gpt_oss.preamble, AGENT_PREAMBLE);

        let gemma = build_agent_prompt_config("gemma-4:26b", 4096, ThinkingSetting::Auto);
        assert_eq!(gemma.thinking_mode, ThinkingMode::PromptToken);
        assert_eq!(gemma.params, json!({ "num_ctx": 4096 }));
        assert!(gemma.preamble.starts_with("<|think|>\n"));
        assert!(gemma.preamble.ends_with(AGENT_PREAMBLE));

        let disabled = build_agent_prompt_config("llama3.2:latest", 4096, ThinkingSetting::Auto);
        assert_eq!(disabled.thinking_mode, ThinkingMode::Disabled);
        assert_eq!(disabled.params, json!({ "num_ctx": 4096 }));
        assert_eq!(disabled.preamble, AGENT_PREAMBLE);
    }

    #[test]
    fn builds_thinking_off_agent_prompt_config() {
        for model in ["qwen3.6:35b", "gpt-oss:20b", "gemma-4:26b"] {
            let config = build_agent_prompt_config(model, 4096, ThinkingSetting::Off);

            assert_eq!(config.thinking_mode, ThinkingMode::Disabled);
            assert_eq!(config.params, json!({ "num_ctx": 4096 }));
            assert_eq!(config.preamble, AGENT_PREAMBLE);
            assert!(config.params.get("think").is_none());
            assert!(!config.preamble.starts_with("<|think|>"));
        }
    }

    #[test]
    fn classifies_recalculate_answers() {
        for input in ["はい", "yes", "Y", "ok", "再計算", "もう一度", "続ける"] {
            assert_eq!(
                classify_recalculate_answer(input),
                RecalculateAnswer::Restart
            );
        }
        for input in ["いいえ", "no", "N", "終了", "終わり", "やめる"] {
            assert_eq!(classify_recalculate_answer(input), RecalculateAnswer::End);
        }
        assert_eq!(
            classify_recalculate_answer("LiCoO2 3g"),
            RecalculateAnswer::NewRequest
        );
    }

    #[test]
    fn accepts_example_confirmation_phrases() {
        for input in [
            "それでお願い",
            "それでお願いします。",
            "その例の通りで",
            "その例どおりで",
            "その通りで",
            "その組成で",
            "例の通りで",
            "例のリストで",
            "そのリストで",
            "提示したリストで",
        ] {
            assert!(is_confirmation_only(input), "{input}");
        }
    }

    #[test]
    fn wraps_last_suggested_formulas_into_user_prompt() {
        let state = ConversationState {
            target_formula: Some("LaCu3Mn4O12".to_string()),
            target_mass_g: Some(5.0),
            precursor_formulas: Vec::new(),
        };
        let prompt = state.wrap_user_input(
            "例のリストで",
            &["La2O3".to_string(), "CuO".to_string(), "Mn2O3".to_string()],
            None,
            &[],
        );

        assert!(prompt.contains("原料組成: 未確定"));
        assert!(prompt.contains("直前提示組成候補: La2O3, CuO, Mn2O3"));
        assert!(prompt.contains("[ユーザー入力]\n例のリストで"));
    }

    #[test]
    fn wraps_previous_calculation_summary_into_user_prompt() {
        let state = ConversationState {
            target_formula: Some("LaCu3Mn4O12".to_string()),
            target_mass_g: Some(5.0),
            precursor_formulas: vec!["La2O3".to_string(), "CuO".to_string(), "Mn2O3".to_string()],
        };
        let summary = "目的組成: LaCu3Mn4O12; 目標質量_g: 5; 原料組成: La2O3, CuO, Mn2O3";
        let prompt = state.wrap_user_input(
            "3gを5gにして",
            &[],
            Some(summary),
            &["作製材料: LaCu3Mn4O12; 目標質量_g: 5; 反応式: 0.5 La2O3 + 3 CuO + 2 Mn2O3 -> LaCu3Mn4O12".to_string()],
        );

        assert!(prompt.contains("前回計算要約: 目的組成: LaCu3Mn4O12"));
        assert!(prompt.contains("確定済み項目: 目的組成, 目標質量, 原料組成"));
        assert!(prompt.contains("これまでの計算一覧:\n1. 作製材料: LaCu3Mn4O12"));
        assert!(prompt.contains("[ユーザー入力]\n3gを5gにして"));
    }

    #[test]
    fn adopts_suggested_precursors_with_extra_volatile_note() {
        let mut app = WeighingAgentApp {
            ollama_base_url: "http://localhost:11434".to_string(),
            model: "qwen3.6:35b".to_string(),
            ollama_num_ctx: DEFAULT_OLLAMA_NUM_CTX,
            thinking_setting: ThinkingSetting::Auto,
            line_editor: DefaultEditor::new().unwrap(),
            state: ConversationState {
                target_formula: Some("LaTaO3".to_string()),
                target_mass_g: Some(5.0),
                precursor_formulas: Vec::new(),
            },
            last_suggested_formulas: vec!["La2O3".to_string(), "Ta2O5".to_string()],
            pending_volatile_result: None,
            pending_volatile_calculation_record: None,
            last_calculation_summary: None,
            calculation_records: Vec::new(),
        };

        app.update_state_from_user_input("それでOK.O2脱離するから組成式反映させてね");

        assert_eq!(app.state.target_formula.as_deref(), Some("LaTaO3"));
        assert_eq!(app.state.target_mass_g, Some(5.0));
        assert_eq!(app.state.precursor_formulas, ["La2O3", "Ta2O5"]);
    }

    #[test]
    fn extracts_last_inline_mass_request() {
        assert_eq!(extract_mass_g("3gを5gにして"), Some(5.0));
        assert_eq!(extract_mass_g("前回と同じで 2.5g"), Some(2.5));
        assert_eq!(extract_mass_g("500mgではなく1g"), Some(1.0));
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
