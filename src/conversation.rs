use crate::chemistry::parse_compound;

#[derive(Debug, Default)]
pub(crate) struct ConversationState {
    pub(crate) target_formula: Option<String>,
    pub(crate) target_mass_g: Option<f64>,
    pub(crate) precursor_formulas: Vec<String>,
}

impl ConversationState {
    pub(crate) fn wrap_user_input(
        &self,
        input: &str,
        last_suggested_formulas: &[String],
        last_calculation_summary: Option<&str>,
        calculation_records: &[String],
    ) -> String {
        format!(
            "[現在の状態]\n目的組成: {}\n目標質量_g: {}\n原料組成: {}\n確定済み項目: {}\n直前提示組成候補: {}\n前回計算要約: {}\nこれまでの計算一覧:\n{}\n\n[ユーザー入力]\n{}",
            self.target_formula.as_deref().unwrap_or("未確定"),
            self.target_mass_g
                .map(|mass| format!("{mass}"))
                .unwrap_or_else(|| "未確定".to_string()),
            if self.precursor_formulas.is_empty() {
                "未確定".to_string()
            } else {
                self.precursor_formulas.join(", ")
            },
            self.confirmed_fields_text(),
            if last_suggested_formulas.is_empty() {
                "なし".to_string()
            } else {
                last_suggested_formulas.join(", ")
            },
            last_calculation_summary.unwrap_or("なし"),
            format_calculation_records(calculation_records),
            input
        )
    }

    fn confirmed_fields_text(&self) -> String {
        let mut fields = Vec::new();
        if self.target_formula.is_some() {
            fields.push("目的組成");
        }
        if self.target_mass_g.is_some() {
            fields.push("目標質量");
        }
        if !self.precursor_formulas.is_empty() {
            fields.push("原料組成");
        }
        if fields.is_empty() {
            "なし".to_string()
        } else {
            fields.join(", ")
        }
    }
}

fn format_calculation_records(records: &[String]) -> String {
    if records.is_empty() {
        return "なし".to_string();
    }

    records
        .iter()
        .enumerate()
        .map(|(index, record)| format!("{}. {}", index + 1, record))
        .collect::<Vec<_>>()
        .join("\n")
}

pub(crate) fn is_confirmation_only(input: &str) -> bool {
    let normalized = input
        .trim()
        .trim_matches(|ch: char| ch.is_ascii_punctuation() || matches!(ch, '。' | '、'))
        .to_ascii_lowercase();
    matches!(
        normalized.as_str(),
        "ok" | "yes"
            | "y"
            | "はい"
            | "お願いします"
            | "お願い"
            | "それで"
            | "それでok"
            | "それでよい"
            | "それで良い"
            | "それでいい"
            | "それでお願い"
            | "それでお願いします"
            | "その例の通りで"
            | "その例どおりで"
            | "その通りで"
            | "そのとおりで"
            | "その組成で"
            | "例の通りで"
            | "例どおりで"
            | "例のリストで"
            | "例リストで"
            | "提示リストで"
            | "提示したリストで"
            | "そのリストで"
    )
}

pub(crate) fn references_suggested_formulas(input: &str) -> bool {
    let normalized = input
        .trim()
        .trim_matches(|ch: char| ch.is_ascii_punctuation() || matches!(ch, '。' | '、'))
        .to_ascii_lowercase();

    [
        "それで",
        "その例",
        "例の",
        "そのリスト",
        "提示",
        "その組成",
        "ok",
        "はい",
    ]
    .iter()
    .any(|pattern| normalized.contains(pattern))
}

pub(crate) fn extract_valid_formulas(input: &str) -> Vec<String> {
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

pub(crate) fn guess_material_formula(input: &str) -> Option<String> {
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

pub(crate) fn extract_mass_g(input: &str) -> Option<f64> {
    if let Some(value) = extract_last_inline_mass_g(input) {
        return Some(value);
    }

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

fn extract_last_inline_mass_g(input: &str) -> Option<f64> {
    let mut last_mass = None;
    let mut iter = input.char_indices().peekable();

    while let Some((start, ch)) = iter.next() {
        if !ch.is_ascii_digit() && ch != '.' {
            continue;
        }

        let mut end = start + ch.len_utf8();
        while let Some((index, next_ch)) = iter.peek().copied() {
            if next_ch.is_ascii_digit() || next_ch == '.' {
                end = index + next_ch.len_utf8();
                iter.next();
            } else {
                break;
            }
        }

        let Some((_, unit)) = iter.peek().copied() else {
            continue;
        };
        if unit != 'g' && unit != 'G' {
            continue;
        }
        if input[..start].ends_with('m') || input[..start].ends_with('M') {
            continue;
        }

        if let Ok(value) = input[start..end].parse::<f64>()
            && value.is_finite()
            && value > 0.0
        {
            last_mass = Some(value);
        }
    }

    last_mass
}
