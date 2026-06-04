use crate::chemistry::parse_compound;

#[derive(Debug, Default)]
pub(crate) struct ConversationState {
    pub(crate) target_formula: Option<String>,
    pub(crate) target_mass_g: Option<f64>,
    pub(crate) precursor_formulas: Vec<String>,
}

impl ConversationState {
    pub(crate) fn wrap_user_input(&self, input: &str) -> String {
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

pub(crate) fn is_confirmation_only(input: &str) -> bool {
    let normalized = input
        .trim()
        .trim_matches(|ch: char| ch.is_ascii_punctuation() || matches!(ch, '。' | '、'))
        .to_ascii_lowercase();
    matches!(
        normalized.as_str(),
        "ok" | "yes" | "y" | "はい" | "それでok" | "それでよい" | "それで良い" | "それでいい"
    )
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
