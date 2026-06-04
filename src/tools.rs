use anyhow::Result;
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use crate::chemistry::{
    WeighingResult, calculate_weighing_with_optional_volatiles, explain_calculation_error,
    is_inorganic_material_like, parse_compound, suggest_precursor_fixes,
};

#[derive(Debug)]
pub(crate) struct ToolCallFailure;

impl fmt::Display for ToolCallFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("tool call failed")
    }
}

impl Error for ToolCallFailure {}

#[derive(Debug, Deserialize)]
pub(crate) struct ValidateFormulaArgs {
    formula: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct ValidateFormulaOutput {
    valid: bool,
    formula: String,
    message: String,
    elements: BTreeMap<String, f64>,
    molar_mass_g_per_mol: Option<f64>,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct ValidateFormulaTool;

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
pub(crate) struct CalculateWeighingArgs {
    target_formula: String,
    target_mass_g: f64,
    precursor_formulas: Vec<String>,
    #[serde(default)]
    volatile_byproduct_formulas: Vec<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct CalculateWeighingOutput {
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
pub(crate) struct CalculateWeighingTool;

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
