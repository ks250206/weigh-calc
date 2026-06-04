use anyhow::{Result, bail};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

const EPS: f64 = 1.0e-8;

#[derive(Debug, Clone)]
pub(crate) struct Compound {
    pub(crate) formula: String,
    pub(crate) atoms: BTreeMap<String, f64>,
    pub(crate) molar_mass: f64,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct WeighingResult {
    pub(crate) formula: String,
    pub(crate) moles: f64,
    pub(crate) grams: f64,
}

#[derive(Debug, Clone)]
pub(crate) struct WeighingCalculation {
    pub(crate) results: Vec<WeighingResult>,
    pub(crate) gas_reactants: Vec<WeighingResult>,
    pub(crate) volatile_byproducts: Vec<WeighingResult>,
    pub(crate) reaction_equation: String,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum CalculationError {
    Inconsistent,
    NotUnique,
    NegativeAmount { formula: String, moles: f64 },
}
pub(crate) fn parse_compound(formula: &str) -> Result<Compound> {
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

pub(crate) fn is_inorganic_material_like(compound: &Compound) -> bool {
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

pub(crate) fn calculate_weighing(
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

pub(crate) fn calculate_weighing_with_optional_volatiles(
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

pub(crate) fn calculate_weighing_with_volatile_byproducts(
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

pub(crate) fn explain_calculation_error(
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

pub(crate) fn suggest_precursor_fixes(target: &Compound, precursors: &[Compound]) -> Vec<String> {
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
