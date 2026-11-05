//! Conditional behavioural generator: (public context, latent Z) -> policy.
//!
//! This module is an inference kernel. Parameters must be supplied explicitly:
//! no pretrained blueprint or training routine is assumed to exist.

use std::f32::consts::PI;
use std::fmt;

use super::codebook::{position_id, rank_id, street_id, suit_id};
use super::{
    ActionKind, BehaviourEventError, BehaviourLatentConfig, BehaviourLatentModel, BehaviourProfile,
    DecisionContext, MAX_BOARD_CARDS, ObservedAction,
};

const STREET_COUNT: usize = 4;
const POSITION_COUNT: usize = 7;
const NUMERIC_COUNT: usize = 11;
const BOARD_FEATURES_PER_CARD: usize = 3;
const ACTION_COUNT: usize = 5;
const MOVE_EXTRA_FEATURES: usize = 4;

/// Deterministic input encoding width, without the latent vector.
pub const GENERATOR_CONTEXT_WIDTH: usize =
    STREET_COUNT + POSITION_COUNT + NUMERIC_COUNT + MAX_BOARD_CARDS * BOARD_FEATURES_PER_CARD;
/// Action kind, amount, commitment, relative size, and all-in flag.
pub const GENERATOR_MOVE_WIDTH: usize = ACTION_COUNT + MOVE_EXTRA_FEATURES;

/// A complete legal *candidate* as provided by the game's rules engine.
/// The generator selects from these, rather than inventing an arbitrary wager.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LegalMove {
    pub kind: ActionKind,
    pub is_all_in: bool,
    pub amount_added_bb: f32,
    pub street_total_after_bb: f32,
}

impl LegalMove {
    fn validate(self, context: &DecisionContext) -> Result<(), GeneratorError> {
        for (name, value) in [
            ("amount_added_bb", self.amount_added_bb),
            ("street_total_after_bb", self.street_total_after_bb),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(GeneratorError::InvalidLegalMove(name));
            }
        }
        if matches!(self.kind, ActionKind::Fold | ActionKind::Check)
            && (self.amount_added_bb != 0.0 || self.is_all_in)
        {
            return Err(GeneratorError::InvalidLegalMove(
                "passive action amount/all-in",
            ));
        }
        if self.amount_added_bb > context.actor_stack_bb + 1e-4 {
            return Err(GeneratorError::InvalidLegalMove(
                "amount exceeds actor stack",
            ));
        }
        let total = context.actor_street_commitment_bb + self.amount_added_bb;
        if !total.is_finite() || (self.street_total_after_bb - total).abs() > 1e-4 * total.max(1.0)
        {
            return Err(GeneratorError::InvalidLegalMove(
                "street total inconsistent",
            ));
        }
        if self.is_all_in
            && (context.actor_stack_bb - self.amount_added_bb).abs()
                > 1e-4 * context.actor_stack_bb.max(1.0)
        {
            return Err(GeneratorError::InvalidLegalMove(
                "all-in amount inconsistent",
            ));
        }
        Ok(())
    }
}

/// A validated dense linear layer, with row-major weights [output, input].
#[derive(Debug, Clone, PartialEq)]
pub struct DenseLayer {
    input: usize,
    output: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
}

impl DenseLayer {
    pub fn new(
        input: usize,
        output: usize,
        weights: Vec<f32>,
        bias: Vec<f32>,
    ) -> Result<Self, GeneratorError> {
        if input == 0
            || output == 0
            || input.checked_mul(output) != Some(weights.len())
            || bias.len() != output
        {
            return Err(GeneratorError::InvalidLayerShape);
        }
        if weights.iter().chain(&bias).any(|x| !x.is_finite()) {
            return Err(GeneratorError::NonFiniteParameter);
        }
        Ok(Self {
            input,
            output,
            weights,
            bias,
        })
    }

    pub fn input(&self) -> usize {
        self.input
    }
    pub fn output(&self) -> usize {
        self.output
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        debug_assert_eq!(input.len(), self.input);
        self.weights
            .chunks_exact(self.input)
            .zip(self.bias.iter())
            .map(|(row, bias)| row.iter().zip(input).fold(*bias, |sum, (w, x)| sum + w * x))
            .collect()
    }
}

/// Weights of a small conditional neural model.
///
/// All parameters are explicit and can later be owned by an autodiff backend.
/// `state` encodes (context, Z), `candidate` scores each legal move jointly
/// with that state, `score` outputs logits, `timing` models log milliseconds.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratorParameters {
    pub state: DenseLayer,
    pub candidate: DenseLayer,
    pub score: DenseLayer,
    pub timing: DenseLayer,
}

/// A distribution over the *given* legal candidates, plus timing parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct BehaviourPolicy {
    moves: Vec<(LegalMove, f32)>,
    timing: TimingDistribution,
}

impl BehaviourPolicy {
    pub fn moves(&self) -> &[(LegalMove, f32)] {
        &self.moves
    }
    pub fn timing(&self) -> TimingDistribution {
        self.timing
    }

    /// Probability assigned to an action kind, summed across its legal sizings.
    pub fn action_probability(&self, kind: ActionKind) -> f32 {
        self.moves
            .iter()
            .filter(|(m, _)| m.kind == kind)
            .map(|(_, p)| p)
            .sum()
    }

    /// External uniform draws make sampling deterministic and reproducible.
    /// Draws: action in [0,1), timing_1 in (0,1), timing_2 in [0,1).
    pub fn sample(&self, draws: GeneratorDraws) -> Result<ObservedAction, GeneratorError> {
        if !(draws.action >= 0.0 && draws.action < 1.0)
            || !(draws.timing_1 > 0.0 && draws.timing_1 < 1.0)
            || !(draws.timing_2 >= 0.0 && draws.timing_2 < 1.0)
            || !draws.action.is_finite()
            || !draws.timing_1.is_finite()
            || !draws.timing_2.is_finite()
        {
            return Err(GeneratorError::InvalidRandomDraw);
        }
        let mut cumulative = 0.0f64;
        let selected = self
            .moves
            .iter()
            .find(|(_, probability)| {
                cumulative += f64::from(*probability);
                f64::from(draws.action) < cumulative
            })
            .ok_or(GeneratorError::InvalidProbabilityMass)?;

        let ms = self.timing.sample_ms(draws.timing_1, draws.timing_2)?;
        let (chosen, _) = selected;
        Ok(ObservedAction {
            kind: chosen.kind,
            is_all_in: chosen.is_all_in,
            amount_added_bb: chosen.amount_added_bb,
            street_total_after_bb: chosen.street_total_after_bb,
            decision_time_ms: Some(ms),
        })
    }
}

/// Timing distribution parameterized in natural log(milliseconds).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimingDistribution {
    pub log_ms_mean: f32,
    pub log_ms_std: f32,
}

impl TimingDistribution {
    fn sample_ms(self, u1: f32, u2: f32) -> Result<u32, GeneratorError> {
        let radius = (-2.0f64 * f64::from(u1).ln()).sqrt();
        let normal = radius * (2.0 * f64::from(PI) * f64::from(u2)).cos();
        let ms = (f64::from(self.log_ms_mean) + f64::from(self.log_ms_std) * normal)
            .exp()
            .round();
        if !ms.is_finite() || ms < 0.0 || ms > f64::from(u32::MAX) {
            return Err(GeneratorError::InvalidGeneratedTiming);
        }
        Ok(ms as u32)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeneratorDraws {
    pub action: f32,
    pub timing_1: f32,
    pub timing_2: f32,
}

/// Untrained conditional neural generator. Construction requires explicit weights.
#[derive(Debug, Clone)]
pub struct NeuralBehaviourGenerator {
    latent: BehaviourLatentConfig,
    parameters: GeneratorParameters,
}

impl NeuralBehaviourGenerator {
    pub fn new(
        latent: BehaviourLatentConfig,
        parameters: GeneratorParameters,
    ) -> Result<Self, GeneratorError> {
        let hidden = parameters.state.output();
        if parameters.state.input() != GENERATOR_CONTEXT_WIDTH + latent.dimension()
            || parameters.candidate.input() != hidden + GENERATOR_MOVE_WIDTH
            || parameters.candidate.output() != hidden
            || parameters.score.input() != hidden
            || parameters.score.output() != 1
            || parameters.timing.input() != hidden
            || parameters.timing.output() != 2
        {
            return Err(GeneratorError::InvalidModelShape);
        }
        Ok(Self { latent, parameters })
    }

    pub fn predict(
        &self,
        profile: &BehaviourProfile,
        context: &DecisionContext,
        legal_moves: &[LegalMove],
    ) -> Result<BehaviourPolicy, GeneratorError> {
        if profile.dimension() != self.latent.dimension() {
            return Err(GeneratorError::WrongLatentDimension {
                expected: self.latent.dimension(),
                actual: profile.dimension(),
            });
        }
        context.validate().map_err(GeneratorError::InvalidContext)?;
        if context.board.len() > MAX_BOARD_CARDS {
            return Err(GeneratorError::TooManyBoardCards);
        }
        if legal_moves.is_empty() {
            return Err(GeneratorError::NoLegalMoves);
        }
        if legal_moves.len() > 4096 {
            return Err(GeneratorError::TooManyLegalMoves);
        }
        for (index, candidate) in legal_moves.iter().enumerate() {
            candidate.validate(context)?;
            if legal_moves[..index].contains(candidate) {
                return Err(GeneratorError::DuplicateLegalMove);
            }
        }

        let mut state_features = context_features(context);
        state_features.extend_from_slice(profile.as_slice());
        let state = tanh_vec(self.parameters.state.forward(&state_features));
        let mut logits = Vec::with_capacity(legal_moves.len());
        for candidate in legal_moves {
            let mut input = state.clone();
            input.extend_from_slice(&move_features(*candidate, context));
            let hidden = tanh_vec(self.parameters.candidate.forward(&input));
            let raw = self.parameters.score.forward(&hidden)[0];
            if !raw.is_finite() {
                return Err(GeneratorError::NonFiniteOutput);
            }
            logits.push(raw);
        }
        let probabilities = softmax(&logits)?;
        let timing_raw = self.parameters.timing.forward(&state);
        let timing = TimingDistribution {
            log_ms_mean: timing_raw[0],
            log_ms_std: softplus(timing_raw[1]) + 0.05,
        };
        if !timing.log_ms_mean.is_finite() || !timing.log_ms_std.is_finite() {
            return Err(GeneratorError::NonFiniteOutput);
        }
        Ok(BehaviourPolicy {
            moves: legal_moves.iter().copied().zip(probabilities).collect(),
            timing,
        })
    }
}

impl BehaviourLatentModel for NeuralBehaviourGenerator {
    fn latent_config(&self) -> BehaviourLatentConfig {
        self.latent
    }
}

fn tanh_vec(values: Vec<f32>) -> Vec<f32> {
    values.into_iter().map(f32::tanh).collect()
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        value.exp().ln_1p()
    }
}

fn softmax(logits: &[f32]) -> Result<Vec<f32>, GeneratorError> {
    let max = logits
        .iter()
        .copied()
        .reduce(f32::max)
        .ok_or(GeneratorError::NoLegalMoves)?;
    let weights: Vec<f64> = logits.iter().map(|&x| f64::from(x - max).exp()).collect();
    let denominator: f64 = weights.iter().sum();
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(GeneratorError::InvalidProbabilityMass);
    }
    let probs: Vec<f32> = weights.iter().map(|&x| (x / denominator) as f32).collect();
    if probs.iter().any(|x| !x.is_finite()) {
        return Err(GeneratorError::NonFiniteOutput);
    }
    Ok(probs)
}

fn context_features(c: &DecisionContext) -> Vec<f32> {
    let mut values = vec![0.0; GENERATOR_CONTEXT_WIDTH];
    values[usize::from(street_id(c.street))] = 1.0;
    values[STREET_COUNT + usize::from(position_id(c.actor_position))] = 1.0;
    let start = STREET_COUNT + POSITION_COUNT;
    let positive_log = |x: f32| x.ln_1p();
    let continuous = [
        c.players_at_table as f32 / 6.0,
        c.players_active as f32 / 6.0,
        positive_log(c.pot_bb),
        positive_log(c.actor_stack_bb),
        positive_log(c.effective_stack_bb),
        positive_log(c.actor_street_commitment_bb),
        positive_log(c.current_bet_bb),
        positive_log(c.to_call_bb),
        c.min_raise_to_bb.map(positive_log).unwrap_or(0.0),
        if c.min_raise_to_bb.is_some() {
            1.0
        } else {
            0.0
        },
        f32::from(c.bet_or_raise_count_this_street) / 8.0,
    ];
    values[start..start + NUMERIC_COUNT].copy_from_slice(&continuous);
    let board_start = start + NUMERIC_COUNT;
    for (index, card) in c.board.iter().enumerate() {
        values[board_start + index * 3] = f32::from(rank_id(card.rank)) / 14.0;
        values[board_start + index * 3 + 1] = f32::from(suit_id(card.suit)) / 4.0;
        values[board_start + index * 3 + 2] = 1.0;
    }
    values
}

fn move_features(m: LegalMove, c: &DecisionContext) -> [f32; GENERATOR_MOVE_WIDTH] {
    let mut values = [0.0; GENERATOR_MOVE_WIDTH];
    let id = match m.kind {
        ActionKind::Fold => 0,
        ActionKind::Check => 1,
        ActionKind::Call => 2,
        ActionKind::Bet => 3,
        ActionKind::Raise => 4,
    };
    values[id] = 1.0;
    values[ACTION_COUNT] = m.amount_added_bb.ln_1p();
    values[ACTION_COUNT + 1] = m.street_total_after_bb.ln_1p();
    values[ACTION_COUNT + 2] = m.amount_added_bb / (c.actor_stack_bb + 1.0);
    values[ACTION_COUNT + 3] = if m.is_all_in { 1.0 } else { 0.0 };
    values
}

#[derive(Debug, Clone, PartialEq)]
pub enum GeneratorError {
    InvalidLayerShape,
    InvalidModelShape,
    NonFiniteParameter,
    NonFiniteOutput,
    InvalidContext(BehaviourEventError),
    TooManyBoardCards,
    NoLegalMoves,
    TooManyLegalMoves,
    DuplicateLegalMove,
    InvalidLegalMove(&'static str),
    WrongLatentDimension { expected: usize, actual: usize },
    InvalidProbabilityMass,
    InvalidRandomDraw,
    InvalidGeneratedTiming,
}

impl fmt::Display for GeneratorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for GeneratorError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::behavioural::{Position, Street};

    fn context() -> DecisionContext {
        DecisionContext {
            street: Street::Flop,
            board: vec![],
            actor_position: Position::Button,
            players_at_table: 2,
            players_active: 2,
            pot_bb: 10.0,
            actor_stack_bb: 80.0,
            effective_stack_bb: 80.0,
            actor_street_commitment_bb: 0.0,
            current_bet_bb: 0.0,
            to_call_bb: 0.0,
            min_raise_to_bb: Some(1.0),
            bet_or_raise_count_this_street: 0,
        }
    }

    fn moves() -> Vec<LegalMove> {
        vec![
            LegalMove {
                kind: ActionKind::Check,
                is_all_in: false,
                amount_added_bb: 0.0,
                street_total_after_bb: 0.0,
            },
            LegalMove {
                kind: ActionKind::Bet,
                is_all_in: false,
                amount_added_bb: 4.0,
                street_total_after_bb: 4.0,
            },
            LegalMove {
                kind: ActionKind::Bet,
                is_all_in: true,
                amount_added_bb: 80.0,
                street_total_after_bb: 80.0,
            },
        ]
    }

    fn zero_layer(input: usize, output: usize) -> DenseLayer {
        DenseLayer::new(input, output, vec![0.0; input * output], vec![0.0; output]).unwrap()
    }

    fn generator(latent_dim: usize) -> NeuralBehaviourGenerator {
        let hidden = 2;
        let parameters = GeneratorParameters {
            state: zero_layer(GENERATOR_CONTEXT_WIDTH + latent_dim, hidden),
            candidate: zero_layer(hidden + GENERATOR_MOVE_WIDTH, hidden),
            score: zero_layer(hidden, 1),
            timing: zero_layer(hidden, 2),
        };
        NeuralBehaviourGenerator::new(BehaviourLatentConfig::new(latent_dim).unwrap(), parameters)
            .unwrap()
    }

    #[test]
    fn probabilities_are_finite_and_normalized() {
        let g = generator(2);
        let z = g.latent.profile(vec![0.0, 0.0]).unwrap();
        let p = g.predict(&z, &context(), &moves()).unwrap();
        assert_eq!(p.moves().len(), 3);
        let total: f32 = p.moves().iter().map(|(_, probability)| probability).sum();
        assert!((total - 1.0).abs() < 1e-6);
        assert!((p.action_probability(ActionKind::Bet) - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn sampling_only_uses_supplied_legal_candidates_and_is_reproducible() {
        let g = generator(2);
        let z = g.latent.profile(vec![0.0, 0.0]).unwrap();
        let p = g.predict(&z, &context(), &moves()).unwrap();
        let draws = GeneratorDraws {
            action: 0.5,
            timing_1: 0.5,
            timing_2: 0.25,
        };
        let action = p.sample(draws).unwrap();
        assert_eq!(action, p.sample(draws).unwrap());
        assert!(moves().iter().any(|candidate| candidate.kind == action.kind
            && candidate.amount_added_bb == action.amount_added_bb
            && candidate.is_all_in == action.is_all_in));
    }

    #[test]
    fn rejects_bad_dimensions_and_empty_candidates() {
        let g = generator(2);
        let z = BehaviourLatentConfig::new(3)
            .unwrap()
            .profile(vec![0.0; 3])
            .unwrap();
        assert_eq!(
            g.predict(&z, &context(), &moves()),
            Err(GeneratorError::WrongLatentDimension {
                expected: 2,
                actual: 3
            })
        );
        let z = g.latent.profile(vec![0.0, 0.0]).unwrap();
        assert_eq!(
            g.predict(&z, &context(), &[]),
            Err(GeneratorError::NoLegalMoves)
        );
    }

    #[test]
    fn latent_changes_policy_when_parameters_use_it() {
        let mut g = generator(1);
        // state[0] = tanh(z), candidate[0] = tanh(state[0] + 2*is_bet)
        g.parameters.state.weights[GENERATOR_CONTEXT_WIDTH] = 1.0;
        g.parameters.candidate.weights[0] = 1.0;
        g.parameters.candidate.weights[2 + 3] = 2.0;
        g.parameters.score.weights[0] = 1.0;
        let low = g
            .predict(&g.latent.profile(vec![-1.0]).unwrap(), &context(), &moves())
            .unwrap();
        let high = g
            .predict(&g.latent.profile(vec![1.0]).unwrap(), &context(), &moves())
            .unwrap();
        assert!(
            (low.action_probability(ActionKind::Bet) - high.action_probability(ActionKind::Bet))
                .abs()
                > 1e-3
        );
    }

    #[test]
    fn rejects_inconsistent_wagers_and_duplicate_candidates() {
        let g = generator(1);
        let z = g.latent.profile(vec![0.0]).unwrap();
        let mut invalid = moves();
        invalid[0].amount_added_bb = 1.0;
        assert_eq!(
            g.predict(&z, &context(), &invalid),
            Err(GeneratorError::InvalidLegalMove(
                "passive action amount/all-in"
            ))
        );
        let dup = vec![moves()[0], moves()[0]];
        assert_eq!(
            g.predict(&z, &context(), &dup),
            Err(GeneratorError::DuplicateLegalMove)
        );
    }
}
