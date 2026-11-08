//! Behavioural Engine data contracts and trainable conditional generator.

mod analyzer;
mod baseline_adaptation;
pub(crate) mod checkpoint;
pub(crate) mod codebook;
mod event;
mod featurizer;
mod generator;
mod history;
mod profile;
mod sequence;

pub(crate) use analyzer::NoiseStream;
pub use analyzer::{
    ANALYZER_EVENT_WIDTH, AnalyzerAdam, AnalyzerError, AnalyzerParameters, BehaviourPosterior,
    MomentaryBehaviour, NeuralBehaviourAnalyzer,
};
pub use baseline_adaptation::{
    BaselineAdaptationConfig, BaselineAdaptationError, BaselineHandReport, PlayerBaselineAdapter,
    PlayerBaselineState,
};
pub use event::{
    ActionHistory, ActionKind, ActionSizing, BehaviourEvent, BehaviourEventError, DecisionContext,
    ExactAction, HandId, ObservedAction, PlayerState, PlayerStatus, Position, PreviousAggression,
    Street,
};
pub use featurizer::{
    BehaviourFeaturizer, BehaviourFeaturizerError, CategoricalFeature, ContinuousFeature,
    EVENT_CATEGORICAL_WIDTH, EVENT_CONTINUOUS_WIDTH, FeaturizedHand, FeaturizedHistory,
    PRE_ACTION_CONTEXT_WIDTH,
};
pub(crate) use featurizer::{FeaturizedHandData, continuous_feature_name, encode_continuous};
pub use generator::{
    ACTION_COUNT, Adam, BehaviourPolicy, DenseLayer, GENERATOR_CONTEXT_WIDTH,
    GENERATOR_SIZING_WIDTH, GENERATOR_TIMING_WIDTH, GeneratorDraws, GeneratorError,
    GeneratorGradientBatch, GeneratorObservation, GeneratorParameters, LegalActionSpace,
    NeuralBehaviourGenerator, TimingDistribution, TrainingLoss, WagerBounds,
};
pub use history::{BehaviourHistory, SubjectHand};
pub use profile::{
    BehaviourBlueprint, BehaviourLatentConfig, BehaviourLatentConfigError, BehaviourProfile,
    BehaviourProfileError, PlayerBaseline,
};
pub use sequence::{BehaviourSequence, BehaviourSequenceError};

impl BehaviourAnalyzer for NeuralBehaviourAnalyzer {
    type Error = AnalyzerError;
    fn analyze(&self, support: &[FeaturizedHand]) -> Result<BehaviourPosterior, Self::Error> {
        self.infer(support)
    }
}

/// Common latent-space contract implemented by both sides of a behavioural model pair.
pub trait BehaviourLatentModel {
    fn latent_config(&self) -> BehaviourLatentConfig;
}

/// Infers a distribution over latent profiles from past hands of the same subject.
pub trait BehaviourAnalyzer: BehaviourLatentModel {
    type Error;

    fn analyze(&self, support: &[FeaturizedHand]) -> Result<BehaviourPosterior, Self::Error>;
}

/// Predicts action, continuous sizing and timing distributions from (context, Z).
/// A separately supplied random draw produces one observed action from this policy.
pub trait BehaviourGenerator: BehaviourLatentModel {
    type Error;

    fn predict(
        &self,
        profile: &BehaviourProfile,
        context: &DecisionContext,
        legal: &LegalActionSpace,
    ) -> Result<BehaviourPolicy, Self::Error>;
}

impl BehaviourGenerator for NeuralBehaviourGenerator {
    type Error = GeneratorError;

    fn predict(
        &self,
        profile: &BehaviourProfile,
        context: &DecisionContext,
        legal: &LegalActionSpace,
    ) -> Result<BehaviourPolicy, Self::Error> {
        NeuralBehaviourGenerator::predict(self, profile, context, legal)
    }
}

/// Rejects Analyzer/Generator pairs trained with different latent dimensions.
pub fn validate_latent_dimensions<A, G>(
    analyzer: &A,
    generator: &G,
) -> Result<BehaviourLatentConfig, BehaviourLatentDimensionError>
where
    A: BehaviourAnalyzer,
    G: BehaviourGenerator,
{
    let analyzer_config = analyzer.latent_config();
    let generator_config = generator.latent_config();

    if analyzer_config != generator_config {
        return Err(BehaviourLatentDimensionError::DimensionMismatch {
            analyzer: analyzer_config.dimension(),
            generator: generator_config.dimension(),
        });
    }

    Ok(analyzer_config)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BehaviourLatentDimensionError {
    DimensionMismatch { analyzer: usize, generator: usize },
}

impl std::fmt::Display for BehaviourLatentDimensionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch {
                analyzer,
                generator,
            } => write!(
                f,
                "behaviour latent dimension mismatch: Analyzer uses {analyzer}, Generator uses {generator}"
            ),
        }
    }
}

impl std::error::Error for BehaviourLatentDimensionError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy)]
    struct StubAnalyzer {
        config: BehaviourLatentConfig,
    }

    impl BehaviourLatentModel for StubAnalyzer {
        fn latent_config(&self) -> BehaviourLatentConfig {
            self.config
        }
    }

    impl BehaviourAnalyzer for StubAnalyzer {
        type Error = AnalyzerError;
        fn analyze(&self, _support: &[FeaturizedHand]) -> Result<BehaviourPosterior, Self::Error> {
            Err(AnalyzerError::NoSubjectDecisions)
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct StubGenerator {
        config: BehaviourLatentConfig,
    }

    impl BehaviourLatentModel for StubGenerator {
        fn latent_config(&self) -> BehaviourLatentConfig {
            self.config
        }
    }

    impl BehaviourGenerator for StubGenerator {
        type Error = std::convert::Infallible;

        fn predict(
            &self,
            _profile: &BehaviourProfile,
            _context: &DecisionContext,
            _legal: &LegalActionSpace,
        ) -> Result<BehaviourPolicy, Self::Error> {
            unreachable!("latent contract test does not predict poker actions")
        }
    }

    #[test]
    fn accepts_pair_with_same_latent_dimension() {
        let config = BehaviourLatentConfig::new(16).unwrap();
        let analyzer = StubAnalyzer { config };
        let generator = StubGenerator { config };

        assert_eq!(
            validate_latent_dimensions(&analyzer, &generator),
            Ok(config)
        );
    }

    #[test]
    fn rejects_pair_with_different_latent_dimensions() {
        let analyzer = StubAnalyzer {
            config: BehaviourLatentConfig::new(16).unwrap(),
        };
        let generator = StubGenerator {
            config: BehaviourLatentConfig::new(32).unwrap(),
        };

        assert_eq!(
            validate_latent_dimensions(&analyzer, &generator),
            Err(BehaviourLatentDimensionError::DimensionMismatch {
                analyzer: 16,
                generator: 32,
            })
        );
    }
}
