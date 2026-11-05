//! Behavioural Engine data contracts.

pub(crate) mod codebook;
mod event;
mod featurizer;
mod history;
mod profile;
mod sequence;

pub use event::{
    ActionKind, BehaviourEvent, BehaviourEventError, DecisionContext, HandId, MAX_BOARD_CARDS,
    ObservedAction, Position, PublicCard, Rank, Street, Suit,
};
pub(crate) use featurizer::FeaturizedHandData;
pub use featurizer::{
    BehaviourFeaturizer, BehaviourFeaturizerError, CategoricalFeature, ContinuousFeature,
    EVENT_CATEGORICAL_WIDTH, EVENT_CONTINUOUS_WIDTH, FeaturizedHand, FeaturizedHistory,
};
pub use history::{BehaviourHistory, SubjectHand};
pub use profile::{
    BehaviourLatentConfig, BehaviourLatentConfigError, BehaviourProfile, BehaviourProfileError,
};
pub use sequence::{BehaviourSequence, BehaviourSequenceError};

/// Common latent-space contract implemented by both sides of a behavioural model pair.
pub trait BehaviourLatentModel {
    fn latent_config(&self) -> BehaviourLatentConfig;
}

/// Infers a behavioural profile from multiple real hands of the same subject.
pub trait BehaviourAnalyzer: BehaviourLatentModel {
    type Error;

    fn analyze(&self, support: &[FeaturizedHand]) -> Result<BehaviourProfile, Self::Error>;
}

/// Generates behaviour for a supplied profile from the public history and current context.
pub trait BehaviourGenerator: BehaviourLatentModel {
    type Error;

    fn generate(
        &self,
        profile: &BehaviourProfile,
        sequence: &BehaviourSequence,
        context: &DecisionContext,
    ) -> Result<ObservedAction, Self::Error>;
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
        type Error = std::convert::Infallible;

        fn analyze(&self, _support: &[FeaturizedHand]) -> Result<BehaviourProfile, Self::Error> {
            Ok(self
                .config
                .profile(vec![0.0; self.config.dimension()])
                .unwrap())
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

        fn generate(
            &self,
            _profile: &BehaviourProfile,
            _sequence: &BehaviourSequence,
            _context: &DecisionContext,
        ) -> Result<ObservedAction, Self::Error> {
            unreachable!("latent contract test does not generate poker actions")
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
