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
    ANALYZER_EVENT_WIDTH, AnalyzerAdam, AnalyzerError, AnalyzerGradientBatch, AnalyzerParameters,
    BehaviourPosterior, MomentaryBehaviour, NeuralBehaviourAnalyzer,
};
pub use baseline_adaptation::{
    BaselineAdaptationConfig, BaselineAdaptationError, BaselineHandReport,
    BaselineStateStorageError, PlayerBaselineAdapter, PlayerBaselineState,
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
    NeuralBehaviourGenerator, ScoringLoss, TimingDistribution, TrainingLoss, WagerBounds,
};
pub use history::{BehaviourHistory, SubjectHand};
pub use profile::{
    BehaviourBlueprint, BehaviourLatentConfig, BehaviourLatentConfigError, BehaviourProfile,
    BehaviourProfileError, PlayerBaseline,
};
pub use sequence::{BehaviourSequence, BehaviourSequenceError};

/// Common latent-space contract implemented by both sides of a behavioural model pair.
pub trait BehaviourLatentModel {
    fn latent_config(&self) -> BehaviourLatentConfig;
}
