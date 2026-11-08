use std::fmt;

/// Fixed dimensionality of the behavioural latent space used by one model pair.
///
/// The dimension is an explicit model hyperparameter. It does not assign any
/// semantic meaning to individual latent coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BehaviourLatentConfig {
    dimension: usize,
}

impl BehaviourLatentConfig {
    pub fn new(dimension: usize) -> Result<Self, BehaviourLatentConfigError> {
        if dimension == 0 {
            return Err(BehaviourLatentConfigError::ZeroDimension);
        }

        Ok(Self { dimension })
    }

    pub fn dimension(self) -> usize {
        self.dimension
    }

    /// Creates a latent profile whose width must exactly match this config.
    pub fn profile(self, embedding: Vec<f32>) -> Result<BehaviourProfile, BehaviourProfileError> {
        BehaviourProfile::new(self, embedding)
    }

    pub fn validate_profile(self, profile: &BehaviourProfile) -> Result<(), BehaviourProfileError> {
        if profile.dimension() != self.dimension {
            return Err(BehaviourProfileError::DimensionMismatch {
                expected: self.dimension,
                observed: profile.dimension(),
            });
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BehaviourLatentConfigError {
    ZeroDimension,
}

impl fmt::Display for BehaviourLatentConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => {
                write!(f, "behaviour latent dimension must be greater than zero")
            }
        }
    }
}

impl std::error::Error for BehaviourLatentConfigError {}

/// Opaque behavioural representation shared by Analyzer and Generator.
#[derive(Debug, Clone, PartialEq)]
pub struct BehaviourProfile {
    embedding: Vec<f32>,
}

impl BehaviourProfile {
    pub fn new(
        config: BehaviourLatentConfig,
        embedding: Vec<f32>,
    ) -> Result<Self, BehaviourProfileError> {
        if embedding.len() != config.dimension() {
            return Err(BehaviourProfileError::DimensionMismatch {
                expected: config.dimension(),
                observed: embedding.len(),
            });
        }

        if let Some(index) = embedding.iter().position(|value| !value.is_finite()) {
            return Err(BehaviourProfileError::NonFiniteValue { index });
        }

        Ok(Self { embedding })
    }

    pub fn dimension(&self) -> usize {
        self.embedding.len()
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.embedding
    }

    pub fn into_vec(self) -> Vec<f32> {
        self.embedding
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BehaviourProfileError {
    DimensionMismatch { expected: usize, observed: usize },
    NonFiniteValue { index: usize },
    InvalidUncertainty { index: usize },
    NonIncreasingHandCount { previous: u64, new: u64 },
}

impl fmt::Display for BehaviourProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, observed } => write!(
                f,
                "behaviour profile dimension mismatch: expected {expected}, got {observed}"
            ),
            Self::NonFiniteValue { index } => {
                write!(
                    f,
                    "behaviour profile contains a non-finite value at {index}"
                )
            }
            Self::InvalidUncertainty { index } => {
                write!(f, "invalid latent uncertainty at {index}")
            }
            Self::NonIncreasingHandCount { previous, new } => {
                write!(
                    f,
                    "baseline hand count must increase from {previous}, got {new}"
                )
            }
        }
    }
}

impl std::error::Error for BehaviourProfileError {}

/// Learned population-level latent reference (Z0). Its dispersion is learned
/// separately; Z0 is a centre, not a claim that all humans have one profile.
#[derive(Debug, Clone, PartialEq)]
pub struct BehaviourBlueprint {
    centre: BehaviourProfile,
    prior_std: Vec<f32>,
}

impl BehaviourBlueprint {
    pub fn new(
        centre: BehaviourProfile,
        prior_std: Vec<f32>,
    ) -> Result<Self, BehaviourProfileError> {
        validate_std(centre.dimension(), &prior_std)?;
        Ok(Self { centre, prior_std })
    }

    pub fn centre(&self) -> &BehaviourProfile {
        &self.centre
    }

    pub fn prior_std(&self) -> &[f32] {
        &self.prior_std
    }
}

/// Personal Z1 estimate, kept distinct from A/G weights and the Z2 posterior.
/// Player identity is held by the caller; never injected into model features.
#[derive(Debug, Clone, PartialEq)]
pub struct PlayerBaseline {
    estimate: BehaviourProfile,
    uncertainty_std: Vec<f32>,
    observed_hands: u64,
}

impl PlayerBaseline {
    /// Strict Z1(0) == Z0, including the initial uncertainty distribution.
    pub fn from_blueprint(blueprint: &BehaviourBlueprint) -> Self {
        Self {
            estimate: blueprint.centre.clone(),
            uncertainty_std: blueprint.prior_std.clone(),
            observed_hands: 0,
        }
    }

    pub fn profile(&self) -> &BehaviourProfile {
        &self.estimate
    }

    pub fn uncertainty_std(&self) -> &[f32] {
        &self.uncertainty_std
    }

    pub fn observed_hands(&self) -> u64 {
        self.observed_hands
    }

    /// Store a new *learned* estimate; the optimization algorithm belongs to
    /// the Z1 adaptation trainer, not this data contract.
    pub fn set_estimate(
        &mut self,
        estimate: BehaviourProfile,
        uncertainty_std: Vec<f32>,
        observed_hands: u64,
    ) -> Result<(), BehaviourProfileError> {
        if observed_hands <= self.observed_hands {
            return Err(BehaviourProfileError::NonIncreasingHandCount {
                previous: self.observed_hands,
                new: observed_hands,
            });
        }
        if estimate.dimension() != self.estimate.dimension() {
            return Err(BehaviourProfileError::DimensionMismatch {
                expected: self.estimate.dimension(),
                observed: estimate.dimension(),
            });
        }
        validate_std(estimate.dimension(), &uncertainty_std)?;
        self.estimate = estimate;
        self.uncertainty_std = uncertainty_std;
        self.observed_hands = observed_hands;
        Ok(())
    }
}

fn validate_std(dimension: usize, std: &[f32]) -> Result<(), BehaviourProfileError> {
    if std.len() != dimension {
        return Err(BehaviourProfileError::DimensionMismatch {
            expected: dimension,
            observed: std.len(),
        });
    }
    if let Some(index) = std
        .iter()
        .position(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(BehaviourProfileError::InvalidUncertainty { index });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn z1_zero_hands_is_exactly_z0() {
        let config = BehaviourLatentConfig::new(3).unwrap();
        let z0 = config.profile(vec![0.4, -0.2, 1.2]).unwrap();
        let blueprint = BehaviourBlueprint::new(z0, vec![0.7, 1.0, 0.3]).unwrap();
        let z1 = PlayerBaseline::from_blueprint(&blueprint);
        assert_eq!(z1.profile(), blueprint.centre());
        assert_eq!(z1.uncertainty_std(), blueprint.prior_std());
        assert_eq!(z1.observed_hands(), 0);
    }

    #[test]
    fn z1_estimates_update_without_touching_blueprint() {
        let config = BehaviourLatentConfig::new(2).unwrap();
        let blueprint =
            BehaviourBlueprint::new(config.profile(vec![0.3, 0.5]).unwrap(), vec![1.0, 1.0])
                .unwrap();
        let mut z1 = PlayerBaseline::from_blueprint(&blueprint);
        z1.set_estimate(config.profile(vec![0.8, 0.0]).unwrap(), vec![0.4, 0.5], 10)
            .unwrap();
        assert_eq!(z1.observed_hands(), 10);
        assert_eq!(blueprint.centre().as_slice(), &[0.3, 0.5]);
        assert!(
            z1.set_estimate(config.profile(vec![0.0, 0.0]).unwrap(), vec![1.0, 1.0], 9)
                .is_err()
        );
        assert_eq!(z1.observed_hands(), 10);
    }

    #[test]
    fn latent_config_rejects_zero_dimension() {
        assert_eq!(
            BehaviourLatentConfig::new(0),
            Err(BehaviourLatentConfigError::ZeroDimension)
        );
    }

    #[test]
    fn accepts_embedding_with_exact_configured_dimension() {
        let config = BehaviourLatentConfig::new(3).unwrap();
        let profile = BehaviourProfile::new(config, vec![0.1, -0.3, 0.9]).unwrap();
        assert_eq!(profile.dimension(), 3);
        assert_eq!(config.dimension(), 3);
    }

    #[test]
    fn rejects_embedding_with_wrong_dimension() {
        let config = BehaviourLatentConfig::new(3).unwrap();
        assert_eq!(
            BehaviourProfile::new(config, vec![0.1, -0.3]),
            Err(BehaviourProfileError::DimensionMismatch {
                expected: 3,
                observed: 2,
            })
        );
    }

    #[test]
    fn config_rejects_profile_from_another_latent_width() {
        let source = BehaviourLatentConfig::new(2).unwrap();
        let target = BehaviourLatentConfig::new(3).unwrap();
        let profile = source.profile(vec![0.1, -0.3]).unwrap();

        assert_eq!(
            target.validate_profile(&profile),
            Err(BehaviourProfileError::DimensionMismatch {
                expected: 3,
                observed: 2,
            })
        );
    }

    #[test]
    fn rejects_non_finite_values() {
        let config = BehaviourLatentConfig::new(2).unwrap();
        assert_eq!(
            BehaviourProfile::new(config, vec![0.0, f32::INFINITY]),
            Err(BehaviourProfileError::NonFiniteValue { index: 1 })
        );
    }
}
