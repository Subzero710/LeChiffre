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
        }
    }
}

impl std::error::Error for BehaviourProfileError {}

#[cfg(test)]
mod tests {
    use super::*;

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
