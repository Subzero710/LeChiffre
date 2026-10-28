use std::fmt;

/// Opaque behavioural representation shared by Analyzer and Generator.
#[derive(Debug, Clone, PartialEq)]
pub struct BehaviourProfile {
    embedding: Vec<f32>,
}

impl BehaviourProfile {
    pub fn new(embedding: Vec<f32>) -> Result<Self, BehaviourProfileError> {
        if embedding.is_empty() {
            return Err(BehaviourProfileError::EmptyEmbedding);
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
    EmptyEmbedding,
    NonFiniteValue { index: usize },
}

impl fmt::Display for BehaviourProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyEmbedding => write!(f, "behaviour profile embedding cannot be empty"),
            Self::NonFiniteValue { index } => {
                write!(f, "behaviour profile contains a non-finite value at {index}")
            }
        }
    }
}

impl std::error::Error for BehaviourProfileError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_finite_embedding() {
        let profile = BehaviourProfile::new(vec![0.1, -0.3, 0.9]).unwrap();
        assert_eq!(profile.dimension(), 3);
    }

    #[test]
    fn rejects_empty_embedding() {
        assert_eq!(
            BehaviourProfile::new(vec![]),
            Err(BehaviourProfileError::EmptyEmbedding)
        );
    }

    #[test]
    fn rejects_non_finite_values() {
        assert_eq!(
            BehaviourProfile::new(vec![0.0, f32::INFINITY]),
            Err(BehaviourProfileError::NonFiniteValue { index: 1 })
        );
    }
}
