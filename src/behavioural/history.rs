use super::{BehaviourSequence, Position};

/// One real hand in which the analyzed subject occupied `subject_position`.
#[derive(Debug, Clone, PartialEq)]
pub struct SubjectHand {
    sequence: BehaviourSequence,
    subject_position: Position,
}

impl SubjectHand {
    pub fn new(sequence: BehaviourSequence, subject_position: Position) -> Self {
        Self {
            sequence,
            subject_position,
        }
    }

    pub fn sequence(&self) -> &BehaviourSequence {
        &self.sequence
    }

    pub fn subject_position(&self) -> Position {
        self.subject_position
    }

    pub fn into_sequence(self) -> BehaviourSequence {
        self.sequence
    }
}

/// Behavioural observations of one subject across multiple real poker hands.
///
/// Subject identity is intentionally kept outside the Behavioural Engine. The
/// ingestion layer is responsible for grouping only hands that belong to the
/// same source-observed player.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BehaviourHistory {
    hands: Vec<SubjectHand>,
}

impl BehaviourHistory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, hand: SubjectHand) {
        self.hands.push(hand);
    }

    pub fn hands(&self) -> &[SubjectHand] {
        &self.hands
    }

    pub fn is_empty(&self) -> bool {
        self.hands.is_empty()
    }

    pub fn len(&self) -> usize {
        self.hands.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_subject_position_per_hand() {
        let mut history = BehaviourHistory::new();
        history.push(SubjectHand::new(
            BehaviourSequence::new("hand-1"),
            Position::UnderTheGun,
        ));
        history.push(SubjectHand::new(
            BehaviourSequence::new("hand-2"),
            Position::Button,
        ));

        assert_eq!(history.len(), 2);
        assert_eq!(history.hands()[0].subject_position(), Position::UnderTheGun);
        assert_eq!(history.hands()[1].subject_position(), Position::Button);
    }
}
