use std::collections::HashMap;

use crate::adapters::PlayerId;

pub const IDENTITY_NAMESPACE: &str = "poker-with-the-boys";

/// Versioned, explicit pseudonymous subject registry for this dataset.
///
/// Existing IDs are immutable. New canonical identities must be appended with
/// new IDs; never renumber existing entries. Names are ingestion metadata only
/// and are not persisted in the behavioural Parquet tables.
pub const SUBJECT_REGISTRY_VERSION: u16 = 1;
pub const SUBJECT_REGISTRY: &[(u64, &str)] = &[
    (1, "Alex"),
    (2, "Annie"),
    (3, "Arash"),
    (4, "David"),
    (5, "Ethan"),
    (6, "George"),
    (7, "James"),
    (8, "Jonah"),
    (9, "Josh"),
    (10, "Max"),
    (11, "Prilik"),
    (12, "Sam"),
    (13, "Spencer"),
    (14, "Stephen"),
];

pub fn subject_ids() -> HashMap<PlayerId, u64> {
    SUBJECT_REGISTRY
        .iter()
        .map(|&(subject_id, canonical_name)| {
            (
                PlayerId::from_source(IDENTITY_NAMESPACE, canonical_name),
                subject_id,
            )
        })
        .collect()
}

/// Dataset-specific identity resolver transcribed from
/// arashrai/poker-with-the-boys/regex_based_graph_night.py.
/// Unknown identities are rejected, never guessed.
pub fn normalize_player_name(name: &str) -> Option<String> {
    let lower = name.trim().to_lowercase();
    match lower.as_str() {
        "peelic" => Some("Prilik".to_string()),
        "arashh" => Some("Arash".to_string()),
        "ash" => Some("Arash".to_string()),
        "arash" => Some("Arash".to_string()),
        "susan" => Some("Arash".to_string()),
        "annie" => Some("Annie".to_string()),
        "spencer" => Some("Spencer".to_string()),
        "spenner" => Some("Spencer".to_string()),
        "spenny" => Some("Spencer".to_string()),
        "spenny2" => Some("Spencer".to_string()),
        "speny" => Some("Spencer".to_string()),
        "spange" => Some("Ethan".to_string()),
        "biz" => Some("Alex".to_string()),
        "guest" => Some("George".to_string()),
        "stevo-ipad" => Some("Stephen".to_string()),
        "stevo" => Some("Stephen".to_string()),
        "stephen" => Some("Stephen".to_string()),
        "david" => Some("David".to_string()),
        "daveed" => Some("David".to_string()),
        "daveeed" => Some("David".to_string()),
        "daveeeed" => Some("David".to_string()),
        "eveeeda" => Some("David".to_string()),
        "dve-eed" => Some("David".to_string()),
        "jerms" => Some("James".to_string()),
        "jems" => Some("James".to_string()),
        "gems" => Some("James".to_string()),
        "james" => Some("James".to_string()),
        "josh" => Some("Josh".to_string()),
        "jonah" => Some("Jonah".to_string()),
        "jonah dlin" => Some("Jonah".to_string()),
        "max" => Some("Max".to_string()),
        "sam" => Some("Sam".to_string()),
        "daveeeeeed" => Some("David".to_string()),
        "daveeeeeeed" => Some("David".to_string()),
        "dave" => Some("David".to_string()),
        "dave-eed" => Some("David".to_string()),
        "daveod" => Some("David".to_string()),
        "stevo-tesla" => Some("Stephen".to_string()),
        _ if !lower.is_empty() && lower.chars().all(|c| "george".contains(c)) => {
            Some("George".to_string())
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    #[test]
    fn normalizes_known_aliases() {
        assert_eq!(normalize_player_name("goero"), Some("George".to_string()));
        assert_eq!(
            normalize_player_name("STEVO-IPAD"),
            Some("Stephen".to_string())
        );
        assert_eq!(normalize_player_name("peelic"), Some("Prilik".to_string()));
        assert_eq!(normalize_player_name("Xavier"), None);
    }

    #[test]
    fn subject_registry_is_unique_and_stable() {
        let ids = subject_ids();
        assert_eq!(ids.len(), SUBJECT_REGISTRY.len());

        let mut subject_ids_seen = HashSet::new();
        let mut canonical_names_seen = HashSet::new();
        for &(subject_id, canonical_name) in SUBJECT_REGISTRY {
            assert!(subject_ids_seen.insert(subject_id));
            assert!(canonical_names_seen.insert(canonical_name));
            assert_eq!(
                ids.get(&PlayerId::from_source(IDENTITY_NAMESPACE, canonical_name)),
                Some(&subject_id)
            );
        }

        assert_eq!(SUBJECT_REGISTRY_VERSION, 1);
    }
}
