//! OHH currency numbers are decimal currency units; Rust values are cents.
//! serde_json arbitrary_precision preserves the original numeric token, so no
//! IEEE-754 conversion can lose a cent. Fractional cents fail explicitly.
use crate::Chips;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::str::FromStr;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum MoneyParseError {
    #[error("malformed currency amount: {0}")]
    Malformed(String),
    #[error("currency amount contains fractional cents: {0}")]
    FractionalCent(String),
    #[error("currency amount exceeds the Chips range: {0}")]
    Overflow(String),
}

pub fn parse_currency(text: &str) -> Result<Chips, MoneyParseError> {
    let malformed = || MoneyParseError::Malformed(text.into());
    let overflow = || MoneyParseError::Overflow(text.into());
    let (mantissa, exponent) = match text.split_once(['e', 'E']) {
        Some((m, e)) => (m, e.parse::<i32>().map_err(|_| malformed())?),
        None => (text, 0),
    };
    let (negative, mantissa) = match mantissa.strip_prefix('-') {
        Some(m) => (true, m),
        None => (false, mantissa),
    };
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    if whole.is_empty()
        || !whole.bytes().all(|b| b.is_ascii_digit())
        || !fraction.bytes().all(|b| b.is_ascii_digit())
        || (mantissa.contains('.') && fraction.is_empty())
    {
        return Err(malformed());
    }
    let mut digits = format!("{whole}{fraction}");
    let mut shift = i64::from(exponent) + 2 - fraction.len() as i64;
    while digits.ends_with('0') {
        digits.pop();
        shift += 1;
    }
    if digits.is_empty() {
        return Ok(0);
    }
    let mut value = digits.parse::<i128>().map_err(|_| overflow())?;
    if shift >= 0 {
        let scale = 10_i128
            .checked_pow(u32::try_from(shift).map_err(|_| overflow())?)
            .ok_or_else(overflow)?;
        value = value.checked_mul(scale).ok_or_else(overflow)?;
    } else {
        let divisor = 10_i128
            .checked_pow(u32::try_from(-shift).map_err(|_| overflow())?)
            .ok_or_else(|| MoneyParseError::FractionalCent(text.into()))?;
        if value % divisor != 0 {
            return Err(MoneyParseError::FractionalCent(text.into()));
        }
        value /= divisor;
    }
    if negative {
        value = -value;
    }
    Chips::try_from(value).map_err(|_| overflow())
}

pub fn format_currency(value: Chips) -> String {
    let amount = i128::from(value).abs();
    format!(
        "{}{whole}.{fraction:02}",
        if value < 0 { "-" } else { "" },
        whole = amount / 100,
        fraction = amount % 100
    )
}

pub fn serialize<S: Serializer>(value: &Chips, serializer: S) -> Result<S::Ok, S::Error> {
    serde_json::Number::from_str(&format_currency(*value))
        .map_err(serde::ser::Error::custom)?
        .serialize(serializer)
}
pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Chips, D::Error> {
    let number = serde_json::Number::deserialize(deserializer)?;
    parse_currency(&number.to_string()).map_err(serde::de::Error::custom)
}
pub mod optional {
    use super::*;
    pub fn serialize<S: Serializer>(
        value: &Option<Chips>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match value {
            Some(value) => serde_json::Number::from_str(&format_currency(*value))
                .map_err(serde::ser::Error::custom)?
                .serialize(serializer),
            None => serializer.serialize_none(),
        }
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<Chips>, D::Error> {
        Option::<serde_json::Number>::deserialize(deserializer)?
            .map(|n| parse_currency(&n.to_string()).map_err(serde::de::Error::custom))
            .transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Serialize, Deserialize)]
    struct Amount {
        #[serde(with = "super")]
        amount: Chips,
    }
    #[test]
    fn decimal_boundary_is_cent_perfect() {
        for chips in [
            0,
            1,
            2,
            106,
            100,
            12_345,
            9_007_199_254_740_993,
            Chips::MAX,
            Chips::MIN,
        ] {
            let json = serde_json::to_string(&Amount { amount: chips }).unwrap();
            let decoded: Amount = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.amount, chips);
        }
        assert_eq!(
            serde_json::to_string(&Amount { amount: 100 }).unwrap(),
            "{\"amount\":1.00}"
        );
        for (token, cents) in [
            ("0.01", 1),
            ("0.02", 2),
            ("1.06", 106),
            ("1e-2", 1),
            ("1.0600", 106),
        ] {
            assert_eq!(parse_currency(token).unwrap(), cents);
        }
        assert!(parse_currency("0.001").is_err());
        assert!(parse_currency("92233720368547758.08").is_err());
        assert!(parse_currency("1e9999999999").is_err());
        assert!(parse_currency("NaN").is_err());
    }
}
