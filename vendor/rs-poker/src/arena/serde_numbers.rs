//! Read statistical floats through JSON values when serde buffers tagged enums.
//! serde_json's arbitrary-precision number tokens are represented as maps in
//! serde's Content buffer. This adapter restores numeric values without routing
//! any monetary amount through a float.
use serde::{Deserialize, Deserializer, de::DeserializeOwned};
pub(crate) fn deserialize<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: DeserializeOwned,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    serde_json::from_value(value).map_err(serde::de::Error::custom)
}
