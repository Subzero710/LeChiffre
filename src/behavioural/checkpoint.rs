//! Strict JSON primitives shared by the single A/G checkpoint contract.
use serde_json::{Value, json};

use super::DenseLayer;

pub(crate) fn exact_keys(value: &Value, keys: &[&str]) -> Result<(), String> {
    let object = value.as_object().ok_or("expected JSON object")?;
    if object.len() != keys.len() || keys.iter().any(|key| !object.contains_key(*key)) {
        return Err(format!("unexpected JSON object keys; expected {keys:?}"));
    }
    Ok(())
}

pub(crate) fn number(value: &Value) -> Result<f64, String> {
    value
        .as_f64()
        .filter(|x| x.is_finite())
        .ok_or_else(|| "expected finite JSON number".into())
}

pub(crate) fn unsigned(value: &Value) -> Result<u64, String> {
    value
        .as_u64()
        .ok_or_else(|| "expected unsigned JSON integer".into())
}

pub(crate) fn dimension(value: &Value) -> Result<usize, String> {
    usize::try_from(unsigned(value)?).map_err(|_| "integer does not fit usize".into())
}

pub(crate) fn float_array(value: &Value) -> Result<Vec<f64>, String> {
    value
        .as_array()
        .ok_or("expected JSON array")?
        .iter()
        .map(number)
        .collect()
}

pub(crate) fn dense_value(layer: &DenseLayer) -> Value {
    json!({"input": layer.input_width(), "output": layer.output_width(),
           "weights":layer.weights(), "bias":layer.biases()})
}

pub(crate) fn dense_from_value(value: &Value) -> Result<DenseLayer, String> {
    exact_keys(value, &["input", "output", "weights", "bias"])?;
    let weights = float_array(&value["weights"])?
        .into_iter()
        .map(|x| {
            let y = x as f32;
            if !y.is_finite() {
                Err("weight outside f32 range".to_string())
            } else {
                Ok(y)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let bias = float_array(&value["bias"])?
        .into_iter()
        .map(|x| {
            let y = x as f32;
            if !y.is_finite() {
                Err("bias outside f32 range".to_string())
            } else {
                Ok(y)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    DenseLayer::new(
        dimension(&value["input"])?,
        dimension(&value["output"])?,
        weights,
        bias,
    )
    .map_err(|err| err.to_string())
}
