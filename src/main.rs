use std::collections::HashMap;
use serde_json::Value;
use ml_inference::{run_prediction, Artifacts, OnnxModel, predict};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let json_str = std::fs::read_to_string("artifacts/parity/sample_000.json")?;
    let data: Value = serde_json::from_str(&json_str)?;

    let mut record_num: HashMap<String, Option<f64>> = HashMap::new();
    if let Some(obj) = data["record_num"].as_object() {
        for (key, value) in obj {
            record_num.insert(
                key.clone(),
                value.as_f64()
            );
        }
    }

    let mut record_cat: HashMap<String, Option<String>> = HashMap::new();
    if let Some(obj) = data["record_cat"].as_object() {
        for (key, value) in obj {
            record_cat.insert(
                key.clone(),
                value.as_str().map(|s| s.to_string())
            );
        }
    }

    println!("record_num has {} entries", record_num.len());
    println!("record_cat has {} entries", record_cat.len());
    println!("NACCID: {:?}", record_cat.get("NACCID"));

    let prediction = run_prediction(
        record_num,
        record_cat,
        "artifacts",
        "artifacts/lgbm.onnx"
    )?;

    println!("label: {}", prediction.label);
    println!("probabilities: {:?}", prediction.probabilities);

    Ok(())
}