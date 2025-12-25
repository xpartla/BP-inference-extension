use std::collections::HashMap;
use ml_inference::artifacts::{Artifacts, load_json};
use ml_inference::{predict, OnnxModel};

#[test]
fn parity_single_sample() {
    let artifacts = Artifacts::load("artifacts");
    let mut sample: serde_json::Value = load_json("artifacts/parity/sample_000.json");

    let mut record_num = HashMap::new();
    let mut record_cat = HashMap::new();

    for (k, v) in sample["numeric"].as_object().unwrap() {
        record_num.insert(k.clone(), v.as_f64());
    }

    for (k, v) in sample["categorical"].as_object().unwrap() {
        record_cat.insert(k.clone(), v.as_str().map(|s| s.to_string()));
    }

    let expected_label = sample["label"].as_i64().unwrap();
    let expected_probs: Vec<f32> = sample["probabilities"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();

    let mut model = OnnxModel::new("artifacts/lgbm.onnx").unwrap();
    let pred = predict(record_num, record_cat, &artifacts, &mut model).unwrap();

    assert_eq!(pred.label, expected_label);

    for (a, b) in pred.probabilities.iter().zip(expected_probs.iter()) {
        assert!((a - b).abs() < 1e-4);
    }
}