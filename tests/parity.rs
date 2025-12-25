use std::collections::HashMap;
use ml_inference::artifacts::{Artifacts, load_json};
use ml_inference::{predict, OnnxModel};

#[test]
fn parity_single_sample() {
    let artifacts = Artifacts::load("artifacts");
    let sample: serde_json::Value = load_json("artifacts/parity/sample_000.json");

    let mut record_num = HashMap::new();
    if let Some(obj) = sample["record_num"].as_object() {
        for (k, v) in obj {
            record_num.insert(k.clone(), v.as_f64());
        }
    }

    let mut record_cat = HashMap::new();
    if let Some(obj) = sample["record_cat"].as_object() {
        for (k, v) in obj {
            record_cat.insert(k.clone(), v.as_str().map(|s| s.to_string()));
        }
    }

    let expected_label = sample["expected_label"].as_i64().unwrap();
    let expected_probs: Vec<f32> = sample["expected_probabilities"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();

    println!("Expected label: {}", expected_label);
    println!("Expected probabilities: {:?}", expected_probs);
    println!("Number of numeric features: {}", record_num.len());
    println!("Number of categorical features: {}", record_cat.len());

    let mut model = OnnxModel::new("artifacts/lgbm_tensor.onnx").unwrap();
    let pred = predict(record_num, record_cat, &artifacts, &mut model).unwrap();

    println!("Predicted label: {}", pred.label);
    println!("Predicted probabilities: {:?}", pred.probabilities);

    assert_eq!(pred.label, expected_label,
               "Label mismatch: expected {} but got {}", expected_label, pred.label);

    assert_eq!(pred.probabilities.len(), expected_probs.len(),
               "Probability vector length mismatch");

    for (i, (actual, expected)) in pred.probabilities.iter().zip(expected_probs.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(diff < 1e-4,
                "Probability {} mismatch: expected {} but got {} (diff: {})",
                i, expected, actual, diff);
    }

    println!("✓ Parity test passed!");
}

#[test]
fn parity_all_samples() {
    let artifacts = Artifacts::load("artifacts");

    for i in 0..10 {
        let filename = format!("artifacts/parity/sample_{:03}.json", i);
        println!("\n=== Testing {} ===", filename);

        let sample: serde_json::Value = load_json(&filename);

        let mut record_num = HashMap::new();
        if let Some(obj) = sample["record_num"].as_object() {
            for (k, v) in obj {
                record_num.insert(k.clone(), v.as_f64());
            }
        }

        let mut record_cat = HashMap::new();
        if let Some(obj) = sample["record_cat"].as_object() {
            for (k, v) in obj {
                record_cat.insert(k.clone(), v.as_str().map(|s| s.to_string()));
            }
        }

        let expected_label = sample["expected_label"].as_i64().unwrap();
        let expected_probs: Vec<f32> = sample["expected_probabilities"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();

        let mut model = OnnxModel::new("artifacts/lgbm_tensor.onnx").unwrap();
        let pred = predict(record_num, record_cat, &artifacts, &mut model).unwrap();

        assert_eq!(pred.label, expected_label,
                   "Sample {}: Label mismatch", i);

        for (j, (actual, expected)) in pred.probabilities.iter().zip(expected_probs.iter()).enumerate() {
            let diff = (actual - expected).abs();
            assert!(diff < 1e-4,
                    "Sample {}, Probability {}: expected {} but got {} (diff: {})",
                    i, j, expected, actual, diff);
        }

        println!("✓ Sample {} passed", i);
    }
}