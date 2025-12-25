use std::collections::HashMap;
use crate::{Artifacts, OnnxModel, Prediction};
use crate::encode::encode_categorical;
use crate::preprocess::{preprocess_numeric};

pub fn predict(
    record_num: HashMap<String, Option<f64>>,
    record_cat: HashMap<String, Option<String>>,
    artifacts: &Artifacts,
    model: &mut OnnxModel,
) -> Result<Prediction, ort::Error> {

    let numeric = preprocess_numeric(
        &record_num,
        &artifacts.schema.numeric,
        &artifacts.numeric,
    );

    let categorical = encode_categorical(
        &record_cat,
        &artifacts.categorical.categories,
    );

    let mut feature_map = numeric;
    feature_map.extend(categorical);

    let mut features =
        Vec::with_capacity(artifacts.feature_order.final_features.len());

    for name in &artifacts.feature_order.final_features {
        let v = feature_map.get(name).unwrap_or_else(|| {
            panic!("Feature '{}' missing after preprocessing", name)
        });
        features.push(*v);
    }

    assert_eq!(
        features.len(),
        artifacts.feature_order.final_features.len(),
    );

    model.predict(features)
}
