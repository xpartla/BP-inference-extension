use std::collections::HashMap;
use crate::artifacts::Artifacts;
use crate::encode::encode_categorical;
use crate::model::OnnxModel;
use crate::preprocess::{apply_feature_mask, preprocess_numeric};

pub fn predict(
    record_num: HashMap<String, Option<f64>>,
    record_cat: HashMap<String, Option<String>>,
    artifacts: &Artifacts,
    model: &mut OnnxModel,
) -> Result<f32, ort::Error> {
    let num = preprocess_numeric(&record_num, &artifacts.schema.numeric, &artifacts.numeric);
    let cat = encode_categorical(&record_cat, &artifacts.categorical.categories);
    let mut features = num;
    features.extend(cat);

    let final_features = apply_feature_mask(&features, &artifacts.feature_mask.indices);
    model.predict(final_features)
}