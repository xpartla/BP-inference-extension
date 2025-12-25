use crate::artifacts::NumericMetadata;
use std::collections::HashMap;

pub fn preprocess_numeric(
    record: &HashMap<String, Option<f64>>,
    num_cols: &[String],
    meta: &NumericMetadata,
) ->Vec<f32> {
    let mut out = Vec::with_capacity(num_cols.len());

    for col in num_cols {
        let mut val = record.get(col).and_then(|v| *v);

        if val.is_none() {
            val = meta.imputer.medians.get(col).copied();
        }

        let mut v = val.unwrap();
        if let Some(bounds) = meta.outliers.get(col) {
            if v < bounds.lower_bound || v > bounds.upper_bound {
                v = meta.scaler.means[col];
            }
        }
        v = (v - meta.scaler.means[col]) / meta.scaler.stds[col];
        out.push(v as f32);
    }
    out
}

pub fn apply_feature_mask(
    features: &[f32],
    mask: &[usize],
) -> Vec<f32> {
    mask.iter().map(|&i| features[i]).collect()
}