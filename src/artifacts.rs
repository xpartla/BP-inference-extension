use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;

#[derive(Deserialize)]
pub struct NumericMetadata {
    pub imputer: Imputer,
    pub scaler: Scaler,
    pub outliers: HashMap<String, OutlierBounds>,
}

#[derive(Deserialize)]
pub struct Imputer {
    pub medians: HashMap<String, f64>,
}

#[derive(Deserialize)]
pub struct Scaler {
    pub means: HashMap<String, f64>,
    pub stds: HashMap<String, f64>,
}

#[derive(Deserialize)]
pub struct OutlierBounds {
    pub lower_bound: f64,
    pub upper_bound: f64,
}

#[derive(Deserialize)]
pub struct CategoricalMetadata {
    pub categories: HashMap<String, Vec<String>>,
}

#[derive(Deserialize)]
pub struct FeatureMask {
    pub indices: Vec<usize>,
}

#[derive(Deserialize)]
pub struct Schema {
    pub numeric: Vec<String>,
    pub categorical: Vec<String>,
}

#[derive(Deserialize)]
pub struct Artifacts {
    pub schema: Schema,
    pub numeric: NumericMetadata,
    pub categorical: CategoricalMetadata,
    pub feature_mask: FeatureMask,
}

pub fn load_json<T: serde::de::DeserializeOwned>(path: &str) -> T {
    let file = File::open(path).expect("Failed to open artifact");
    serde_json::from_reader(file).expect("Failed to parse artifact")
}

impl Artifacts {
    pub fn load(dir: &str) -> Self {
        let schema: Schema =
            load_json(&format!("{}/schema.json", dir));

        let numeric: NumericMetadata =
            load_json(&format!("{}/numeric.json", dir));

        let categorical: CategoricalMetadata =
            load_json(&format!("{}/categorical.json", dir));

        let feature_mask: FeatureMask =
            load_json(&format!("{}/feature_mask.json", dir));

        Self {
            schema,
            numeric,
            categorical,
            feature_mask,
        }
    }
}

