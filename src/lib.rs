use std::collections::HashMap;

pub mod artifacts;
pub mod schema;
pub mod preprocess;
pub mod encode;
pub mod model;
pub mod predict;

pub use artifacts::{Artifacts, load_json};
pub use model::{OnnxModel, Prediction};
pub use predict::predict;

pub fn run_prediction(
    record_num: HashMap<String, Option<f64>>,
    record_cat: HashMap<String, Option<String>>,
    artifacts_path: &str,
    model_path: &str,
) -> Result<Prediction, Box<dyn std::error::Error>> {
    let artifacts = Artifacts::load(artifacts_path);
    let mut model = OnnxModel::new(model_path)?;

    let prediction = predict(record_num, record_cat, &artifacts, &mut model)?;

    Ok(prediction)
}