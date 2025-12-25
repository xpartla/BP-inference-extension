use crate::model::OnnxModel;

mod artifacts;
mod schema;
mod preprocess;
mod encode;
mod model;
mod predict;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = OnnxModel::new("artifacts/lgbm.onnx")?;

    let dummy = vec![0.0; 50];
    let prediction = model.predict(dummy)?;

    println!("label: {}", prediction.label);
    println!("probabilities: {:?}", prediction.probabilities);

    Ok(())
}
