use crate::model::OnnxModel;

mod artifacts;
mod schema;
mod preprocess;
mod encode;
mod model;
mod predict;

fn main() {
    let model = OnnxModel::new("artifacts/lgbm.onnx");

    let dummy = vec![0.0; 50];
    let p = model.expect("REASON").predict(dummy);
    println!("Prediction: {:?}", p);
}
// TODO: update MSVC to test build
// TODO: end to end test
// TODO: parity test
