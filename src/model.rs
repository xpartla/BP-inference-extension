use ort::session::Session;

pub struct OnnxModel {
    session: Session,
}

pub struct Prediction {
    pub label: i64,
    pub probabilities: Vec<f32>,
}

impl OnnxModel {
    pub fn new(path: &str) -> Result<Self, ort::Error> {
        let session = Session::builder()?.commit_from_file(path)?;

        println!("Inputs:");
        for input in session.inputs.iter() {
            println!("  name = {}", input.name);
        }

        println!("Outputs:");
        for output in session.outputs.iter() {
            println!("  name = {}", output.name);
        }

        Ok(Self { session })
    }

    pub fn predict(&mut self, input: Vec<f32>) -> Result<Prediction, ort::Error> {
        let shape = vec![1, input.len()];
        let input_tensor = ort::value::Value::from_array((shape, input))?;

        let outputs = self.session.run(
            ort::inputs!["input" => &input_tensor]
        )?;

        let label = outputs["label"]
            .try_extract_tensor::<i64>()?
            .1[0];

        let probabilities = outputs["probabilities"]
            .try_extract_tensor::<f32>()?
            .1
            .to_vec();

        Ok(Prediction { label, probabilities })
    }
}