use ort::session::Session;

pub struct OnnxModel {
    session: Session,
}

impl OnnxModel {
    pub fn new(path: &str) -> Result<Self, ort::Error> {
        let session = Session::builder()?
            .commit_from_file(path)?;

        Ok(Self { session })
    }

    pub fn predict(&mut self, input: Vec<f32>) -> Result<f32, ort::Error> {
        let input_len = input.len();

        let shape = vec![1, input_len];
        let input_tensor = ort::value::Value::from_array((shape, input))?;

        let outputs = self.session.run(ort::inputs!["input" => &input_tensor])?;

        let output = outputs["output"].try_extract_tensor::<f32>()?;
        let data = output.1;

        Ok(data[0])
    }
}