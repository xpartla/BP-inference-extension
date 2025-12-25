use std::collections::HashMap;

pub fn encode_categorical(
    record: &HashMap<String, Option<String>>,
    cat_meta: &HashMap<String, Vec<String>>,
) -> Vec<f32> {
    let mut out = Vec::new();

    for (col, categories) in cat_meta {
        let value = record.get(col).and_then(|v| v.as_deref());

        for cat in categories {
            out.push(if Some(cat.as_str()) == value {
                1.0
            } else {
                0.0
            });
        }
    }
    out
}