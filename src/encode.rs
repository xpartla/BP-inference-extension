use std::collections::HashMap;

pub fn encode_categorical(
    record: &HashMap<String, Option<String>>,
    cat_meta: &HashMap<String, Vec<String>>,
) -> HashMap<String, f32> {
    let mut out = HashMap::new();

    for (col, categories) in cat_meta {
        let value = record.get(col).and_then(|v| v.as_deref());

        for cat in categories {
            let feature_name = format!("{}_{}", col, cat);
            out.insert(
                feature_name,
                if Some(cat.as_str()) == value { 1.0 } else { 0.0 },
            );
        }
    }

    out
}
