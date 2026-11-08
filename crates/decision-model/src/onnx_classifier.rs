use std::{collections::BTreeMap, path::Path, sync::Mutex, time::Instant};

use anyhow::{anyhow, bail, Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use serde::Deserialize;
use tokenizers::{Tokenizer, TruncationParams};

/// Max tokens per inference. BERT-family classifiers are trained at 512.
const MAX_TOKENS: usize = 512;

/// Result of scoring one text.
#[derive(Debug, Clone)]
pub struct Score {
    /// Probability (0..=1) of the positive label, e.g. "INJECTION".
    pub risk: f32,
    /// Probability per label, in label-index order.
    pub labels: Vec<(String, f32)>,
    pub latency_ms: f64,
}

/// An ONNX text classifier loaded from a model directory:
///
/// ```text
/// models/classifier/
///   config.json        id2label (label names)
///   tokenizer.json     Hugging Face tokenizer
///   onnx/model.onnx    the network
///   manifest.json      optional: name, revision, positive_label
/// ```
pub struct OnnxClassifier {
    id: String,
    tokenizer: Tokenizer,
    // `Session::run` takes `&mut self`; the Mutex lets one `OnnxClassifier` be shared via `Arc`.
    session: Mutex<Session>,
    input_names: Vec<String>,
    labels: Vec<String>,
    positive: usize,
}

#[derive(Deserialize)]
struct HfConfig {
    #[serde(default)]
    id2label: BTreeMap<String, String>,
    #[serde(default)]
    num_labels: Option<usize>,
}

#[derive(Deserialize, Default)]
struct Manifest {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    revision: Option<String>,
    #[serde(default)]
    positive_label: Option<String>,
}

impl OnnxClassifier {
    /// Load once at startup; loading per request costs hundreds of milliseconds.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let manifest: Manifest = match std::fs::read(dir.join("manifest.json")) {
            Ok(bytes) => serde_json::from_slice(&bytes).context("manifest.json")?,
            Err(_) => Manifest::default(),
        };
        let config: HfConfig =
            serde_json::from_slice(&std::fs::read(dir.join("config.json")).with_context(|| {
                format!(
                    "{}/config.json missing: run scripts/fetch_model.py",
                    dir.display()
                )
            })?)
            .context("config.json")?;
        let labels = labels_in_order(&config)?;
        let positive = positive_index(&labels, manifest.positive_label.as_deref())?;

        let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("tokenizer.json: {e}"))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_TOKENS,
                ..Default::default()
            }))
            .map_err(|e| anyhow!("tokenizer truncation: {e}"))?;
        tokenizer.with_padding(None); // one text per run: no padding needed

        let model_path = dir.join("onnx/model.onnx");
        if !model_path.exists() {
            bail!(
                "{} missing: run scripts/export_onnx.sh if only safetensors exist",
                model_path.display()
            );
        }
        let env = ort::init()
            .with_name("classifier")
            .build()
            .map_err(|e| anyhow!("onnx runtime init: {e}"))?;
        let session = Session::builder(&env)
            .map_err(|e| anyhow!("session builder: {e}"))?
            // Minimal ONNX Runtime builds reject optimization settings; keep going without them.
            .with_optimization_level(GraphOptimizationLevel::All)
            .unwrap_or_else(|e| e.recover())
            .commit_from_file(&model_path)
            .map_err(|e| anyhow!("{}: {e}", model_path.display()))?;
        let input_names = session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();

        let name = manifest.name.unwrap_or_else(|| "classifier".into());
        let id = match manifest.revision {
            Some(rev) => format!("{name}@{}", &rev[..rev.len().min(12)]),
            None => name,
        };

        Ok(Self {
            id,
            tokenizer,
            session: Mutex::new(session),
            input_names,
            labels,
            positive,
        })
    }

    /// Model id for the audit log, e.g. `classifier@f171d7baecaf`.
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Blocking CPU work. From async code, call inside `tokio::task::spawn_blocking`.
    pub fn score(&self, text: &str) -> Result<Score> {
        let start = Instant::now();
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("tokenize: {e}"))?;
        let len = enc.get_ids().len();
        let as_i64 = |v: &[u32]| v.iter().map(|&x| i64::from(x)).collect::<Vec<_>>();

        // Feed only the inputs this model declares (some exports omit token_type_ids).
        let mut inputs = Vec::with_capacity(self.input_names.len());
        for name in &self.input_names {
            let data = match name.as_str() {
                "input_ids" => as_i64(enc.get_ids()),
                "attention_mask" => as_i64(enc.get_attention_mask()),
                "token_type_ids" => as_i64(enc.get_type_ids()),
                other => bail!("model expects unsupported input {other:?}"),
            };
            inputs.push((name.clone(), Tensor::from_array(([1usize, len], data))?));
        }

        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow!("session lock poisoned"))?;
        let outputs = session.run(inputs)?;
        let (_, logits) = outputs[0].try_extract_tensor::<f32>()?;
        if logits.len() != self.labels.len() {
            bail!(
                "model returned {} logits for {} labels",
                logits.len(),
                self.labels.len()
            );
        }
        let probs = softmax(logits);

        Ok(Score {
            risk: probs[self.positive],
            labels: self.labels.iter().cloned().zip(probs).collect(),
            latency_ms: start.elapsed().as_secs_f64() * 1e3,
        })
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max); // subtract max: avoids exp overflow
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// `{"0": "SAFE", "1": "INJECTION"}` → `["SAFE", "INJECTION"]`.
/// No `id2label`: Hugging Face's default, `LABEL_0..LABEL_{n-1}` with n = `num_labels` or 2.
fn labels_in_order(config: &HfConfig) -> Result<Vec<String>> {
    if config.id2label.is_empty() {
        return Ok((0..config.num_labels.unwrap_or(2))
            .map(|i| format!("LABEL_{i}"))
            .collect());
    }
    let mut pairs = config
        .id2label
        .iter()
        .map(|(k, v)| {
            Ok((
                k.parse::<usize>()
                    .with_context(|| format!("id2label key {k:?}"))?,
                v.clone(),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    pairs.sort_by_key(|(i, _)| *i);
    if pairs.iter().enumerate().any(|(pos, (i, _))| pos != *i) {
        bail!("id2label indices are not 0..n");
    }
    Ok(pairs.into_iter().map(|(_, v)| v).collect())
}

/// Which label counts as "risky". Explicit name from manifest, or index 1 for binary models.
fn positive_index(labels: &[String], positive: Option<&str>) -> Result<usize> {
    match positive {
        Some(p) => labels
            .iter()
            .position(|l| l.eq_ignore_ascii_case(p))
            .with_context(|| format!("positive_label {p:?} not in {labels:?}")),
        None if labels.len() == 2 => Ok(1),
        None => bail!(
            "{} labels {labels:?}: set positive_label in models.json",
            labels.len()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one_and_keeps_order() {
        let p = softmax(&[1.0, 3.0, 2.0]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(p[1] > p[2] && p[2] > p[0]);
    }

    #[test]
    fn softmax_handles_large_logits() {
        let p = softmax(&[1000.0, 1000.0]);
        assert!((p[0] - 0.5).abs() < 1e-6);
    }

    fn config(pairs: &[(&str, &str)], num_labels: Option<usize>) -> HfConfig {
        let id2label = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        HfConfig {
            id2label,
            num_labels,
        }
    }

    #[test]
    fn labels_sorted_by_index() {
        assert!(labels_in_order(&config(&[("10", "c"), ("2", "b"), ("0", "a")], None)).is_err()); // gap
        let c = config(&[("1", "INJECTION"), ("0", "SAFE")], None);
        assert_eq!(labels_in_order(&c).unwrap(), ["SAFE", "INJECTION"]);
    }

    #[test]
    fn labels_default_like_hugging_face() {
        assert_eq!(
            labels_in_order(&config(&[], None)).unwrap(),
            ["LABEL_0", "LABEL_1"]
        );
        assert_eq!(labels_in_order(&config(&[], Some(3))).unwrap().len(), 3);
    }

    #[test]
    fn positive_label_resolution() {
        let two = vec!["SAFE".to_string(), "INJECTION".to_string()];
        assert_eq!(positive_index(&two, None).unwrap(), 1);
        assert_eq!(positive_index(&two, Some("safe")).unwrap(), 0);
        assert!(positive_index(&two, Some("nope")).is_err());
        let three = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        assert!(positive_index(&three, None).is_err());
    }
}
