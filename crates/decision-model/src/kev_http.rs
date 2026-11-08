//! Client for a local Kev server (`scripts/kev_server.sh`, `POST /v1/systemone`).
//!
//! Bridge until Kev runs in-process via candle: same questions, same answers, one localhost hop.

use std::collections::BTreeMap;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A typed question. Kev answers each one independently from the shared state.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    /// Yes / no. Answer: probability of yes.
    Noul { instructions: String },
    /// One of named options (name → description). Answer: probability per option.
    Choice {
        instructions: String,
        criteria: BTreeMap<String, String>,
    },
}

#[derive(Serialize)]
struct Request<'a> {
    state: &'a Value,
    model: &'a str,
    questions: &'a BTreeMap<String, Question>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Noul {
        noul: f32,
    },
    Choice {
        choice: String,
        confidence: f32,
        probabilities: BTreeMap<String, f32>,
    },
    Score {
        score: f32,
        probabilities: BTreeMap<String, f32>,
    },
}

#[derive(Debug, Clone, Deserialize)]
pub struct Response {
    pub answers: BTreeMap<String, Answer>,
    /// Model time reported by the server (excludes HTTP overhead).
    pub latency_ms: Option<f64>,
}

pub struct KevHttp {
    url: String,
    client: reqwest::Client,
}

impl KevHttp {
    /// `base` e.g. `http://127.0.0.1:8008`.
    pub fn new(base: &str) -> Self {
        Self {
            url: format!("{}/v1/systemone", base.trim_end_matches('/')),
            client: reqwest::Client::new(),
        }
    }

    /// Ask typed questions about `state` (text, or a JSON object Kev renders as labelled text).
    pub async fn ask(
        &self,
        state: &Value,
        questions: &BTreeMap<String, Question>,
    ) -> Result<Response> {
        let res = self
            .client
            .post(&self.url)
            .json(&Request {
                state,
                model: "kev-latest",
                questions,
            })
            .send()
            .await
            .with_context(|| format!("POST {}: is scripts/kev_server.sh running?", self.url))?;
        let status = res.status();
        if !status.is_success() {
            bail!(
                "kev server {status}: {}",
                res.text().await.unwrap_or_default()
            );
        }
        res.json().await.context("kev server response")
    }

    /// Probability (0..=1) that the answer to a yes/no question about `state` is yes.
    pub async fn noul(&self, state: &str, instructions: &str) -> Result<f32> {
        let questions = BTreeMap::from([(
            "q".to_string(),
            Question::Noul {
                instructions: instructions.into(),
            },
        )]);
        match self
            .ask(&Value::String(state.into()), &questions)
            .await?
            .answers
            .remove("q")
        {
            Some(Answer::Noul { noul }) => Ok(noul),
            other => bail!("expected a noul answer, got {other:?}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_matches_system_one_schema() {
        let state = Value::String("hi".into());
        let questions = BTreeMap::from([
            (
                "inj".to_string(),
                Question::Noul {
                    instructions: "Injection?".into(),
                },
            ),
            (
                "team".to_string(),
                Question::Choice {
                    instructions: "Who?".into(),
                    criteria: BTreeMap::from([("billing".into(), "Charges".into())]),
                },
            ),
        ]);
        let json = serde_json::to_value(Request {
            state: &state,
            model: "kev-latest",
            questions: &questions,
        })
        .unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "state": "hi",
                "model": "kev-latest",
                "questions": {
                    "inj": {"type": "noul", "instructions": "Injection?"},
                    "team": {"type": "choice", "instructions": "Who?", "criteria": {"billing": "Charges"}}
                }
            })
        );
    }

    #[test]
    fn parses_server_response() {
        let body = r#"{"model":"kev-latest","latency_ms":41.2,"usage":{"input_tokens":30,"output_tokens":9},
            "answers":{"inj":{"type":"noul","noul":0.8123},
                       "team":{"type":"choice","choice":"billing","confidence":0.7,"probabilities":{"billing":0.85,"shipping":0.15}}}}"#;
        let r: Response = serde_json::from_str(body).unwrap();
        assert!(matches!(r.answers["inj"], Answer::Noul { noul } if (noul - 0.8123).abs() < 1e-6));
        assert!(matches!(&r.answers["team"], Answer::Choice { choice, .. } if choice == "billing"));
        assert_eq!(r.latency_ms, Some(41.2));
    }
}
