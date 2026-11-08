//! Local decision models.
//!
//! - `KevHttp`: asks Kev (local reference server) typed questions over HTTP.
//! - `OnnxClassifier`: loads an ONNX sequence classifier from disk and scores text for risk.

mod kev_http;
mod onnx_classifier;

pub use kev_http::{Answer, KevHttp, Question, Response};
pub use onnx_classifier::{OnnxClassifier, Score};
