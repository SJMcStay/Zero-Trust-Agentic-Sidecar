//! Alerts (`src/alerts.rs`): tell the human when the sidecar blocks or flags something, and ask
//! them about borderline calls.
//!
//! - [`desktop`]: a native notification sent by the sidecar itself (`[alerts] desktop = true`):
//!   macOS `osascript`, Linux `notify-send`. Rate-limited, fire-and-forget.
//! - [`AlertHub`]: the recent alerts as a stream, `GET /v1/alerts` (SSE, loopback only, bearer
//!   token), read by `scripts/alert_notifier.py`. A model score just over its threshold (within
//!   `human_approval_band`) waits up to `human_approval_timeout_ms` for the human's "Allow once" or
//!   "Block" (`POST /v1/alerts/:id/decision`). No notifier listening, no answer or a timeout all
//!   mean block. Approval never overrides a deterministic policy rule.
//!
//! While a notifier is listening, callers skip [`desktop`] so each event pops up once.

use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{
        sse::{Event, KeepAlive},
        IntoResponse, Sse,
    },
    Json,
};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::process::Stdio;
use std::{
    collections::{HashMap, VecDeque},
    convert::Infallible,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::sync::{broadcast, oneshot};

use crate::routes::AppState;

// ==============================================================================
// Desktop notification from the sidecar itself
// ==============================================================================

/// At most one notification per this interval (a burst of blocked retries shows once).
pub const MIN_GAP: Duration = Duration::from_secs(3);

static LAST_SHOWN: Mutex<Option<Instant>> = Mutex::new(None);

/// Shows a notification without waiting for it. Must run inside a Tokio runtime.
pub fn desktop(title: &str, message: &str) {
    {
        let mut last = LAST_SHOWN.lock().unwrap_or_else(|e| e.into_inner());
        if last.is_some_and(|shown| shown.elapsed() < MIN_GAP) {
            return;
        }
        *last = Some(Instant::now());
    }

    let (program, args) = if cfg!(target_os = "macos") {
        let script = format!(
            "display notification {} with title {}",
            applescript_string(message),
            applescript_string(title)
        );
        ("osascript", vec!["-e".to_string(), script])
    } else {
        ("notify-send", vec![title.to_string(), message.to_string()])
    };

    match tokio::process::Command::new(program)
        .args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(mut child) => {
            tokio::spawn(async move {
                let _ = child.wait().await; // reap the process
            });
        }
        Err(err) => tracing::debug!(error = %err, program, "desktop alert unavailable"),
    }
}

/// An AppleScript string literal: quotes and backslashes escaped, kept on one line.
fn applescript_string(text: &str) -> String {
    let one_line: String = text
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    format!(
        "\"{}\"",
        one_line.replace('\\', "\\\\").replace('"', "\\\"")
    )
}

// ==============================================================================
// Alert stream and one-time human approval
// ==============================================================================

const HISTORY_LIMIT: usize = 128;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub time: String,
    /// The calling agent (`User-Agent`, e.g. `claude-cli/2.1.319`).
    pub agent: String,
    pub session_id: Option<String>,
    pub route: String,
    /// What was blocked: an endpoint, a tool, or the tool result that carried an injection.
    pub target: String,
    pub rule: String,
    pub reason: String,
    pub injection_confidence: Option<f32>,
    pub malicious_intent: Option<f32>,
    pub approval_required: bool,
    /// `pending`, `blocked`, `allow_once`, `block`, or `monitor` (would have blocked).
    pub action: String,
}

#[derive(Debug, Deserialize)]
pub struct Decision {
    pub decision: String,
}

struct Inner {
    history: Mutex<VecDeque<Alert>>,
    tx: broadcast::Sender<Alert>,
    pending: Mutex<HashMap<String, oneshot::Sender<bool>>>,
}

#[derive(Clone)]
pub struct AlertHub(Arc<Inner>);

impl Default for AlertHub {
    fn default() -> Self {
        let (tx, _) = broadcast::channel(HISTORY_LIMIT);
        Self(Arc::new(Inner {
            history: Mutex::new(VecDeque::new()),
            tx,
            pending: Mutex::new(HashMap::new()),
        }))
    }
}

impl AlertHub {
    pub fn publish(&self, alert: Alert) {
        if let Ok(mut history) = self.0.history.lock() {
            if history.len() == HISTORY_LIMIT {
                history.pop_front();
            }
            history.push_back(alert.clone());
        }
        let _ = self.0.tx.send(alert);
    }

    /// A notifier is subscribed to the stream.
    pub fn has_listeners(&self) -> bool {
        self.0.tx.receiver_count() > 0
    }

    /// Recent alerts, oldest first.
    pub fn recent(&self) -> Vec<Alert> {
        self.0
            .history
            .lock()
            .map(|h| h.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Publishes `alert` as pending and waits for the human. True only for "Allow once" within
    /// `timeout`. With nobody listening, blocks at once instead of holding the request.
    pub async fn request_approval(&self, mut alert: Alert, timeout: Duration) -> bool {
        if !self.has_listeners() {
            alert.approval_required = false;
            alert.action = "blocked".into();
            self.publish(alert);
            return false;
        }
        let id = alert.id.clone();
        let (tx, rx) = oneshot::channel();
        if let Ok(mut pending) = self.0.pending.lock() {
            pending.insert(id.clone(), tx);
        }
        self.publish(alert);
        let approved = tokio::time::timeout(timeout, rx)
            .await
            .ok()
            .and_then(Result::ok)
            .unwrap_or(false);
        if let Ok(mut pending) = self.0.pending.lock() {
            pending.remove(&id);
        }
        self.update_action(&id, if approved { "allow_once" } else { "blocked" });
        approved
    }

    fn update_action(&self, id: &str, action: &str) {
        let updated = self.0.history.lock().ok().and_then(|mut h| {
            h.iter_mut().find(|a| a.id == id).map(|a| {
                a.action = action.to_string();
                a.clone()
            })
        });
        if let Some(alert) = updated {
            let _ = self.0.tx.send(alert);
        }
    }

    fn decide(&self, id: &str, allow: bool) -> bool {
        self.0
            .pending
            .lock()
            .ok()
            .and_then(|mut p| p.remove(id))
            .is_some_and(|tx| tx.send(allow).is_ok())
    }
}

/// Audit reason when the human let a borderline call through once.
pub const HUMAN_ALLOWED_ONCE: &str = "human allowed a borderline score once";

/// A score at or above `threshold` but within `band` of it: worth asking the human about.
pub fn near_threshold(score: f32, threshold: f32, band: f32) -> bool {
    score >= threshold && score <= threshold + band
}

fn authorized(headers: &HeaderMap, state: &AppState) -> bool {
    let Some(value) = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
    else {
        return false;
    };
    let Some(token) = value.strip_prefix("Bearer ") else {
        return false;
    };
    use subtle::ConstantTimeEq;
    token
        .as_bytes()
        .ct_eq(state.config.jwt_secret.as_bytes())
        .into()
}

/// `GET /v1/alerts`: recent alerts, then live ones, as SSE `alert` events.
pub async fn stream(
    State(state): State<AppState>,
    headers: HeaderMap,
    peer: Option<axum::extract::ConnectInfo<SocketAddr>>,
) -> impl IntoResponse {
    if peer.is_some_and(|p| !p.0.ip().is_loopback()) {
        return (StatusCode::FORBIDDEN, "loopback clients only").into_response();
    }
    if !authorized(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, "unauthorized").into_response();
    }
    let rx = state.alert_hub.0.tx.subscribe();
    let history = state.alert_hub.recent();
    let initial =
        futures_util::stream::iter(history.into_iter().map(|a| {
            Ok::<Event, Infallible>(Event::default().event("alert").json_data(a).unwrap())
        }));
    let live = futures_util::stream::unfold(rx, |mut rx| async move {
        loop {
            match rx.recv().await {
                Ok(alert) => {
                    return Some((
                        Ok::<Event, Infallible>(
                            Event::default().event("alert").json_data(alert).unwrap(),
                        ),
                        rx,
                    ))
                }
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(broadcast::error::RecvError::Closed) => return None,
            }
        }
    });
    Sse::new(initial.chain(live))
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// `POST /v1/alerts/:id/decision` with `{"decision": "allow_once" | "block"}`.
pub async fn decide(
    State(state): State<AppState>,
    Path(id): Path<String>,
    headers: HeaderMap,
    peer: Option<axum::extract::ConnectInfo<SocketAddr>>,
    Json(body): Json<Decision>,
) -> impl IntoResponse {
    if peer.is_some_and(|p| !p.0.ip().is_loopback()) {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({"error":"loopback clients only"})),
        )
            .into_response();
    }
    if !authorized(&headers, &state) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({"error":"unauthorized"})),
        )
            .into_response();
    }
    let allow = match body.decision.as_str() {
        "allow_once" => true,
        "block" => false,
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error":"decision must be allow_once or block"})),
            )
                .into_response()
        }
    };
    if state.alert_hub.decide(&id, allow) {
        state.alert_hub.update_action(&id, &body.decision);
        (
            StatusCode::OK,
            Json(serde_json::json!({"decision":body.decision})),
        )
            .into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error":"approval expired or already decided"})),
        )
            .into_response()
    }
}

#[allow(clippy::too_many_arguments)]
pub fn new_alert(
    agent: &str,
    session_id: Option<String>,
    route: &str,
    target: &str,
    rule: &str,
    reason: &str,
    injection: Option<f32>,
    intent: Option<f32>,
    approval_required: bool,
) -> Alert {
    Alert {
        id: uuid_v4(),
        time: chrono::Utc::now().to_rfc3339(),
        agent: agent.to_string(),
        session_id,
        route: route.to_string(),
        target: target.to_string(),
        rule: rule.to_string(),
        reason: reason.to_string(),
        injection_confidence: injection,
        malicious_intent: intent,
        approval_required,
        action: if approval_required {
            "pending"
        } else {
            "blocked"
        }
        .into(),
    }
}

fn uuid_v4() -> String {
    static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    format!(
        "{}-{}-{}",
        chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default(),
        std::process::id(),
        NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applescript_strings_cannot_break_out() {
        assert_eq!(applescript_string(r#"a "b" \c"#), r#""a \"b\" \\c""#);
        assert_eq!(
            applescript_string("x\" & do shell script \"rm"),
            r#""x\" & do shell script \"rm""#
        );
        assert_eq!(applescript_string("line1\nline2"), "\"line1 line2\"");
    }

    fn pending_alert() -> Alert {
        new_alert(
            "agent",
            None,
            "POST /v1/messages",
            "Read notes.md",
            "model_risk_high",
            "r",
            Some(0.75),
            None,
            true,
        )
    }

    #[tokio::test]
    async fn approval_waits_for_the_human_and_records_the_outcome() {
        let hub = AlertHub::default();
        let mut rx = hub.0.tx.subscribe(); // a notifier is listening
        let waiting = {
            let hub = hub.clone();
            tokio::spawn(async move {
                hub.request_approval(pending_alert(), Duration::from_secs(5))
                    .await
            })
        };
        let pending = rx.recv().await.unwrap();
        assert_eq!(pending.action, "pending");
        assert!(hub.decide(&pending.id, true));
        assert!(waiting.await.unwrap(), "allow once");
        assert_eq!(rx.recv().await.unwrap().action, "allow_once");
        assert!(!hub.decide(&pending.id, true), "decided only once");
    }

    #[tokio::test]
    async fn no_listener_or_timeout_blocks() {
        let hub = AlertHub::default();
        let started = Instant::now();
        assert!(
            !hub.request_approval(pending_alert(), Duration::from_secs(30))
                .await
        );
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "no notifier: no wait"
        );
        assert_eq!(hub.recent().last().unwrap().action, "blocked");

        let _rx = hub.0.tx.subscribe();
        assert!(
            !hub.request_approval(pending_alert(), Duration::from_millis(50))
                .await,
            "timeout"
        );
        assert_eq!(hub.recent().last().unwrap().action, "blocked");
    }

    #[test]
    fn near_threshold_is_the_band_just_above_it() {
        assert!(near_threshold(0.70, 0.70, 0.10));
        assert!(near_threshold(0.80, 0.70, 0.10));
        assert!(!near_threshold(0.69, 0.70, 0.10));
        assert!(!near_threshold(0.81, 0.70, 0.10));
        assert!(
            near_threshold(0.99, 0.30, 1.0),
            "band 1.0 asks about every block"
        );
    }
}
