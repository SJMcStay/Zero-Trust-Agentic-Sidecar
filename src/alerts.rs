//! In-memory alert stream and one-time human approval for borderline ML denials.
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
use std::{
    collections::{HashMap, VecDeque},
    convert::Infallible,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::sync::{broadcast, oneshot};

use crate::routes::AppState;

const HISTORY_LIMIT: usize = 128;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub time: String,
    pub agent: String,
    pub session_id: Option<String>,
    pub route: String,
    pub target: String,
    pub rule: String,
    pub reason: String,
    pub injection_confidence: Option<f32>,
    pub malicious_intent: Option<f32>,
    pub approval_required: bool,
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

    pub async fn request_approval(&self, alert: Alert, timeout: Duration) -> bool {
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
    let history: Vec<Alert> = state
        .alert_hub
        .0
        .history
        .lock()
        .map(|h| h.iter().cloned().collect())
        .unwrap_or_default();
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
