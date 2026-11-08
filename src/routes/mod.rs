//! Axum Router & Application State Setup (`src/routes/mod.rs`)
//!
//! Exposes shared `AppState`, router construction, and core middleware.

pub mod execute;
pub mod messages;
pub mod upstream;

use std::sync::{Arc, Mutex};

use axum::{
    body::Body,
    extract::Request,
    http::{header, HeaderValue, Method, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde_json::json;

use crate::{
    alerts::AlertHub,
    audit::AuditLogger,
    config::Config,
    ml::DecisionEngine,
    policy::{LivePolicy, PolicyEngine},
    turn::BlockedMemory,
};

pub use execute::execute_handler;
pub use messages::messages_handler;

/// Shared thread-safe application state injected into all route handlers.
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub policy_engine: Arc<PolicyEngine>,
    pub decision_engine: Arc<dyn DecisionEngine>,
    pub audit_logger: Arc<AuditLogger>,
    pub http_client: reqwest::Client,
    /// Re-reads the policy file when it changes (set by `main.rs`; `None` in tests).
    pub live_policy: Option<Arc<LivePolicy>>,
    /// Content blocked earlier in a conversation (Route A `remember_blocked`).
    pub blocked: Arc<Mutex<BlockedMemory>>,
    /// Recent alerts and pending human approvals (`GET /v1/alerts`).
    pub alert_hub: AlertHub,
}

impl AppState {
    /// Creates a new AppState instance.
    pub fn new(
        config: Arc<Config>,
        policy_engine: Arc<PolicyEngine>,
        decision_engine: Arc<dyn DecisionEngine>,
        audit_logger: Arc<AuditLogger>,
        http_client: reqwest::Client,
    ) -> Self {
        Self {
            config,
            policy_engine,
            decision_engine,
            audit_logger,
            http_client,
            live_policy: None,
            blocked: Arc::new(Mutex::new(BlockedMemory::default())),
            alert_hub: AlertHub::default(),
        }
    }

    /// Re-read the policy file when it changes on disk.
    pub fn with_live_policy(mut self, live: Arc<LivePolicy>) -> Self {
        self.live_policy = Some(live);
        self
    }

    /// The policy for this request: the live (reloaded) one when set, else the startup one.
    pub fn current_policy(&self) -> Arc<PolicyEngine> {
        match &self.live_policy {
            Some(live) => live.current(),
            None => Arc::clone(&self.policy_engine),
        }
    }

    /// Convenience constructor for testing with SyntheticKevEngine and captured audit logs.
    pub fn test_state(
        config: Config,
        policy_engine: PolicyEngine,
        decision_engine: Arc<dyn DecisionEngine>,
        audit_logger: AuditLogger,
    ) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self {
            config: Arc::new(config),
            policy_engine: Arc::new(policy_engine),
            decision_engine,
            audit_logger: Arc::new(audit_logger),
            http_client,
            live_policy: None,
            blocked: Arc::new(Mutex::new(BlockedMemory::default())),
            alert_hub: AlertHub::default(),
        }
    }

    /// Accessor for decision engine.
    pub fn engine(&self) -> &Arc<dyn DecisionEngine> {
        &self.decision_engine
    }

    /// Accessor for policy engine.
    pub fn policy(&self) -> &Arc<PolicyEngine> {
        &self.policy_engine
    }

    /// Accessor for audit logger.
    pub fn logger(&self) -> &Arc<AuditLogger> {
        &self.audit_logger
    }
}

/// Unauthenticated health check endpoint (`GET /healthz`).
pub async fn healthz_handler() -> impl IntoResponse {
    (StatusCode::OK, Json(json!({ "status": "ok" })))
}

/// Request timing and tracing middleware.
pub async fn trace_middleware(req: Request, next: Next) -> Response {
    let method = req.method().clone();
    let uri = req.uri().clone();
    let start = std::time::Instant::now();

    tracing::debug!(method = %method, uri = %uri, "incoming HTTP request");

    let response = next.run(req).await;

    let latency = start.elapsed();
    tracing::info!(
        method = %method,
        uri = %uri,
        status = response.status().as_u16(),
        latency_ms = latency.as_millis(),
        "completed HTTP request"
    );

    response
}

/// Lightweight CORS middleware handling preflight OPTIONS and adding permissive headers.
pub async fn cors_middleware(req: Request, next: Next) -> Response {
    if req.method() == Method::OPTIONS {
        return Response::builder()
            .status(StatusCode::NO_CONTENT)
            .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
            .header(
                header::ACCESS_CONTROL_ALLOW_METHODS,
                "GET, POST, OPTIONS, PUT, DELETE",
            )
            .header(
                header::ACCESS_CONTROL_ALLOW_HEADERS,
                "authorization, content-type, x-api-key, anthropic-version, x-sidecar-timestamp, x-sidecar-signature, anthropic-beta",
            )
            .header(header::ACCESS_CONTROL_MAX_AGE, "86400")
            .body(Body::empty())
            .unwrap();
    }

    let mut response = next.run(req).await;
    response.headers_mut().insert(
        header::ACCESS_CONTROL_ALLOW_ORIGIN,
        HeaderValue::from_static("*"),
    );
    response.headers_mut().insert(
        header::ACCESS_CONTROL_ALLOW_HEADERS,
        HeaderValue::from_static("*"),
    );
    response
}

/// Constructs the primary Axum router configured with routes and middleware.
pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(healthz_handler))
        .route("/v1/messages", post(messages::messages_handler))
        // Relayed unscored for Claude Code (LLM-gateway guide); see routes/upstream.rs.
        .route(
            "/v1/messages/count_tokens",
            post(upstream::passthrough_handler),
        )
        .route("/v1/models", get(upstream::passthrough_handler))
        .route("/v1/models/:id", get(upstream::passthrough_handler))
        .route(
            "/api/hello",
            get(upstream::passthrough_handler).head(upstream::passthrough_handler),
        )
        .route("/api/execute", post(execute::execute_handler))
        .route("/v1/alerts", get(crate::alerts::stream))
        .route("/v1/alerts/:id/decision", post(crate::alerts::decide))
        .layer(axum::middleware::from_fn(trace_middleware))
        .layer(axum::middleware::from_fn(cors_middleware))
        .with_state(state)
}
