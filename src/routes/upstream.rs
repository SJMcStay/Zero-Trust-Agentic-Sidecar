//! Transparent forwarding to the LLM API (`src/routes/upstream.rs`).
//!
//! Claude Code's LLM-gateway guide: forward `anthropic-*` headers unchanged, never rewrite the
//! request body, return the upstream's headers (`x-should-retry`, rate-limit headers, ...), and
//! serve `/v1/messages/count_tokens`, `/v1/models` and `HEAD /api/hello` alongside
//! `/v1/messages`. Only those paths are passed through: with a sidecar-held API key, a catch-all
//! would hand every Anthropic endpoint to any local process.
//!
//! Auth: if the sidecar has `ANTHROPIC_API_KEY`, it replaces the client's credentials. If not,
//! the client's own `authorization` / `x-api-key` pass through, so a claude.ai login works.

use axum::{
    body::{Body, Bytes},
    extract::State,
    http::{header, HeaderMap, HeaderName, HeaderValue, Method, StatusCode, Uri},
    response::{IntoResponse, Response},
};

use crate::routes::{messages::AnthropicErrorEnvelope, AppState};

/// `anthropic-version` sent when the client sets none.
pub const DEFAULT_ANTHROPIC_VERSION: &str = "2023-06-01";

/// Hop-by-hop and transport headers never copied between the two connections.
const HOP_BY_HOP: [&str; 10] = [
    "connection",
    "keep-alive",
    "proxy-connection",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
];

/// Headers for the upstream request: everything the client sent except transport headers,
/// with the sidecar's key when it has one.
pub fn upstream_request_headers(incoming: &HeaderMap, sidecar_key: Option<&str>) -> HeaderMap {
    let mut out = HeaderMap::new();
    for (name, value) in incoming {
        let n = name.as_str();
        // content-length is recomputed; accept-encoding is dropped so the upstream answers
        // uncompressed and the bytes we relay match the headers we relay.
        if HOP_BY_HOP.contains(&n) || n == "content-length" || n == "accept-encoding" {
            continue;
        }
        if sidecar_key.is_some() && (n == "authorization" || n == "x-api-key") {
            continue;
        }
        out.append(name.clone(), value.clone());
    }
    if let Some(key) = sidecar_key {
        if let Ok(value) = HeaderValue::from_str(key) {
            out.insert(HeaderName::from_static("x-api-key"), value);
        }
    }
    if !out.contains_key("anthropic-version") {
        out.insert(
            HeaderName::from_static("anthropic-version"),
            HeaderValue::from_static(DEFAULT_ANTHROPIC_VERSION),
        );
    }
    out
}

/// `https://api.anthropic.com` from `https://api.anthropic.com/v1/messages`.
pub fn upstream_base(upstream_messages_url: &str) -> String {
    let trimmed = upstream_messages_url.trim_end_matches('/');
    if let Some(base) = trimmed.strip_suffix("/v1/messages") {
        return base.to_string();
    }
    match reqwest::Url::parse(trimmed) {
        Ok(url) => url.origin().ascii_serialization(),
        Err(_) => trimmed.to_string(),
    }
}

/// `path?query` of the incoming request, appended to `base`.
pub fn with_query(base: &str, uri: &Uri) -> String {
    match uri.query() {
        Some(q) if !q.is_empty() => format!("{base}?{q}"),
        _ => base.to_string(),
    }
}

/// Sends the request upstream and relays status, headers and body (streamed).
/// `stream_requested`: the client asked for SSE (sets SSE headers if the upstream omits them).
pub async fn forward(
    state: &AppState,
    method: Method,
    url: &str,
    incoming: &HeaderMap,
    body: Bytes,
    stream_requested: bool,
) -> Response {
    let headers = upstream_request_headers(incoming, state.config.anthropic_api_key.as_deref());
    let started = std::time::Instant::now();
    let upstream = match state
        .http_client
        .request(method.clone(), url)
        .headers(headers)
        .body(body)
        .send()
        .await
    {
        Ok(resp) => resp,
        Err(err) => {
            tracing::debug!(error = %err, %url, "UP upstream unreachable -> 502");
            let body = AnthropicErrorEnvelope::api_error(format!(
                "Upstream Anthropic connection failure: {err}"
            ));
            return (
                StatusCode::BAD_GATEWAY,
                [(header::CONTENT_TYPE, "application/json")],
                serde_json::to_string(&body).unwrap_or_default(),
            )
                .into_response();
        }
    };

    let status =
        StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut response_headers = HeaderMap::new();
    for (name, value) in upstream.headers() {
        let n = name.as_str();
        if HOP_BY_HOP.contains(&n) || n == "content-length" {
            continue;
        }
        if let (Ok(name), Ok(value)) = (
            HeaderName::from_bytes(name.as_str().as_bytes()),
            HeaderValue::from_bytes(value.as_bytes()),
        ) {
            response_headers.append(name, value);
        }
    }
    let is_sse = response_headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.starts_with("text/event-stream"))
        || (stream_requested
            && status.is_success()
            && !response_headers.contains_key(header::CONTENT_TYPE));
    if !response_headers.contains_key(header::CONTENT_TYPE) {
        let ct = if is_sse {
            "text/event-stream"
        } else {
            "application/json"
        };
        response_headers.insert(header::CONTENT_TYPE, HeaderValue::from_static(ct));
    }
    if is_sse {
        response_headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
        response_headers.insert(header::CONNECTION, HeaderValue::from_static("keep-alive"));
    }
    tracing::debug!(
        status = status.as_u16(),
        %method,
        %url,
        sse = is_sse,
        upstream_ms = started.elapsed().as_millis() as u64,
        "UP upstream answered; relaying"
    );

    let mut response = Response::new(Body::from_stream(upstream.bytes_stream()));
    *response.status_mut() = status;
    *response.headers_mut() = response_headers;
    response
}

/// `POST /v1/messages/count_tokens`, `GET /v1/models[/:id]`, `HEAD|GET /api/hello`: relayed
/// unchanged. Nothing to score (no model turn is produced).
pub async fn passthrough_handler(
    State(state): State<AppState>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let base = upstream_base(&state.config.upstream_llm_url);
    let url = with_query(&format!("{base}{}", uri.path()), &uri);
    tracing::debug!(%method, path = uri.path(), "PASS relayed without scoring");
    forward(&state, method, &url, &headers, body, false).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.append(
                HeaderName::from_bytes(k.as_bytes()).unwrap(),
                HeaderValue::from_str(v).unwrap(),
            );
        }
        h
    }

    #[test]
    fn client_credentials_pass_through_without_a_sidecar_key() {
        let out = upstream_request_headers(
            &headers(&[
                ("authorization", "Bearer oauth-token"),
                ("anthropic-beta", "oauth-2025-04-20,fine-grained"),
                ("anthropic-version", "2023-06-01"),
                ("x-claude-code-session-id", "s1"),
                ("host", "127.0.0.1:8080"),
                ("content-length", "10"),
                ("accept-encoding", "gzip"),
            ]),
            None,
        );
        assert_eq!(out["authorization"], "Bearer oauth-token");
        assert_eq!(out["anthropic-beta"], "oauth-2025-04-20,fine-grained");
        assert_eq!(out["x-claude-code-session-id"], "s1");
        for dropped in ["host", "content-length", "accept-encoding"] {
            assert!(
                !out.contains_key(dropped),
                "{dropped} must not be forwarded"
            );
        }
    }

    #[test]
    fn sidecar_key_replaces_client_credentials_and_version_defaults() {
        let out = upstream_request_headers(
            &headers(&[
                ("authorization", "Bearer client"),
                ("x-api-key", "client-key"),
            ]),
            Some("sk-sidecar"),
        );
        assert!(!out.contains_key("authorization"));
        assert_eq!(out["x-api-key"], "sk-sidecar");
        assert_eq!(out.get_all("x-api-key").iter().count(), 1);
        assert_eq!(out["anthropic-version"], DEFAULT_ANTHROPIC_VERSION);
    }

    #[test]
    fn base_and_query_are_derived_from_the_messages_url() {
        assert_eq!(
            upstream_base("https://api.anthropic.com/v1/messages"),
            "https://api.anthropic.com"
        );
        assert_eq!(
            upstream_base("http://127.0.0.1:9100/v1/messages/"),
            "http://127.0.0.1:9100"
        );
        assert_eq!(
            upstream_base("http://127.0.0.1:9100/other/path"),
            "http://127.0.0.1:9100"
        );
        let uri: Uri = "/v1/messages?beta=true".parse().unwrap();
        assert_eq!(
            with_query("https://x/v1/messages", &uri),
            "https://x/v1/messages?beta=true"
        );
        let plain: Uri = "/v1/messages".parse().unwrap();
        assert_eq!(
            with_query("https://x/v1/messages", &plain),
            "https://x/v1/messages"
        );
    }
}
