//! Zero-Trust Agentic Sidecar Server Entrypoint (`src/main.rs`)

use std::sync::Arc;
use zero_trust_agentic_sidecar::{
    create_router, kev_dirs_from_manifest, resolve_kev_artifacts, select_device, AppState,
    AuditLogger, Config, DecisionEngine, KevCandleEngine, KevEngineKind, KevHttpEngine, MlError,
    PolicyEngine, SyntheticKevEngine,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize tracing subscriber
    // ZT_DEBUG=1 / 2 turns on per-step debug output (see src/debug.rs); RUST_LOG overrides.
    tracing_subscriber::fmt()
        .with_env_filter(zero_trust_agentic_sidecar::debug::log_filter())
        .init();

    tracing::info!("Initializing Zero-Trust Agentic Sidecar...");

    // 2. Load runtime configuration with fail-closed secret validation
    let config = match Config::from_env() {
        Ok(cfg) => Arc::new(cfg),
        Err(err) => {
            tracing::error!("FATAL: Configuration error: {err}");
            std::process::exit(1);
        }
    };
    tracing::info!(
        "Configuration loaded. Target bind address: {}",
        config.sidecar_addr
    );
    tracing::debug!(
        engine = ?config.kev_engine,
        kev_url = %config.kev_url,
        kev_model = %config.kev_model,
        upstream = %config.upstream_llm_url,
        backend = %config.flask_backend_url,
        policy = ?config.policy_path,
        anthropic_api_key = if config.anthropic_api_key.is_some() { "set" } else { "not set" },
        "startup config (secrets never printed)"
    );

    // 3. Initialize structured JSONL audit logger with secret redaction
    let audit_logger = Arc::new(AuditLogger::from_config(&config));
    tracing::info!("Structured audit logger initialized");

    // 4. Load deterministic policy configuration from TOML
    let policy_engine = match PolicyEngine::from_file(&config.policy_path) {
        Ok(engine) => Arc::new(engine),
        Err(err) => {
            tracing::error!(
                "FATAL: Failed to load policy configuration from '{:?}': {err}",
                config.policy_path
            );
            std::process::exit(1);
        }
    };
    tracing::info!(
        "Deterministic policy engine loaded from {:?}",
        config.policy_path
    );
    let d = policy_engine.defaults();
    tracing::debug!(
        max_injection = d.max_injection_confidence,
        max_intent = d.max_malicious_intent,
        model_timeout_ms = d.model_timeout_ms,
        fail_closed = d.fail_closed,
        "policy thresholds"
    );

    // 5. Initialize the DecisionEngine chosen by KEV_ENGINE. Fail closed: if it cannot start,
    //    the sidecar exits instead of serving with a weaker engine.
    let decision_engine: Arc<dyn DecisionEngine> =
        match init_decision_engine(&config, &policy_engine).await {
            Ok(engine) => engine,
            Err(err) => {
                tracing::error!("FATAL: decision engine unavailable: {err}");
                std::process::exit(1);
            }
        };
    tracing::info!("Decision engine ready: {}", decision_engine.model_id());

    // 6. Initialize connection-pooled HTTP client
    let http_client = reqwest::Client::builder()
        .pool_max_idle_per_host(32)
        .pool_idle_timeout(std::time::Duration::from_secs(90))
        .tcp_keepalive(std::time::Duration::from_secs(60))
        .build()?;

    // 7. Assemble shared AppState & Axum Router
    // Edits to the policy file apply without a restart (checked at most once a second;
    // a broken edit keeps the last good policy).
    let live_policy = Arc::new(zero_trust_agentic_sidecar::LivePolicy::new(
        config.policy_path.clone(),
        Arc::clone(&policy_engine),
    ));
    let state = AppState::new(
        config.clone(),
        policy_engine,
        decision_engine,
        audit_logger,
        http_client,
    )
    .with_live_policy(live_policy);
    let app = create_router(state);

    // 8. Bind TCP listener
    let listener = tokio::net::TcpListener::bind(config.sidecar_addr).await?;
    tracing::info!(
        "Zero-Trust Agentic Sidecar listening on http://{}",
        config.sidecar_addr
    );

    // 9. Start server with graceful shutdown handling
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    tracing::info!("Zero-Trust Agentic Sidecar shut down cleanly");
    Ok(())
}

/// Builds the engine selected by `KEV_ENGINE`. Every failure is returned, never papered over.
async fn init_decision_engine(
    config: &Config,
    policy_engine: &PolicyEngine,
) -> Result<Arc<dyn DecisionEngine>, MlError> {
    match config.kev_engine {
        KevEngineKind::Http => {
            let engine = KevHttpEngine::new(&config.kev_url, config.kev_model.clone())?;
            engine.check_ready().await?;
            tracing::info!("Using Kev reference server at {}", config.kev_url);
            Ok(Arc::new(engine))
        }
        KevEngineKind::Candle => {
            let (kev_dir, base_dir) =
                kev_dirs_from_manifest(std::path::Path::new("models.json"), &config.kev_model)?;
            let paths = resolve_kev_artifacts(&kev_dir, &base_dir)?;
            let device = select_device();
            tracing::info!("Loading KevCandleEngine from {kev_dir:?} on {device:?}");
            let timeout =
                std::time::Duration::from_millis(policy_engine.defaults().model_timeout_ms);
            Ok(Arc::new(KevCandleEngine::load(&paths, device, 4, timeout)?))
        }
        KevEngineKind::Synthetic => {
            tracing::warn!(
                "KEV_ENGINE=synthetic: deterministic toy graph, not Kev. Tests and demos only."
            );
            // CPU: the toy graph is tiny, and Metal kernel warm-up can exceed its 500 ms budget.
            Ok(Arc::new(SyntheticKevEngine::for_cpu()?))
        }
    }
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install CTRL+C signal handler");
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(e) => {
                tracing::error!("Failed to install SIGTERM signal handler: {e}");
            }
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            tracing::info!("Received Ctrl+C, draining active connections...");
        },
        _ = terminate => {
            tracing::info!("Received SIGTERM, draining active connections...");
        },
    }
}
