//! Zero-Trust Agentic Sidecar Server Entrypoint (`src/main.rs`)

use std::sync::Arc;
use tracing_subscriber::EnvFilter;
use zero_trust_agentic_sidecar::{
    create_router, download_kev_artifacts, select_device, AppState, AuditLogger, Config,
    DecisionEngine, KevCandleEngine, PolicyEngine, SyntheticKevEngine,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize tracing subscriber
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "zero_trust_agentic_sidecar=info,axum=info".into()),
        )
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

    // 5. Initialize DecisionEngine (KevCandleEngine or SyntheticKevEngine fallback)
    let use_synthetic = std::env::var("USE_SYNTHETIC_MODEL")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let decision_engine: Arc<dyn DecisionEngine> = if use_synthetic {
        tracing::info!("USE_SYNTHETIC_MODEL is set; initializing SyntheticKevEngine");
        Arc::new(SyntheticKevEngine::new()?)
    } else {
        match download_kev_artifacts(
            &config.model_cache_dir,
            &config.kev_repo_id,
            &config.kev_base_repo_id,
        ) {
            Ok(paths) => {
                let device = select_device();
                tracing::info!("Loading KevCandleEngine on device: {:?}", device);
                let timeout =
                    std::time::Duration::from_millis(policy_engine.defaults().model_timeout_ms);
                match KevCandleEngine::load(&paths, device, 4, timeout) {
                    Ok(engine) => {
                        tracing::info!(
                            "KevCandleEngine loaded successfully with folded LoRA weights"
                        );
                        Arc::new(engine)
                    }
                    Err(err) => {
                        tracing::warn!(
                            "Failed to load KevCandleEngine ({err}). Falling back to SyntheticKevEngine"
                        );
                        Arc::new(SyntheticKevEngine::new()?)
                    }
                }
            }
            Err(err) => {
                tracing::warn!(
                    "Model artifacts unavailable ({err}). Falling back to SyntheticKevEngine"
                );
                Arc::new(SyntheticKevEngine::new()?)
            }
        }
    };

    // 6. Initialize connection-pooled HTTP client
    let http_client = reqwest::Client::builder()
        .pool_max_idle_per_host(32)
        .pool_idle_timeout(std::time::Duration::from_secs(90))
        .tcp_keepalive(std::time::Duration::from_secs(60))
        .build()?;

    // 7. Assemble shared AppState & Axum Router
    let state = AppState::new(
        config.clone(),
        policy_engine,
        decision_engine,
        audit_logger,
        http_client,
    );
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
