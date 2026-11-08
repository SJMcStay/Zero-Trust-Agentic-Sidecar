//! Debug output (`src/debug.rs`). Off by default; switched on by environment variables.
//!
//! - `ZT_DEBUG=1`: one line per step of every request. Route A steps are `A1`..`A6`, Route B
//!   steps `B1`..`B7`, Kev calls `KEV`: what came in, what was sent to Kev, the scores, the
//!   decision and why, where it was forwarded, and how long each step took.
//! - `ZT_DEBUG=2`: also previews of prompt / tool text and payload values. These can contain
//!   secrets: local debugging only.
//! - `ZT_DEBUG=3`: previews untruncated (the exact text Kev scores), for tuning skip patterns.
//! - `RUST_LOG=...` overrides both (standard `tracing` filter syntax).
//!
//! The same `ZT_DEBUG` switches on step output in `scripts/*.sh` (`set -x`),
//! `scripts/fetch_model.py` and `backend/app.py`.

use std::sync::OnceLock;
use tracing_subscriber::EnvFilter;

/// Characters of text shown by `ZT_DEBUG=2` previews.
pub const PREVIEW_CHARS: usize = 200;

/// Preview length: [`PREVIEW_CHARS`], or no limit with `ZT_DEBUG=3`.
pub fn preview_chars() -> usize {
    static LIMIT: OnceLock<usize> = OnceLock::new();
    *LIMIT.get_or_init(|| match std::env::var("ZT_DEBUG").as_deref() {
        Ok("3") | Ok("full") => usize::MAX,
        _ => PREVIEW_CHARS,
    })
}

/// Log filter: `RUST_LOG` if set, else the level `ZT_DEBUG` asks for, else info.
pub fn log_filter() -> EnvFilter {
    if let Ok(filter) = EnvFilter::try_from_default_env() {
        return filter;
    }
    let level = match std::env::var("ZT_DEBUG").as_deref() {
        Ok("2") | Ok("3") | Ok("trace") | Ok("full") => "trace",
        Ok("1") | Ok("true") | Ok("debug") => "debug",
        _ => "info",
    };
    EnvFilter::new(format!("zero_trust_agentic_sidecar={level},axum=info"))
}

/// First `max` characters of `text` on one line (newlines shown as `⏎`), `…` if cut.
pub fn preview(text: &str, max: usize) -> String {
    let mut out: String = text
        .chars()
        .take(max)
        .map(|c| if c == '\n' || c == '\r' { '⏎' } else { c })
        .collect();
    if text.chars().nth(max).is_some() {
        out.push('…');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_is_one_line_and_char_safe() {
        assert_eq!(preview("a\nb", 10), "a⏎b");
        assert_eq!(preview("héllo wörld", 5), "héllo…");
        assert_eq!(preview("🚀🚀🚀", 2), "🚀🚀…");
        assert_eq!(preview("", 5), "");
    }
}
