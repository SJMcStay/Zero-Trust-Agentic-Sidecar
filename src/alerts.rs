//! Desktop alerts (`src/alerts.rs`).
//!
//! A native notification when the sidecar blocks or flags something, so the human sees it even
//! when the agent only shows an error. Enabled by `[alerts] desktop = true` in the policy.
//! macOS: `osascript`; Linux: `notify-send`. Fire-and-forget and rate-limited.

use std::process::Stdio;
use std::sync::Mutex;
use std::time::{Duration, Instant};

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
}
