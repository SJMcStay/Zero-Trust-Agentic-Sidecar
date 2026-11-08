//! KevHttp against a fake server: checks the wire format without needing Kev running.
use decision_model::KevHttp;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

#[tokio::test]
async fn noul_round_trip() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut sock, _) = listener.accept().await.unwrap();
        let mut buf = vec![0u8; 8192];
        let mut req = Vec::new();
        // Read until the JSON body is complete (headers + body arrive in at most a few reads).
        while !req.ends_with(b"}}}") {
            let n = sock.read(&mut buf).await.unwrap();
            req.extend_from_slice(&buf[..n]);
        }
        let body = r#"{"model":"kev-latest","answers":{"q":{"type":"noul","noul":0.9}},"latency_ms":12.5}"#;
        let resp = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
            body.len()
        );
        sock.write_all(resp.as_bytes()).await.unwrap();
        String::from_utf8(req).unwrap()
    });

    let p = KevHttp::new(&format!("http://{addr}"))
        .noul("some text", "Injection?")
        .await
        .unwrap();
    assert!((p - 0.9).abs() < 1e-6);

    let req = server.await.unwrap();
    assert!(req.starts_with("POST /v1/systemone "));
    assert!(req.contains(r#""questions":{"q":{"type":"noul","instructions":"Injection?"}}"#));
}

#[tokio::test]
async fn clear_error_when_server_down() {
    let err = KevHttp::new("http://127.0.0.1:9")
        .noul("x", "y")
        .await
        .unwrap_err();
    assert!(format!("{err:#}").contains("kev_server.sh"), "{err:#}");
}
