use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::io::Cursor;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::Mutex;
use rustls::crypto::ring::default_provider;
use tokio_rustls::rustls::pki_types::ServerName;
use tokio_rustls::rustls::{ClientConfig, RootCertStore};
use tokio_rustls::TlsConnector;

const MAGIC: &[u8; 8] = b"AKAITUNL";
const V1: u8 = 1;

const DATA: u8 = 0x01;
const NEW_CONN: u8 = 0x02;
const CLOSE: u8 = 0x03;
const PING: u8 = 0x04;
const PONG: u8 = 0x05;

struct Frame {
    msg_type: u8,
    payload: Vec<u8>,
}

fn write_frame(msg_type: u8, payload: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(5 + payload.len());
    buf.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    buf.push(msg_type);
    buf.extend_from_slice(payload);
    buf
}

async fn read_frame<R: AsyncReadExt + Unpin>(reader: &mut R) -> Result<Frame> {
    let mut hdr = [0u8; 5];
    reader.read_exact(&mut hdr).await?;
    let payload_len = u32::from_be_bytes(hdr[..4].try_into()?) as usize;
    let msg_type = hdr[4];
    let mut payload = vec![0u8; payload_len];
    if payload_len > 0 {
        reader.read_exact(&mut payload).await?;
    }
    Ok(Frame { msg_type, payload })
}

struct ConnState {
    write_tx: tokio::sync::mpsc::Sender<Vec<u8>>,
}

type TlsWriteHalf = tokio::io::WriteHalf<tokio_rustls::client::TlsStream<TcpStream>>;

pub struct TunnelClient {
    server_host: String,
    server_port: u16,
    worker_id: String,
    rpc_port: u16,
    ca_cert_pem: Vec<u8>,
    client_cert_pem: Vec<u8>,
    client_key_pem: Vec<u8>,
    conns: Arc<Mutex<HashMap<u32, ConnState>>>,
    connected: Arc<AtomicBool>,
}

impl TunnelClient {
    pub fn new(
        server_host: &str,
        server_port: u16,
        worker_id: &str,
        rpc_port: u16,
        ca_cert_pem: Vec<u8>,
        client_cert_pem: Vec<u8>,
        client_key_pem: Vec<u8>,
        connected: Arc<AtomicBool>,
    ) -> Self {
        Self {
            server_host: server_host.to_string(),
            server_port,
            worker_id: worker_id.to_string(),
            rpc_port,
            ca_cert_pem,
            client_cert_pem,
            client_key_pem,
            conns: Arc::new(Mutex::new(HashMap::new())),
            connected,
        }
    }

    pub async fn run(&self) -> Result<()> {
        let connector = self.build_tls_connector()
            .map_err(|e| anyhow::anyhow!("TLS config failed: {e}"))?;
        let conns = self.conns.clone();
        let connected = self.connected.clone();

        loop {
            connected.store(false, Ordering::Relaxed);
            match self.connect_and_serve(&connector).await {
                Ok(()) => {
                    eprintln!("tunnel disconnected, reconnecting in 5s...");
                }
                Err(e) => {
                    eprintln!("tunnel error: {e}, reconnecting in 5s...");
                }
            }
            conns.lock().await.clear();
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        }
    }

    fn build_tls_connector(&self) -> Result<TlsConnector> {
        let _ = default_provider().install_default();
        let mut root_store = RootCertStore::empty();
        let ca_certs = rustls_pemfile::certs(&mut Cursor::new(self.ca_cert_pem.clone()))
            .collect::<Result<Vec<_>, _>>()
            .context("failed to parse CA cert")?;
        if ca_certs.is_empty() {
            bail!("no CA certificates found in PEM");
        }
        for cert in ca_certs {
            root_store.add(cert).context("invalid CA cert")?;
        }

        let client_certs = rustls_pemfile::certs(&mut Cursor::new(self.client_cert_pem.clone()))
            .collect::<Result<Vec<_>, _>>()
            .context("failed to parse client cert")?;
        if client_certs.is_empty() {
            bail!("no client certificates found in PEM");
        }

        let client_key = rustls_pemfile::private_key(&mut Cursor::new(self.client_key_pem.clone()))
            .context("failed to parse client key")?
            .context("no client key in PEM")?;

        tracing::info!(
            "tls: {} CA certs, {} client certs",
            root_store.len(),
            client_certs.len(),
        );

        let config = ClientConfig::builder()
            .with_root_certificates(root_store)
            .with_client_auth_cert(client_certs, client_key)
            .map_err(|e| anyhow::anyhow!("with_client_auth_cert failed: {e}"))?;

        Ok(TlsConnector::from(Arc::new(config)))
    }

    async fn connect_and_serve(&self, connector: &TlsConnector) -> Result<()> {
        let server_host = self.server_host.clone();
        let worker_id = self.worker_id.clone();
        let rpc_port = self.rpc_port;
        let conns = self.conns.clone();
        let connected = self.connected.clone();

        eprintln!("tunnel: connecting to {}:{}...", server_host, self.server_port);

        let domain = ServerName::try_from(server_host.clone())
            .map_err(|e| anyhow::anyhow!("invalid server name: {e}"))?;

        let tcp = TcpStream::connect((&*server_host, self.server_port)).await
            .context("TCP connect failed")?;
        eprintln!("tunnel: TCP connected to {}:{}", server_host, self.server_port);

        let tls = connector.connect(domain, tcp).await
            .context("TLS handshake failed")?;
        eprintln!("tunnel: TLS handshake complete");

        let (mut reader, mut writer) = tokio::io::split(tls);

        let wid_bytes = worker_id.as_bytes();
        let mut handshake = Vec::with_capacity(8 + 1 + 2 + wid_bytes.len() + 2);
        handshake.extend_from_slice(MAGIC);
        handshake.push(V1);
        handshake.extend_from_slice(&(wid_bytes.len() as u16).to_be_bytes());
        handshake.extend_from_slice(wid_bytes);
        handshake.extend_from_slice(&rpc_port.to_be_bytes());

        writer.write_all(&handshake).await?;
        writer.flush().await?;

        connected.store(true, Ordering::Relaxed);
        println!("tunnel connected to {}:{}", server_host, self.server_port);

        let writer = Arc::new(Mutex::new(writer));

        let ping_writer = writer.clone();
        let ping_handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));
            loop {
                interval.tick().await;
                let frame = write_frame(PING, &[]);
                let mut w = ping_writer.lock().await;
                if w.write_all(&frame).await.is_err() {
                    break;
                }
                let _ = w.flush().await;
            }
        });

        let result = async {
            loop {
                let frame = read_frame(&mut reader).await?;

                match frame.msg_type {
                    NEW_CONN if frame.payload.len() >= 4 => {
                        let conn_id = u32::from_be_bytes(frame.payload[..4].try_into()?);
                        let w = writer.clone();
                        let c = conns.clone();

                        tokio::spawn(async move {
                            if let Err(e) = serve_conn(conn_id, rpc_port, w, c).await {
                                tracing::debug!("conn {} ended: {e}", conn_id);
                            }
                        });
                    }

                    CLOSE if frame.payload.len() >= 4 => {
                        let conn_id = u32::from_be_bytes(frame.payload[..4].try_into()?);
                        let mut map = conns.lock().await;
                        if let Some(conn) = map.remove(&conn_id) {
                            let _ = conn.write_tx.send(Vec::new()).await;
                        }
                    }

                    PONG => {}

                    _ => {
                        tracing::warn!("unknown frame type {} len={}", frame.msg_type, frame.payload.len());
                    }
                }
            }
        }.await;

        ping_handle.abort();
        result
    }
}

async fn serve_conn(
    conn_id: u32,
    rpc_port: u16,
    tunnel_writer: Arc<Mutex<TlsWriteHalf>>,
    conns: Arc<Mutex<HashMap<u32, ConnState>>>,
) -> Result<()> {
    let local = TcpStream::connect(format!("127.0.0.1:{}", rpc_port)).await
        .context("connect to local rpc-server failed")?;
    let (mut rpc_reader, mut rpc_writer) = local.into_split();

    let (write_tx, mut write_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(256);

    conns.lock().await.insert(conn_id, ConnState { write_tx: write_tx.clone() });

    let conn_id_bytes = conn_id.to_be_bytes();

    let tw1 = tunnel_writer.clone();
    let up = async move {
        let mut buf = [0u8; 65536];
        loop {
            let n = match rpc_reader.read(&mut buf).await {
                Ok(0) => break,
                Ok(n) => n,
                Err(_) => break,
            };
            let mut payload = conn_id_bytes.to_vec();
            payload.extend_from_slice(&buf[..n]);
            let f = write_frame(DATA, &payload);
            let mut w = tw1.lock().await;
            if w.write_all(&f).await.is_err() {
                break;
            }
            let _ = w.flush().await;
        }
        let f = write_frame(CLOSE, &conn_id_bytes);
        let mut w = tw1.lock().await;
        let _ = w.write_all(&f).await;
        let _ = w.flush().await;
    };

    let down = async move {
        while let Some(data) = write_rx.recv().await {
            if data.is_empty() {
                break;
            }
            if rpc_writer.write_all(&data).await.is_err() {
                break;
            }
            let _ = rpc_writer.flush().await;
        }
    };

    tokio::select! {
        _ = up => {},
        _ = down => {},
    }

    conns.lock().await.remove(&conn_id);
    Ok(())
}