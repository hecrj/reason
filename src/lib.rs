mod error;

pub use error::Error;

use serde::Deserialize;
use serde_json::json;
use sipper::{FutureExt, Sipper, Straw, StreamExt, sipper};
use tokio::io::{self, AsyncBufReadExt};
use tokio::process;
use tokio::task;
use tokio::time;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub use reqwest::IntoUrl;
pub use url::Url;

#[derive(Debug, Clone)]
pub struct Reason {
    name: String,
    server: Arc<Server>,
}

#[derive(Debug, Clone)]
pub enum Source {
    Local(PathBuf),
    Remote(Url),
}

impl Reason {
    const LLAMA_CPP_CONTAINER_CPU: &'static str = "ghcr.io/ggerganov/llama.cpp:server-b4600";
    const LLAMA_CPP_CONTAINER_CUDA: &'static str = "ghcr.io/ggerganov/llama.cpp:server-cuda-b4600";
    const LLAMA_CPP_CONTAINER_ROCM: &'static str = "ghcr.io/hecrj/icebreaker:server-rocm-b4600";

    pub async fn connect(host: impl IntoUrl, model: &str) -> Result<Self, Error> {
        let host = host.into_url()?;
        let client = reqwest::Client::new();

        loop {
            if let Ok(_) = client
                .get(format!(
                    "{host}/v1/models",
                    host = host.as_str().trim_end_matches('/')
                ))
                .timeout(Duration::from_secs(5))
                .send()
                .await?
                .error_for_status()
            {
                break;
            }

            time::sleep(Duration::from_secs(1)).await;
        }

        Ok(Self {
            name: model.to_owned(),
            server: Arc::new(Server::Remote(host)),
        })
    }

    pub fn boot(model: impl AsRef<Path>, backend: Backend) -> impl Straw<Self, BootEvent, Error> {
        #[derive(Clone)]
        struct Sender(sipper::Sender<BootEvent>);

        impl Sender {
            async fn log(&mut self, log: String) {
                let _ = self.0.send(BootEvent::Logged(log)).await;
            }

            async fn progress(&mut self, stage: &'static str, percent: u32) {
                let _ = self.0.send(BootEvent::Progressed { stage, percent }).await;
            }
        }

        sipper(async move |sender| {
            let model = model.as_ref().to_owned();
            let model_file = model.file_stem().unwrap_or_default();
            let name = model
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();

            let mut sender = Sender(sender);
            sender.progress("Detecting executor...", 0).await;

            let (server, stdout, stderr) = if let Ok(version) =
                process::Command::new("llama-server")
                    .arg("--version")
                    .output()
                    .await
            {
                sender
                    .log("Local llama-server binary found!".to_owned())
                    .await;

                let mut lines = version.stdout.lines();

                while let Some(line) = lines.next_line().await? {
                    sender.log(line).await;
                }

                sender.progress("Launching assistant...", 99).await;

                sender
                    .log(format!(
                        "Launching {model} with local llama-server...",
                        model = model_file.display()
                    ))
                    .await;

                let mut server = Server::launch_with_executable("llama-server", &model, backend)?;
                let stdout = server.stdout.take();
                let stderr = server.stderr.take();

                (
                    Server::Process {
                        _handle: server,
                        model,
                    },
                    stdout,
                    stderr,
                )
            } else if let Ok(_docker) = process::Command::new("docker")
                .arg("version")
                .output()
                .await
            {
                sender
                    .log(format!(
                        "Launching {model} with Docker...",
                        model = model_file.display()
                    ))
                    .await;

                sender.progress("Preparing container...", 0).await;

                let volume = model.parent().unwrap_or(Path::new("."));

                let command = match backend {
                    Backend::Cpu => {
                        format!(
                            "create --rm -p {port}:80 -v {volume}:/models \
                            {container} --model /models/{filename} \
                            --port 80 --host 0.0.0.0",
                            filename = model_file.display(),
                            container = Self::LLAMA_CPP_CONTAINER_CPU,
                            port = Server::PORT,
                            volume = volume.display(),
                        )
                    }
                    Backend::Cuda => {
                        format!(
                            "create --rm --gpus all -p {port}:80 -v {volume}:/models \
                            {container} --model /models/{filename} \
                            --port 80 --host 0.0.0.0 --gpu-layers 40",
                            filename = model_file.display(),
                            container = Self::LLAMA_CPP_CONTAINER_CUDA,
                            port = Server::PORT,
                            volume = volume.display(),
                        )
                    }
                    Backend::Rocm => {
                        format!(
                            "create --rm -p {port}:80 -v {volume}:/models \
                            --device=/dev/kfd --device=/dev/dri \
                            --security-opt seccomp=unconfined --group-add video \
                            {container} --model /models/{filename} \
                            --port 80 --host 0.0.0.0 --gpu-layers 40",
                            filename = model_file.display(),
                            container = Self::LLAMA_CPP_CONTAINER_ROCM,
                            port = Server::PORT,
                            volume = volume.display(),
                        )
                    }
                };

                let mut docker = process::Command::new("docker")
                    .args(Server::parse_args(&command))
                    .kill_on_drop(true)
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .spawn()?;

                let notify_progress = {
                    let mut sender = sender.clone();

                    let output = io::BufReader::new(docker.stderr.take().expect("piped stderr"));

                    async move {
                        let mut lines = output.lines();

                        while let Ok(Some(log)) = lines.next_line().await {
                            sender.log(log).await;
                        }
                    }
                };

                let _handle = task::spawn(notify_progress);

                let container = {
                    let output = io::BufReader::new(docker.stdout.take().expect("piped stdout"));

                    let mut lines = output.lines();

                    lines
                        .next_line()
                        .await?
                        .ok_or_else(|| Error::DockerFailed("no container id returned by docker"))?
                };

                if !docker.wait().await?.success() {
                    return Err(Error::DockerFailed("failed to create container"));
                }

                sender.progress("Launching assistant...", 99).await;

                let server = Server::Container {
                    id: container.clone(),
                    model,
                };

                let _start = process::Command::new("docker")
                    .args(["start", &container])
                    .output()
                    .await?;

                let mut logs = process::Command::new("docker")
                    .args(["logs", "-f", &container])
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .spawn()?;

                (server, logs.stdout.take(), logs.stderr.take())
            } else {
                return Err(Error::NoExecutorAvailable);
            };

            let log_output = {
                let mut sender = sender.clone();

                let mut lines = {
                    use futures_util::stream;
                    use tokio_stream::wrappers::LinesStream;

                    let stdout = stdout.expect("piped stdout");
                    let stderr = stderr.expect("piped stderr");

                    let stdout = io::BufReader::new(stdout);
                    let stderr = io::BufReader::new(stderr);

                    stream::select(
                        LinesStream::new(stdout.lines()),
                        LinesStream::new(stderr.lines()),
                    )
                };

                async move {
                    while let Some(line) = lines.next().await {
                        if let Ok(log) = line {
                            sender.log(log).await;
                        }
                    }

                    false
                }
                .boxed()
            };

            let check_health = {
                let address = server.host();

                async move {
                    loop {
                        time::sleep(Duration::from_secs(1)).await;

                        if let Ok(response) = reqwest::get(format!("{address}/health")).await {
                            if response.error_for_status().is_ok() {
                                return true;
                            }
                        }
                    }
                }
                .boxed()
            };

            let log_handle = task::spawn(log_output);

            if check_health.await {
                log_handle.abort();

                return Ok(Self {
                    name,
                    server: Arc::new(server),
                });
            }

            Err(Error::ExecutorFailed("llama-server exited unexpectedly"))
        })
    }

    pub fn reply<'a>(
        &'a self,
        prompt: &'a str,
        messages: &'a [Message],
        append: &'a [Message],
    ) -> impl Straw<Reply, (Reply, Token), Error> + 'a {
        sipper(move |mut progress| async move {
            let mut reasoning = None;
            let mut reasoning_started_at: Option<Instant> = None;
            let mut content = String::new();
            let mut reasoning_content = String::new();

            let mut completion = self.complete(prompt, messages, append).pin();

            while let Some(token) = completion.sip().await {
                match &token {
                    Token::Reasoning(token) => {
                        reasoning = {
                            let mut reasoning = reasoning.take().unwrap_or_else(|| Reasoning {
                                content: String::new(),
                                duration: Duration::ZERO,
                            });

                            if let Some(reasoning_started_at) = reasoning_started_at {
                                reasoning.duration = reasoning_started_at.elapsed();
                            } else {
                                reasoning_started_at = Some(Instant::now());
                            }

                            reasoning_content.push_str(token);
                            reasoning.content = reasoning_content.trim().to_owned();

                            Some(reasoning)
                        };
                    }
                    Token::Talking(token) => {
                        content.push_str(token);
                    }
                }

                progress
                    .send((
                        Reply {
                            reasoning: reasoning.clone(),
                            content: content.trim().to_owned(),
                            last_token: if let Token::Talking(token) = &token {
                                Some(token.clone())
                            } else {
                                None
                            },
                        },
                        token,
                    ))
                    .await;
            }

            Ok(Reply {
                reasoning: reasoning.clone(),
                content: content.trim().to_owned(),
                last_token: None,
            })
        })
    }

    pub fn complete<'a>(
        &'a self,
        system_prompt: &'a str,
        messages: &'a [Message],
        append: &'a [Message],
    ) -> impl Straw<(), Token, Error> + 'a {
        sipper(move |mut sender| async move {
            let client = reqwest::Client::new();

            let request = {
                let messages: Vec<_> = [("system", system_prompt)]
                    .into_iter()
                    .chain(messages.iter().chain(append).map(Message::to_tuple))
                    .map(|(role, content)| {
                        json!({
                            "role": role,
                            "content": content
                        })
                    })
                    .collect();

                client
                    .post(format!(
                        "{host}/v1/chat/completions",
                        host = self.server.host(),
                    ))
                    .json(&json!({
                        "model": self.name,
                        "messages": messages,
                        "stream": true,
                        "cache_prompt": true,
                    }))
            };

            let mut response = request.send().await?.error_for_status()?;
            let mut buffer = Vec::new();
            let mut is_reasoning = None;

            while let Some(chunk) = response.chunk().await? {
                buffer.extend(chunk);

                let mut lines = buffer
                    .split(|byte| *byte == 0x0A)
                    .filter(|bytes| !bytes.is_empty());

                let last_line = if buffer.ends_with(&[0x0A]) {
                    &[]
                } else {
                    lines.next_back().unwrap_or_default()
                };

                for line in lines {
                    if let Ok(data) = std::str::from_utf8(line) {
                        #[derive(Deserialize)]
                        struct Data {
                            choices: Vec<Choice>,
                        }

                        #[derive(Deserialize)]
                        struct Choice {
                            delta: Delta,
                        }

                        #[derive(Deserialize)]
                        struct Delta {
                            content: Option<String>,
                        }

                        if data == "data: [DONE]" {
                            break;
                        }

                        let mut data: Data = serde_json::from_str(
                            data.trim().strip_prefix("data: ").unwrap_or(data),
                        )?;

                        if let Some(choice) = data.choices.first_mut() {
                            if let Some(content) = &mut choice.delta.content {
                                match is_reasoning {
                                    None if content.contains("<think>") => {
                                        is_reasoning = Some(true);
                                        *content = content.replace("<think>", "");
                                    }
                                    Some(true) if content.contains("</think>") => {
                                        is_reasoning = Some(false);
                                        *content = content.replace("</think>", "");
                                    }
                                    _ => {}
                                }

                                let _ = sender
                                    .send(if is_reasoning.unwrap_or_default() {
                                        Token::Reasoning(content.clone())
                                    } else {
                                        Token::Talking(content.clone())
                                    })
                                    .await;
                            }
                        }
                    };
                }

                buffer = last_line.to_vec();
            }

            Ok(())
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn source(&self) -> Source {
        match self.server.as_ref() {
            Server::Container { model, .. } | Server::Process { model, .. } => {
                Source::Local(model.clone())
            }
            Server::Remote(url) => Source::Remote(url.clone()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Cuda,
    Rocm,
}

impl Backend {
    pub fn detect(graphics_adapter: &str) -> Self {
        if graphics_adapter.contains("NVIDIA") {
            Self::Cuda
        } else if graphics_adapter.contains("AMD") {
            Self::Rocm
        } else {
            Self::Cpu
        }
    }

    pub fn uses_gpu(self) -> bool {
        match self {
            Backend::Cuda | Backend::Rocm => true,
            Backend::Cpu => false,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Message {
    System(String),
    Assistant(String),
    User(String),
}

impl Message {
    pub fn to_tuple(&self) -> (&'static str, &str) {
        match self {
            Self::System(content) => ("system", content),
            Self::Assistant(content) => ("assistant", content),
            Self::User(content) => ("user", content),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Reply {
    pub reasoning: Option<Reasoning>,
    pub content: String,
    pub last_token: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Reasoning {
    pub content: String,
    pub duration: Duration,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    Reasoning(String),
    Talking(String),
}

#[derive(Debug)]
enum Server {
    Container {
        id: String,
        model: PathBuf,
    },
    Process {
        _handle: process::Child,
        model: PathBuf,
    },
    Remote(Url),
}

impl Server {
    const PORT: u64 = 8080;

    fn launch_with_executable(
        executable: &'static str,
        model: impl AsRef<Path>,
        backend: Backend,
    ) -> Result<process::Child, Error> {
        let gpu_flags = match backend {
            Backend::Cpu => "",
            Backend::Cuda | Backend::Rocm => "--gpu-layers 80",
        };

        let server = process::Command::new(executable)
            .args(Self::parse_args(&format!(
                "--model {model} --port {port} --host 127.0.0.1 {gpu_flags}",
                port = Self::PORT,
                model = model.as_ref().display()
            )))
            .kill_on_drop(true)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()?;

        Ok(server)
    }

    fn parse_args(command: &str) -> impl Iterator<Item = &str> {
        command
            .split(' ')
            .map(str::trim)
            .filter(|arg| !arg.is_empty())
    }

    fn host(&self) -> String {
        match self {
            Server::Container { .. } | Server::Process { .. } => {
                format!("http://localhost:{port}", port = Self::PORT)
            }
            Server::Remote(url) => url.as_str().trim_end_matches("/").to_owned(),
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        use std::process;

        match self {
            Self::Container { id, .. } => {
                let _ = process::Command::new("docker")
                    .args(["stop", id])
                    .stdin(process::Stdio::null())
                    .stdout(process::Stdio::null())
                    .stderr(process::Stdio::null())
                    .spawn();
            }
            Self::Process { .. } => {}
            Self::Remote(_url) => {}
        }
    }
}

#[derive(Debug, Clone)]
pub enum BootEvent {
    Progressed { stage: &'static str, percent: u32 },
    Logged(String),
}
