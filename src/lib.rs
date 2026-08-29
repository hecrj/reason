mod error;

pub mod tool;

pub use error::Error;
pub use tool::Tool;

use serde::Deserialize;
use serde_json::json;
use sipper::{Sipper, Straw, sipper};
use tokio::time;

use core::fmt;
use std::path::PathBuf;
use std::time::Duration;

pub use reqwest::IntoUrl;
pub use url::Url;

#[derive(Debug, Clone)]
pub struct Reason {
    client: reqwest::Client,
    url: Url,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Model(String);

impl Model {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone)]
pub enum Source {
    Local(PathBuf),
    Remote(Url),
}

impl Reason {
    pub fn connect(url: impl IntoUrl) -> impl Future<Output = Result<Self, Error>> + 'static {
        let url = url.into_url();
        let client = reqwest::Client::new();

        async move {
            let url = url?;

            loop {
                if client
                    .get(format!("{url}v1/models"))
                    .timeout(Duration::from_secs(5))
                    .send()
                    .await?
                    .error_for_status()
                    .is_ok()
                {
                    break;
                }

                time::sleep(Duration::from_secs(1)).await;
            }

            Ok(Self { client, url })
        }
    }

    pub fn url(&self) -> &Url {
        &self.url
    }

    fn endpoint(&self, path: &str) -> String {
        format!("{}v1/{}", self.url, path)
    }

    pub async fn list_models(&self) -> Result<Vec<Model>, Error> {
        #[derive(Deserialize)]
        struct Response {
            data: Vec<ResponseModel>,
        }

        #[derive(Deserialize)]
        struct ResponseModel {
            id: String,
        }

        let models: Response = self
            .client
            .get(self.endpoint("models"))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;

        Ok(models
            .data
            .into_iter()
            .map(|model| Model(model.id))
            .collect())
    }

    pub fn reply(
        &self,
        model: &Model,
        messages: &[Message],
        tools: &[Tool],
    ) -> impl Straw<Reply, Event, Error> {
        sipper(move |mut progress| async move {
            let mut completion = self.complete(model, messages, tools).pin();
            let mut reply = Reply::default();

            while let Some(event) = completion.sip().await {
                reply.update(&event);
                progress.send(event).await;
            }

            completion.await?;

            Ok(reply)
        })
    }

    pub fn complete(
        &self,
        model: &Model,
        messages: &[Message],
        tools: &[Tool],
    ) -> impl Straw<(), Event, Error> {
        sipper(move |mut sender| async move {
            let client = reqwest::Client::new();

            let request = {
                let messages: Vec<_> = messages.iter().map(Message::to_json).collect();

                client
                    .post(format!("{url}v1/chat/completions", url = self.url,))
                    .json(&json!({
                        "model": model.0,
                        "messages": messages,
                        "tools": tools,
                        "stream": true,
                        "cache_prompt": true,
                    }))
            };

            let mut response = request.send().await?.error_for_status()?;
            let mut buffer = Vec::new();

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
                    #[derive(Deserialize)]
                    struct Data {
                        choices: Vec<Choice>,
                    }

                    #[derive(Deserialize)]
                    struct Choice {
                        delta: Delta,
                    }

                    #[derive(Deserialize)]
                    #[serde(untagged)]
                    enum Delta {
                        Text { content: String },
                        Reasoning { reasoning_content: String },
                        Call { tool_calls: [ToolCall; 1] },
                    }

                    #[derive(Deserialize)]
                    #[serde(untagged)]
                    enum ToolCall {
                        New { id: tool::Id, function: Function },
                        Update { function: FunctionUpdate },
                    }

                    #[derive(Deserialize)]
                    struct Function {
                        name: String,
                        arguments: String,
                    }

                    #[derive(Deserialize)]
                    struct FunctionUpdate {
                        arguments: String,
                    }

                    const PREFIX: usize = b"data:".len();

                    if line.len() < PREFIX {
                        continue;
                    }

                    if log::log_enabled!(log::Level::Debug) {
                        log::debug!("{}", str::from_utf8(line).unwrap_or_default());
                    }

                    let Ok(data): Result<Data, _> = serde_json::from_slice(&line[PREFIX..]) else {
                        continue;
                    };

                    let Some(choice) = data.choices.first() else {
                        continue;
                    };

                    match &choice.delta {
                        Delta::Reasoning { reasoning_content } => {
                            let _ = sender
                                .send(Event::ReasoningChanged {
                                    delta: reasoning_content.clone(),
                                })
                                .await;
                        }
                        Delta::Text { content } => {
                            let _ = sender
                                .send(Event::ContentChanged {
                                    delta: content.clone(),
                                })
                                .await;
                        }
                        Delta::Call { tool_calls } => match &tool_calls[0] {
                            ToolCall::New { id, function } => {
                                sender
                                    .send(Event::ToolCallAdded(tool::Call {
                                        id: id.clone(),
                                        name: function.name.clone(),
                                        arguments: function.arguments.clone(),
                                    }))
                                    .await;
                            }
                            ToolCall::Update { function } => {
                                sender
                                    .send(Event::ArgumentsChanged {
                                        delta: function.arguments.clone(),
                                    })
                                    .await;
                            }
                        },
                    }
                }

                buffer = last_line.to_vec();
            }

            Ok(())
        })
    }
}

#[derive(Debug, Clone)]
pub enum Message {
    System(String),
    Assistant(Reply),
    User(String),
    Tool(tool::Response),
}

impl Message {
    pub fn system(prompt: impl AsRef<str>) -> Self {
        Self::System(prompt.as_ref().to_owned())
    }

    pub fn user(prompt: impl AsRef<str>) -> Self {
        Self::User(prompt.as_ref().to_owned())
    }

    pub fn to_json(&self) -> serde_json::Value {
        match self {
            Self::System(content) => json!({
                "role": "system",
                "content": content,
            }),
            Self::Assistant(reply) => {
                let mut message = serde_json::Map::new();

                message.insert(
                    "role".to_owned(),
                    serde_json::Value::String("assistant".to_owned()),
                );

                if !reply.reasoning.is_empty() {
                    message.insert(
                        "reasoning_content".to_owned(),
                        serde_json::Value::String(reply.reasoning.clone()),
                    );
                }

                message.insert(
                    "content".to_owned(),
                    serde_json::Value::String(reply.content.clone()),
                );

                if !reply.tool_calls.is_empty() {
                    let tool_calls: Vec<_> = reply
                        .tool_calls
                        .iter()
                        .map(|call| {
                            json!({
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": call.arguments,
                                }
                            })
                        })
                        .collect();

                    message.insert(
                        "tool_calls".to_owned(),
                        serde_json::Value::Array(tool_calls),
                    );
                }

                serde_json::Value::Object(message)
            }
            Self::User(content) => json!({
                "role": "user",
                "content": content,
            }),
            Self::Tool(response) => json!({
                "role": "tool",
                "tool_call_id": response.id,
                "content": response.content,
            }),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Reply {
    pub reasoning: String,
    pub content: String,
    pub tool_calls: Vec<tool::Call>,
}

impl Reply {
    pub fn update(&mut self, event: &Event) {
        match event {
            Event::ReasoningChanged { delta } => {
                self.reasoning.push_str(delta);
            }
            Event::ContentChanged { delta } => {
                self.content.push_str(delta);
            }
            Event::ToolCallAdded(call) => {
                self.tool_calls.push(call.clone());
            }
            Event::ArgumentsChanged { delta, .. } => {
                let Some(call) = self.tool_calls.last_mut() else {
                    return;
                };

                call.arguments.push_str(delta);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum Event {
    ReasoningChanged { delta: String },
    ContentChanged { delta: String },
    ToolCallAdded(tool::Call),
    ArgumentsChanged { delta: String },
}

impl Event {
    pub fn text(&self) -> Option<&str> {
        match self {
            Event::ReasoningChanged { delta, .. } | Event::ContentChanged { delta, .. } => {
                Some(delta)
            }
            Event::ToolCallAdded { .. } | Event::ArgumentsChanged { .. } => None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum BootEvent {
    Progressed { stage: &'static str, percent: u32 },
    Logged(String),
}
