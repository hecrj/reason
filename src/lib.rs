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
use std::time::{Duration, Instant};

pub use reqwest::IntoUrl;
pub use url::Url;

#[derive(Debug, Clone)]
pub struct Reason {
    client: reqwest::Client,
    url: Url,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Model(String);

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
    pub async fn connect(url: impl IntoUrl) -> Result<Self, Error> {
        let url = url.into_url()?;
        let client = reqwest::Client::new();

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
        append: &[Message],
        tools: &[Tool],
    ) -> impl Straw<Reply, Event, Error> {
        sipper(move |mut progress| async move {
            let mut completion = self.complete(model, messages, append, tools).pin();
            let mut reply = Reply {
                outputs: Vec::new(),
            };

            while let Some(event) = completion.sip().await {
                reply.update(&event);
                progress.send(event).await;
            }

            Ok(reply)
        })
    }

    pub fn complete(
        &self,
        model: &Model,
        messages: &[Message],
        append: &[Message],
        tools: &[Tool],
    ) -> impl Straw<(), Event, Error> {
        sipper(move |mut sender| async move {
            let client = reqwest::Client::new();

            let request = {
                let messages: Vec<_> = messages
                    .iter()
                    .chain(append)
                    .map(Message::to_json)
                    .collect();

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

            enum Mode {
                Reasoning,
                Messaging,
                ToolCalling,
            }

            let mut mode = None;
            let mut mode_started_at = Instant::now();

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

                    let Ok(data): Result<Data, _> = serde_json::from_slice(&line[PREFIX..]) else {
                        continue;
                    };

                    let Some(choice) = data.choices.first() else {
                        continue;
                    };

                    match &choice.delta {
                        Delta::Text { content } => {
                            match mode {
                                None | Some(Mode::Messaging) if content.contains("<think>") => {
                                    mode = Some(Mode::Reasoning);
                                    mode_started_at = Instant::now();

                                    sender
                                        .send(Event::OutputAdded {
                                            output: Output::Reasoning(Reasoning::default()),
                                        })
                                        .await;

                                    continue;
                                }
                                Some(Mode::Reasoning) if content.contains("</think>") => {
                                    mode = Some(Mode::Messaging);
                                    mode_started_at = Instant::now();

                                    continue;
                                }
                                None => {
                                    mode = Some(Mode::Messaging);
                                    mode_started_at = Instant::now();

                                    sender
                                        .send(Event::OutputAdded {
                                            output: Output::Message(String::new()),
                                        })
                                        .await;
                                }
                                _ => {}
                            }

                            if let Some(Mode::Reasoning | Mode::Messaging) = mode {
                                let _ = sender
                                    .send(Event::TextChanged {
                                        delta: content.clone(),
                                        duration: mode_started_at.elapsed(),
                                    })
                                    .await;
                            }
                        }
                        Delta::Call { tool_calls } => {
                            if !matches!(mode, Some(Mode::ToolCalling)) {
                                mode = Some(Mode::ToolCalling);
                                mode_started_at = Instant::now();

                                sender
                                    .send(Event::OutputAdded {
                                        output: Output::ToolCalls(Vec::new()),
                                    })
                                    .await;
                            }

                            match &tool_calls[0] {
                                ToolCall::New { id, function } => {
                                    sender
                                        .send(Event::ToolCallAdded {
                                            id: id.clone(),
                                            name: function.name.clone(),
                                            arguments: function.arguments.clone(),
                                        })
                                        .await;
                                }
                                ToolCall::Update { function } => {
                                    sender
                                        .send(Event::ArgumentsChanged {
                                            delta: function.arguments.clone(),
                                            duration: mode_started_at.elapsed(),
                                        })
                                        .await;
                                }
                            }
                        }
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
    Assistant(Output),
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
            Self::Assistant(output) => match output {
                Output::Reasoning(reasoning) => json!({
                    "role": "assistant",
                    "content": reasoning.text,
                }),
                Output::Message(text) => json!({
                    "role": "assistant",
                    "content": text,
                }),
                Output::ToolCalls(calls) => {
                    let tool_calls: Vec<_> = calls
                        .iter()
                        .map(|call| match call {
                            tool::Call::Function {
                                id,
                                name,
                                arguments,
                            } => json!({
                                "id": id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": arguments,
                                }
                            }),
                        })
                        .collect();

                    json!({
                        "role": "assistant",
                        "tool_calls": tool_calls,
                    })
                }
            },
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

#[derive(Debug, Clone)]
pub struct Reply {
    pub outputs: Vec<Output>,
}

impl Reply {
    pub fn update(&mut self, event: &Event) {
        match event {
            Event::OutputAdded { output } => {
                self.outputs.push(output.clone());
            }
            Event::TextChanged { delta, duration } => match self.outputs.last_mut() {
                Some(Output::Reasoning(reasoning)) => {
                    reasoning.text.push_str(delta);
                    reasoning.duration = *duration;
                }
                Some(Output::Message(text)) => {
                    text.push_str(delta);
                }
                None | Some(Output::ToolCalls(_)) => {}
            },
            Event::ToolCallAdded {
                id,
                name,
                arguments,
            } => {
                let Some(Output::ToolCalls(calls)) = self.outputs.last_mut() else {
                    return;
                };

                calls.push(tool::Call::Function {
                    id: id.clone(),
                    name: name.clone(),
                    arguments: arguments.clone(),
                });
            }
            Event::ArgumentsChanged { delta, .. } => {
                let Some(Output::ToolCalls(calls)) = self.outputs.last_mut() else {
                    return;
                };

                let Some(tool::Call::Function { arguments, .. }) = calls.last_mut() else {
                    return;
                };

                arguments.push_str(delta);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum Output {
    Reasoning(Reasoning),
    Message(String),
    ToolCalls(Vec<tool::Call>),
}

impl Output {
    pub fn text(&self) -> Option<&str> {
        match self {
            Output::Reasoning(reasoning) => Some(&reasoning.text),
            Output::Message(text) => Some(text),
            Output::ToolCalls(_) => None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Reasoning {
    pub text: String,
    pub duration: Duration,
}

#[derive(Debug, Clone)]
pub enum Event {
    OutputAdded {
        output: Output,
    },
    TextChanged {
        delta: String,
        duration: Duration,
    },
    ToolCallAdded {
        id: tool::Id,
        name: String,
        arguments: String,
    },
    ArgumentsChanged {
        delta: String,
        duration: Duration,
    },
}

impl Event {
    pub fn text(&self) -> Option<&str> {
        match self {
            Event::OutputAdded { output, .. } => output.text(),
            Event::TextChanged { delta, .. } => Some(delta),
            Event::ToolCallAdded { .. } => None,
            Event::ArgumentsChanged { .. } => None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum BootEvent {
    Progressed { stage: &'static str, percent: u32 },
    Logged(String),
}
