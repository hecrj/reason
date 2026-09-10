mod error;

pub mod model;
pub mod tool;

pub use error::Error;
pub use model::Model;
pub use tool::Tool;

use serde::Deserialize;
use serde_json::json;
use sipper::{Sipper, Straw, sipper};

use std::path::PathBuf;
use std::time::Duration;

pub use reqwest::IntoUrl;
pub use url::Url;

#[derive(Debug, Clone)]
pub struct Reason {
    client: reqwest::Client,
    url: Url,
}

#[derive(Debug, Clone)]
pub enum Source {
    Local(PathBuf),
    Remote(Url),
}

impl Reason {
    pub fn connect(
        url: impl IntoUrl,
    ) -> impl Future<Output = Result<(Self, Vec<Model>), Error>> + 'static {
        let url = url.into_url();
        let client = reqwest::Client::new();

        async move {
            let url = url?;
            let reason = Self { client, url };
            let models = reason.list_models().await?;

            Ok((reason, models))
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
            #[serde(default)]
            meta: Option<Meta>,
        }

        #[derive(Deserialize)]
        struct Meta {
            n_ctx: u64,
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
            .map(|model| Model {
                id: model::Id(model.id),
                context_size: model.meta.map(|meta| meta.n_ctx),
            })
            .collect())
    }

    pub fn reply(
        &self,
        model: &model::Id,
        messages: &[Message],
        tools: &[Tool],
    ) -> impl Straw<Reply, Event, Error> {
        sipper(async move |mut progress| {
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
        model: &model::Id,
        messages: &[Message],
        tools: &[Tool],
    ) -> impl Straw<(), Event, Error> {
        sipper(async move |mut sender| {
            let client = reqwest::Client::new();

            let request = {
                let messages: Vec<_> = messages.iter().map(Message::to_json).collect();

                client
                    .post(format!("{url}v1/chat/completions", url = self.url))
                    .json(&json!({
                        "model": model.0,
                        "messages": messages,
                        "tools": tools,
                        "stream": true,
                        "cache_prompt": true,
                        "timings_per_token": true,
                        "return_progress": true,
                        "prompt_cache_options": { "mode": "implicit" },
                    }))
            };

            let mut response = request.send().await?.error_for_status()?;
            let mut buffer = Vec::new();
            let mut parser = Parser::default();

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
                    if let Some(event) = parser.parse(line) {
                        sender.send(event).await;
                    }
                }

                buffer = last_line.to_vec();
            }

            Ok(())
        })
    }
}

/// Stateful conversion of a completion stream's `data:` lines into [`Event`]s
///
/// Derives [`Timings::reasoning`] from the server's cumulative `predicted`
/// stat: a reasoning span is measured against the `predicted` sample of the
/// chunk preceding it (or its own first chunk if there is none), and settles
/// at the span's last chunk. As `predicted` starts at zero when generation
/// begins, the first span includes the time to the first reasoning token.
#[derive(Default)]
struct Parser {
    /// `predicted.total` to measure the open reasoning span against
    reasoning_open: Option<Duration>,
    /// The reasoning time of closed spans
    reasoning: Duration,
    /// `predicted.total` of the last chunk with timings
    predicted: Option<Duration>,
}

impl Parser {
    fn parse(&mut self, line: &[u8]) -> Option<Event> {
        const PREFIX: usize = b"data:".len();

        if log::log_enabled!(log::Level::Debug) {
            log::debug!("{}", String::from_utf8_lossy(line));
        }

        if line.len() < PREFIX {
            return None;
        }

        let Ok(data): Result<Data, _> = serde_json::from_slice(&line[PREFIX..]) else {
            return None;
        };

        let delta = match data.kind {
            Kind::PromptProgress { prompt_progress } => Delta::PromptProcessed(Progress {
                total: prompt_progress.total.max(prompt_progress.processed),
                processed: prompt_progress.processed.max(prompt_progress.cache),
                cached: prompt_progress.cache,
            }),
            Kind::Choices { choices } => {
                let choice = choices.first()?;

                match &choice.delta {
                    Delta_::Reasoning { reasoning_content } => {
                        Delta::ReasoningChanged(reasoning_content.clone())
                    }
                    Delta_::Text { content } => Delta::ContentChanged(content.clone()),
                    Delta_::Call { tool_calls } => Delta::ToolCallsChanged(
                        tool_calls
                            .iter()
                            .map(|call| match call {
                                ToolCall::New { id, function } => {
                                    tool::Delta::CallAdded(tool::Call {
                                        id: id.clone(),
                                        name: function.name.clone(),
                                        arguments: function.arguments.clone(),
                                    })
                                }
                                ToolCall::Update { function } => {
                                    tool::Delta::ArgumentsChanged(function.arguments.clone())
                                }
                            })
                            .collect(),
                    ),
                }
            }
        };

        let mut timings: Option<Timings> = data.timings.map(|timings| Timings {
            cached: timings.cache_n,
            prompt: Generation {
                amount: timings.prompt_n,
                total: Duration::from_secs_f64(timings.prompt_ms / 1_000.),
                token: Duration::from_secs_f64(timings.prompt_per_token_ms / 1_000.),
            },
            predicted: Generation {
                amount: timings.predicted_n,
                total: Duration::from_secs_f64(timings.predicted_ms / 1_000.),
                token: Duration::from_secs_f64(timings.predicted_per_token_ms / 1_000.),
            },
            reasoning: Duration::ZERO,
        });

        if let Some(timings) = &mut timings {
            timings.reasoning = self.sample_reasoning(timings, &delta);
        }

        Some(Event { delta, timings })
    }

    /// Feed a chunk's timings into the reasoning derivation, returning the
    /// cumulative reasoning time through the chunk
    fn sample_reasoning(&mut self, timings: &Timings, delta: &Delta) -> Duration {
        let predicted = timings.predicted.total;
        let reasoning = matches!(delta, Delta::ReasoningChanged(_));

        if !reasoning
            && let Some(open) = self.reasoning_open.take()
            && let Some(end) = self.predicted
        {
            self.reasoning += end.saturating_sub(open);
        }

        if reasoning && self.reasoning_open.is_none() {
            // Measured against the previous chunk, so the content generated
            // before the span is not re-counted as reasoning
            self.reasoning_open = Some(self.predicted.unwrap_or(predicted));
        }

        let through = self
            .reasoning_open
            .map_or(Duration::ZERO, |open| predicted.saturating_sub(open));

        self.predicted = Some(predicted);
        self.reasoning + through
    }
}

#[derive(Deserialize)]
struct Data {
    #[serde(flatten)]
    kind: Kind,
    #[serde(default)]
    timings: Option<Timings_>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Kind {
    Choices { choices: Vec<Choice> },
    PromptProgress { prompt_progress: PromptProgress },
}

#[derive(Deserialize)]
struct Choice {
    delta: Delta_,
}

#[derive(Deserialize)]
struct Timings_ {
    cache_n: u64,
    prompt_n: u64,
    prompt_ms: f64,
    prompt_per_token_ms: f64,
    predicted_n: u64,
    predicted_ms: f64,
    predicted_per_token_ms: f64,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Delta_ {
    Text { content: String },
    Reasoning { reasoning_content: String },
    Call { tool_calls: Vec<ToolCall> },
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

#[derive(Deserialize)]
struct PromptProgress {
    total: u64,
    cache: u64,
    processed: u64,
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
        match &event.delta {
            Delta::PromptProcessed(_) => {}
            Delta::ReasoningChanged(delta) => {
                self.reasoning.push_str(delta);
            }
            Delta::ContentChanged(delta) => {
                self.content.push_str(delta);
            }
            Delta::ToolCallsChanged(deltas) => {
                for delta in deltas {
                    match delta {
                        tool::Delta::CallAdded(call) => {
                            self.tool_calls.push(call.clone());
                        }
                        tool::Delta::ArgumentsChanged(delta) => {
                            let Some(call) = self.tool_calls.last_mut() else {
                                return;
                            };

                            call.arguments.push_str(delta);
                        }
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Event {
    pub delta: Delta,
    pub timings: Option<Timings>,
}

#[derive(Debug, Clone)]
pub enum Delta {
    PromptProcessed(Progress),
    ReasoningChanged(String),
    ContentChanged(String),
    ToolCallsChanged(Vec<tool::Delta>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Progress {
    pub total: u64,
    pub processed: u64,
    pub cached: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Timings {
    pub cached: u64,
    pub prompt: Generation,
    pub predicted: Generation,
    /// The cumulative time the model spent reasoning through this chunk
    pub reasoning: Duration,
}

impl Timings {
    pub fn total_tokens(&self) -> u64 {
        self.cached + self.prompt.amount + self.predicted.amount
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Generation {
    pub amount: u64,
    pub total: Duration,
    pub token: Duration,
}

impl Delta {
    pub fn text(&self) -> Option<&str> {
        match self {
            Self::ReasoningChanged(delta) | Self::ContentChanged(delta) => Some(delta),
            Self::PromptProcessed(_) | Self::ToolCallsChanged(_) => None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum BootEvent {
    Progressed { stage: &'static str, percent: u32 },
    Logged(String),
}

#[cfg(test)]
mod test {
    use super::*;

    fn data(value: serde_json::Value) -> String {
        format!("data: {value}")
    }

    fn timings(predicted_ms: u64) -> serde_json::Value {
        json!({
            "cache_n": 0,
            "prompt_n": 100,
            "prompt_ms": 50.0,
            "prompt_per_token_ms": 0.5,
            "predicted_n": 10,
            "predicted_ms": predicted_ms as f64,
            "predicted_per_token_ms": 50.0,
        })
    }

    fn parse(lines: &[String]) -> Vec<Event> {
        let mut parser = Parser::default();

        lines
            .iter()
            .filter_map(|line| parser.parse(line.as_bytes()))
            .collect()
    }

    fn reasoning(events: &[Event], index: usize) -> Duration {
        events[index].timings.expect("timings").reasoning
    }

    #[test]
    fn reasoning_timing() {
        let events = parse(&[
            data(json!({"prompt_progress": {"total": 100, "cache": 10, "processed": 10}})),
            data(json!({
                "prompt_progress": {"total": 100, "cache": 10, "processed": 100},
                "timings": timings(0),
            })),
            data(json!({
                "choices": [{"delta": {"reasoning_content": "hmm"}}],
                "timings": timings(1_000),
            })),
            data(json!({
                "choices": [{"delta": {"reasoning_content": " hmm"}}],
                "timings": timings(3_500),
            })),
            data(json!({
                "choices": [{"delta": {"content": "Hello"}}],
                "timings": timings(4_000),
            })),
        ]);

        assert_eq!(reasoning(&events, 1), Duration::ZERO);
        assert_eq!(reasoning(&events, 2), Duration::from_millis(1_000));
        assert_eq!(reasoning(&events, 3), Duration::from_millis(3_500));

        // Settled when the span closes, and sticky thereafter
        assert_eq!(reasoning(&events, 4), Duration::from_millis(3_500));
    }

    #[test]
    fn interleaved_reasoning() {
        let events = parse(&[
            data(json!({
                "prompt_progress": {"total": 100, "cache": 0, "processed": 100},
                "timings": timings(0),
            })),
            data(json!({
                "choices": [{"delta": {"reasoning_content": "a"}}],
                "timings": timings(1_000),
            })),
            data(json!({
                "choices": [{"delta": {"reasoning_content": "b"}}],
                "timings": timings(3_000),
            })),
            data(json!({
                "choices": [{"delta": {"content": "c"}}],
                "timings": timings(4_000),
            })),
            data(json!({
                "choices": [{"delta": {"reasoning_content": "d"}}],
                "timings": timings(5_000),
            })),
            data(json!({
                "choices": [{"delta": {"content": "e"}}],
                "timings": timings(6_000),
            })),
        ]);

        // The first span settles at its own last chunk (3_000), not the
        // content chunk's 4_000; the second span measures 5_000 - 4_000
        assert_eq!(reasoning(&events, 3), Duration::from_millis(3_000));
        assert_eq!(reasoning(&events, 4), Duration::from_millis(4_000));
        assert_eq!(reasoning(&events, 5), Duration::from_millis(4_000));
    }

    #[test]
    fn reasoning_without_prompt_progress() {
        let events = parse(&[
            data(json!({
                "choices": [{"delta": {"reasoning_content": "a"}}],
                "timings": timings(500),
            })),
            data(json!({
                "choices": [{"delta": {"reasoning_content": "b"}}],
                "timings": timings(800),
            })),
            data(json!({
                "choices": [{"delta": {"content": "c"}}],
                "timings": timings(900),
            })),
        ]);

        // With no preceding timed sample, the span measures from its own
        // first chunk
        assert_eq!(reasoning(&events, 0), Duration::ZERO);
        assert_eq!(reasoning(&events, 1), Duration::from_millis(300));
        assert_eq!(reasoning(&events, 2), Duration::from_millis(300));
    }

    #[test]
    fn missing_timings() {
        let events = parse(&[
            data(json!({"choices": [{"delta": {"reasoning_content": "a"}}]})),
            data(json!({"choices": [{"delta": {"content": "b"}}]})),
        ]);

        assert_eq!(events.len(), 2);
        assert!(events[0].timings.is_none());
        assert!(events[1].timings.is_none());
    }

    #[test]
    fn malformed_lines() {
        let events = parse(&[
            String::from("data: [DONE]"),
            String::from("data:"),
            data(json!({"choices": []})),
            data(json!({
                "choices": [{"delta": {"content": "ok"}}],
                "timings": timings(100),
            })),
        ]);

        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0].delta,
            Delta::ContentChanged(content) if content == "ok"
        ));
    }

    #[test]
    fn tool_call_delta() {
        let events = parse(&[data(json!({
            "choices": [{"delta": {
                "tool_calls": [
                    {"id": "call-1", "function": {"name": "bash", "arguments": "{\"command\":"}},
                    {"function": {"arguments": "\"ls\"}"}}
                ]
            }}]
        }))]);

        let Delta::ToolCallsChanged(deltas) = &events[0].delta else {
            panic!("expected tool calls");
        };

        assert!(matches!(
            &deltas[0],
            tool::Delta::CallAdded(call) if call.name == "bash"
        ));
        assert!(matches!(
            &deltas[1],
            tool::Delta::ArgumentsChanged(fragment) if fragment == "\"ls\"}"
        ));
    }
}
