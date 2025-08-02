use reason::tool;
use reason::{Message, Output, Reason, Tool};

use anyhow::bail;
use sipper::Sipper;
use techne::client::{self, Client};
use techne::mcp;
use techne::server::{self, Server};

use std::env;
use std::io;

#[tokio::main]
pub async fn main() -> anyhow::Result<()> {
    let Some(model) = env::args().nth(1) else {
        bail!("Model argument not provided!");
    };

    if model == "--server" {
        return Ok(run_mcp_server().await?);
    }

    let mut mcp = {
        let transport = client::Stdio::run("cargo", ["run", "--example", "mcp", "--", "--server"])?;
        Client::new("reason", env!("CARGO_PKG_VERSION"), transport).await?
    };

    let mut boot = Reason::boot(model, reason::Backend::Cuda).pin();

    while let Some(progress) = boot.sip().await {
        match progress {
            reason::BootEvent::Progressed { stage, percent } => {
                println!("{stage} ({percent}%)");
            }
            reason::BootEvent::Logged(log) => {
                println!("{log}");
            }
        }
    }

    let reason = boot.await?;

    let tools: Vec<_> = mcp
        .list_tools()
        .await?
        .into_iter()
        .map(Tool::from)
        .collect();

    dbg!(&tools);

    let mut messages = vec![
        Message::system("You are a helpful assistant."),
        Message::user("What is the weather in Night City?"),
    ];

    let reply = reason.reply(&messages, &[], &tools).await?;

    dbg!(&reply);

    for output in reply.outputs {
        messages.push(Message::Assistant(output.clone()));

        let Output::ToolCalls(tools) = output else {
            continue;
        };

        for tool in tools {
            let tool::Call::Function {
                id,
                name,
                arguments,
            } = tool;

            let Ok(arguments) = serde_json::from_str(&arguments) else {
                continue;
            };

            let response = mcp.call_tool(name, arguments).await?;

            messages.push(Message::Tool(tool::Response {
                id,
                content: match response.content {
                    mcp::server::Content::Unstructured(items) => items
                        .into_iter()
                        .filter_map(|item| {
                            if let mcp::server::content::Unstructured::Text { text } = item {
                                Some(text)
                            } else {
                                None
                            }
                        })
                        .collect(),
                    mcp::server::Content::Structured(value) => serde_json::to_string(&value)?,
                },
            }));
        }
    }

    let reply = reason.reply(&messages, &[], &tools).await?;

    dbg!(&reply);

    Ok(())
}

async fn run_mcp_server() -> io::Result<()> {
    use server::tool::{string, tool};

    let server = Server::new("weather-station", env!("CARGO_PKG_VERSION"));
    let transport = server::Stdio::current();

    let tools = [tool(
        fetch_weather,
        string("location", "The location to fetch the weather from"),
    )
    .name("fetch_weather")
    .description("Returns the weather for the provided location")];

    server.tools(tools).run(transport).await
}

async fn fetch_weather(location: String) -> String {
    format!("It is sunny in {location}.")
}
