use reason::tool;
use reason::{Message, Output, Reason, Tool};

use anyhow::bail;
use sipper::Sipper;
use techne::client::{self, Client};
use techne::mcp;
use techne::server::{self, Server};

use std::env;
use std::io::{self, Write};

#[tokio::main]
pub async fn main() -> anyhow::Result<()> {
    let Some(model) = env::args().nth(1) else {
        bail!("Model argument not provided!");
    };

    if model == "--server" {
        return Ok(run_mcp_server().await?);
    }

    let mut mcp = {
        print!("> URL of MCP server (blank to simulate one): ");
        io::stdout().flush()?;

        let mut address = String::new();
        let _ = io::stdin().read_line(&mut address)?;

        if address.trim().is_empty() {
            let transport =
                client::Stdio::run("cargo", ["run", "--example", "mcp", "--", "--server"])?;

            Client::new("reason", env!("CARGO_PKG_VERSION"), transport).await?
        } else {
            let transport = client::Http::new(address.trim())?;

            Client::new("reason", env!("CARGO_PKG_VERSION"), transport).await?
        }
    };

    println!("");

    let server = mcp.server().information();

    println!(
        "- Connected to MCP server: {} ({})",
        server.name, server.version,
    );

    let tools: Vec<_> = mcp
        .list_tools()
        .await?
        .into_iter()
        .map(Tool::from)
        .collect();

    println!("- Available tools:");

    for Tool::Function { function } in &tools {
        println!("    {}\n        {}", function.name, function.description);
    }

    println!("");
    println!("- Booting {model}...");

    let mut boot = Reason::boot(model, reason::Backend::Cuda).pin();

    while let Some(progress) = boot.sip().await {
        match progress {
            reason::BootEvent::Progressed { stage, percent } => {
                println!("- {stage} ({percent}%)");
            }
            reason::BootEvent::Logged(_log) => {}
        }
    }

    let reason = boot.await?;

    println!("");
    println!("-------------------");
    println!("Assistant is ready. Break the ice!");
    println!("-------------------");

    let mut messages = vec![Message::system("You are a helpful assistant.")];
    let mut message = String::new();
    let mut is_processing = false;

    loop {
        if !is_processing {
            print!("\n> ");
            io::stdout().flush()?;

            let _ = io::stdin().read_line(&mut message)?;

            if message.trim().is_empty() {
                if message.contains("\n") {
                    message.clear();
                    continue;
                }

                return Ok(());
            }

            messages.push(Message::User(message.trim().to_owned()));
            message.clear();
        }

        let mut reply = reason.reply(&messages, &[], &tools).pin();

        println!("");

        while let Some(event) = reply.sip().await {
            if let Some(text) = event.text() {
                print!("{text}");
            }

            io::stdout().flush()?;
        }

        println!("");

        let reply = reply.await?;
        is_processing = false;

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

                println!("=> {name}: {arguments}");

                let response = mcp.call_tool(name, arguments).await?;

                let content = match response.content {
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
                };

                println!("<= {content}");
                println!("");

                messages.push(Message::Tool(tool::Response { id, content }));

                is_processing = true;
            }
        }
    }
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
