use reason::{Assistant, Message};

use anyhow::bail;
use sipper::Sipper;

use std::env;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let Some(model) = env::args().nth(1) else {
        bail!("Model argument not provided!");
    };

    let mut boot = Assistant::boot(model, reason::Backend::Cuda).pin();

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

    let assistant = boot.await?;

    println!("-------------------");
    println!("Assistant is ready. Break the ice!");
    println!("-------------------");

    let mut message = String::new();
    let mut messages = Vec::new();

    loop {
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

        let mut reply = assistant
            .reply("You are a helpful assistant.", &messages, &[])
            .pin();

        println!("");

        while let Some((_reply, token)) = reply.sip().await {
            let token = match token {
                reason::Token::Reasoning(token) => token,
                reason::Token::Talking(token) => token,
            };

            print!("{token}");
            io::stdout().flush()?;
        }

        let reply = reply.await?;
        messages.push(Message::Assistant(reply.content));

        println!("");
    }
}
