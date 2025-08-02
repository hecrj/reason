use reason::{Message, Reason};

use anyhow::bail;
use sipper::Sipper;

use std::env;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let Some(model) = env::args().nth(1) else {
        bail!("Model argument not provided!");
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

    println!("-------------------");
    println!("Assistant is ready. Break the ice!");
    println!("-------------------");

    let mut message = String::new();
    let mut messages = vec![Message::system("You are a helpful assistant")];

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

        let mut reply = reason.reply(&messages, &[], &[]).pin();

        println!("");

        while let Some(event) = reply.sip().await {
            if let Some(text) = event.text() {
                print!("{text}");
            }

            io::stdout().flush()?;
        }

        let reply = reply.await?;
        messages.extend(reply.outputs.into_iter().map(Message::Assistant));

        println!("");
    }
}
