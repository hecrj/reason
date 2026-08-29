use reason::{Message, Reason};

use llama_server::Server;
use sipper::Sipper;

use std::io::{self, Write};
use std::process;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let server = {
        let latest = llama_server::Build::latest().await?;
        Server::download(latest, llama_server::backend::Set::all()).await?
    };

    let mut instance = server
        .boot("models", llama_server::Settings::default())
        .await?;
    instance.wait_until_ready().await?;

    let reason = Reason::connect(instance.url().clone()).await?;
    let models = reason.list_models().await?;

    if models.is_empty() {
        println!("No models available in the server!");
        process::exit(1);
    }

    println!("-------------------");
    println!("Choose a model:");

    for (i, model) in models.iter().enumerate() {
        println!("  {n}. {model}", n = i + 1);
    }

    let model = {
        let mut n = String::new();
        let _ = io::stdin().read_line(&mut n)?;
        let n = n
            .trim()
            .parse::<usize>()
            .expect("model selection must be a number");

        models
            .get(n.saturating_sub(1))
            .expect("model selection must be in range")
    };

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

        let mut reply = reason.reply(model, &messages, &[], &[]).pin();

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
