//! Command-line tools for inspecting and verifying JUQE expert packs.
#![forbid(unsafe_code)]

mod args;
mod commands;
mod output;

use std::{error::Error, process::ExitCode};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn main() -> ExitCode {
    let result = args::Command::parse(std::env::args_os().skip(1))
        .map_err(Into::into)
        .and_then(commands::execute);
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("juqe: {error}");
            ExitCode::FAILURE
        }
    }
}
