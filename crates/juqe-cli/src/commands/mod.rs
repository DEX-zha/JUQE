mod inspect;
mod verify;

use crate::{
    Result,
    args::{Command, USAGE},
    output,
};
use std::io::{self, Write};

pub(crate) fn execute(command: Command) -> Result<()> {
    let mut stdout = io::stdout().lock();
    match command {
        Command::Help => writeln!(stdout, "{USAGE}")?,
        Command::Inspect { index } => inspect::run(&index, &mut stdout)?,
        Command::Verify(arguments) => {
            let result = verify::run(&arguments)?;
            output::verification(&mut stdout, arguments.layer, arguments.expert, &result)?;
        }
    }
    Ok(())
}
