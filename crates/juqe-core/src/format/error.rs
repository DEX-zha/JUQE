use std::{error::Error, fmt};

/// Invalid or unsupported expert metadata, with source-line context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormatError(String);
impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
impl Error for FormatError {}
pub(super) fn invalid(message: impl Into<String>) -> FormatError {
    FormatError(message.into())
}
