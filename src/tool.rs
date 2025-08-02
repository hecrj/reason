use serde::{Deserialize, Serialize};

pub use skema::Schema;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Tool {
    Function { function: Function },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    pub description: String,
    pub parameters: Schema,
}

#[derive(Debug, Clone)]
pub enum Call {
    Function {
        id: Id,
        name: String,
        arguments: String,
    },
}

#[derive(Debug, Clone)]
pub struct Response {
    pub id: Id,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Id(String);

#[cfg(feature = "techne")]
mod techne {
    use super::*;

    use techne_mcp as mcp;

    impl From<mcp::server::Tool> for Tool {
        fn from(tool: mcp::server::Tool) -> Self {
            Self::Function {
                function: Function {
                    name: tool.name,
                    description: tool.description,
                    parameters: tool.input_schema,
                },
            }
        }
    }
}
