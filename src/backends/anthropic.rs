//! Anthropic API client implementation for chat and completion functionality.
//!
//! This module provides integration with Anthropic's Claude models through their API.

use std::collections::HashMap;
use std::sync::Arc;

use crate::{
    builder::{LLMBackend, SystemContent, SystemPrompt},
    chat::{
        ChatMessage, ChatProvider, ChatResponse, ChatRole, MessageType, StreamChunk, Tool,
        ToolChoice, Usage,
    },
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::{ModelListRawEntry, ModelListRequest, ModelListResponse, ModelsProvider},
    stt::SpeechToTextProvider,
    tts::TextToSpeechProvider,
    FunctionCall, ToolCall,
};
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use chrono::{DateTime, Utc};
use futures::stream::Stream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

mod structured_output;

/// Configuration for the Anthropic client.
///
/// This struct holds all configuration options and is wrapped in an `Arc`
/// to enable cheap cloning of the `Anthropic` client.
#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    /// API key for authentication with Anthropic.
    pub api_key: String,
    /// Model identifier (e.g., "claude-3-sonnet-20240229").
    pub model: String,
    /// Maximum tokens to generate in responses.
    pub max_tokens: u32,
    /// Sampling temperature for response randomness. `None` when the caller
    /// never set one - left unset on the wire rather than defaulted, since
    /// Claude's 5-generation models (Sonnet 5, Opus 5, Fable 5.1, Mythos 5.1)
    /// reject `temperature`/`top_p`/`top_k` outright with a 400 (sampling
    /// controls were replaced by adaptive thinking), while older models still
    /// accept them.
    pub temperature: Option<f32>,
    /// Request timeout in seconds.
    pub timeout_seconds: u64,
    /// System prompt to guide model behavior.
    pub system: SystemPrompt,
    /// Top-p (nucleus) sampling parameter.
    pub top_p: Option<f32>,
    /// Top-k sampling parameter.
    pub top_k: Option<u32>,
    /// Available tools for the model to use.
    pub tools: Option<Vec<Tool>>,
    /// Tool choice configuration.
    pub tool_choice: Option<ToolChoice>,
    /// Whether extended thinking is enabled.
    pub reasoning: bool,
    /// Budget tokens for extended thinking.
    pub thinking_budget_tokens: Option<u32>,
    /// Native structured output configuration sent to the Messages API.
    pub output_config: Option<Value>,
    /// Original constraints enforced locally when the native schema is adapted,
    /// or when native output is unavailable for the schema at all. These
    /// requests are buffered before exposing streaming output.
    pub output_validator: Option<Arc<jsonschema::JSONSchema>>,
    /// Set when the caller's schema uses a construct Anthropic's native
    /// `output_config` cannot express (empty/boolean sub-schema, a free-form
    /// object, `patternProperties`, or `additionalProperties` other than
    /// `false`). Appended to the system prompt so the model is still told the
    /// expected shape, with `output_validator` enforcing it locally.
    pub output_schema_instruction: Option<String>,
}

/// Client for interacting with Anthropic's API.
///
/// Provides methods for chat and completion requests using Anthropic's models.
///
/// The client uses `Arc` internally for configuration, making cloning cheap
/// (only an atomic reference count increment). This allows sharing a single
/// client across multiple tasks without expensive deep copies.
#[derive(Debug, Clone)]
pub struct Anthropic {
    /// Shared configuration wrapped in Arc for cheap cloning.
    pub config: Arc<AnthropicConfig>,
    /// HTTP client for making requests.
    pub client: Client,
}

/// Anthropic-specific tool format that matches their API structure
#[derive(Serialize, Debug)]
struct AnthropicTool<'a> {
    name: &'a str,
    description: &'a str,
    #[serde(rename = "input_schema")]
    schema: &'a serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<&'a serde_json::Value>,
}

/// Configuration for the thinking feature
#[derive(Serialize, Debug)]
struct ThinkingConfig {
    #[serde(rename = "type")]
    thinking_type: String,
    budget_tokens: u32,
}

/// System prompt in the request - can be either a string or vector of content objects
#[derive(Serialize, Debug)]
#[serde(untagged)]
enum RequestSystemPrompt<'a> {
    String(&'a str),
    Messages(&'a [SystemContent]),
}

/// Owned counterpart of `RequestSystemPrompt`, used when the system prompt
/// must be extended (e.g. with a schema instruction) rather than borrowed
/// as-is from `AnthropicConfig`.
enum OwnedSystemPrompt {
    String(String),
    Messages(Vec<SystemContent>),
}

impl OwnedSystemPrompt {
    fn as_request(&self) -> RequestSystemPrompt<'_> {
        match self {
            OwnedSystemPrompt::String(s) => RequestSystemPrompt::String(s),
            OwnedSystemPrompt::Messages(m) => RequestSystemPrompt::Messages(m),
        }
    }
}

/// Request payload for Anthropic's messages API endpoint.
#[derive(Serialize, Debug)]
struct AnthropicCompleteRequest<'a> {
    messages: Vec<AnthropicMessage<'a>>,
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<RequestSystemPrompt<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<ThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_config: Option<&'a Value>,
}

/// Individual message in an Anthropic chat conversation.
#[derive(Serialize, Debug)]
struct AnthropicMessage<'a> {
    role: &'a str,
    content: Vec<MessageContent<'a>>,
}

#[derive(Serialize, Debug)]
struct MessageContent<'a> {
    #[serde(rename = "type")]
    message_type: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_url: Option<ImageUrlContent<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<ImageSource<'a>>,
    // tool use
    #[serde(skip_serializing_if = "Option::is_none", rename = "id")]
    tool_use_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "name")]
    tool_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "input")]
    tool_input: Option<Value>,
    // tool result
    #[serde(skip_serializing_if = "Option::is_none", rename = "tool_use_id")]
    tool_result_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "content")]
    tool_output: Option<String>,
}

#[derive(Serialize, Debug)]
struct ImageUrlContent<'a> {
    url: &'a str,
}

#[derive(Serialize, Debug)]
struct ImageSource<'a> {
    #[serde(rename = "type")]
    source_type: &'a str,
    media_type: &'a str,
    data: String,
}

/// Response from Anthropic's messages API endpoint.
#[derive(Deserialize, Debug)]
struct AnthropicCompleteResponse {
    content: Vec<AnthropicContent>,
    usage: Option<AnthropicUsage>,
}

/// Usage information from Anthropic API response.
#[derive(Deserialize, Debug)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_creation_input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_read_input_tokens: Option<u32>,
}

/// Content block within an Anthropic API response.
#[derive(Serialize, Deserialize, Debug)]
struct AnthropicContent {
    text: Option<String>,
    #[serde(rename = "type")]
    content_type: Option<String>,
    thinking: Option<String>,
    name: Option<String>,
    input: Option<serde_json::Value>,
    id: Option<String>,
}

/// Response from Anthropic's streaming messages API endpoint.
/// https://docs.anthropic.com/en/api/messages-streaming
/// https://docs.anthropic.com/claude/reference/messages-streaming
#[derive(Deserialize, Debug)]
struct AnthropicStreamResponse {
    #[serde(rename = "type")]
    response_type: String,
    /// Index of the content block (for content_block_start, content_block_delta, content_block_stop)
    index: Option<usize>,
    /// Content block for content_block_start events
    content_block: Option<AnthropicStreamContentBlock>,
    /// Delta for content_block_delta and message_delta events
    delta: Option<AnthropicDelta>,
}

/// Content block within an Anthropic streaming content_block_start event.
#[derive(Deserialize, Debug)]
struct AnthropicStreamContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    /// Tool use ID (for tool_use blocks)
    id: Option<String>,
    /// Tool name (for tool_use blocks)
    name: Option<String>,
    /// Initial text (for text blocks, usually empty)
    #[allow(dead_code)]
    text: Option<String>,
}

/// Delta content within an Anthropic streaming response.
#[derive(Deserialize, Debug)]
struct AnthropicDelta {
    #[serde(rename = "type")]
    delta_type: Option<String>,
    /// Text content (for text_delta)
    text: Option<String>,
    /// Partial JSON string (for input_json_delta)
    partial_json: Option<String>,
    /// Stop reason (for message_delta)
    stop_reason: Option<String>,
}

impl std::fmt::Display for AnthropicCompleteResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for content in self.content.iter() {
            match content.content_type {
                Some(ref t) if t == "tool_use" => write!(
                    f,
                    "{{\n \"name\": {}, \"input\": {}\n}}",
                    content.name.clone().unwrap_or_default(),
                    content.input.clone().unwrap_or(serde_json::Value::Null)
                )?,
                Some(ref t) if t == "thinking" => {
                    write!(f, "{}", content.thinking.clone().unwrap_or_default())?
                }
                _ => write!(
                    f,
                    "{}",
                    self.content
                        .iter()
                        .map(|c| c.text.clone().unwrap_or_default())
                        .collect::<Vec<_>>()
                        .join("\n")
                )?,
            }
        }
        Ok(())
    }
}

impl ChatResponse for AnthropicCompleteResponse {
    fn text(&self) -> Option<String> {
        Some(
            self.content
                .iter()
                .filter_map(|c| {
                    if c.content_type == Some("text".to_string()) || c.content_type.is_none() {
                        c.text.clone()
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("\n"),
        )
    }

    fn thinking(&self) -> Option<String> {
        self.content
            .iter()
            .find(|c| c.content_type == Some("thinking".to_string()))
            .and_then(|c| c.thinking.clone())
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        match self
            .content
            .iter()
            .filter_map(|c| {
                if c.content_type == Some("tool_use".to_string()) {
                    Some(ToolCall {
                        id: c.id.clone().unwrap_or_default(),
                        call_type: "function".to_string(),
                        function: FunctionCall {
                            name: c.name.clone().unwrap_or_default(),
                            arguments: serde_json::to_string(
                                &c.input.clone().unwrap_or(serde_json::Value::Null),
                            )
                            .unwrap_or_default(),
                        },
                    })
                } else {
                    None
                }
            })
            .collect::<Vec<ToolCall>>()
        {
            v if v.is_empty() => None,
            v => Some(v),
        }
    }

    fn usage(&self) -> Option<Usage> {
        self.usage.as_ref().map(|anthropic_usage| {
            let cached_tokens = anthropic_usage.cache_creation_input_tokens.unwrap_or(0)
                + anthropic_usage.cache_read_input_tokens.unwrap_or(0);
            Usage {
                prompt_tokens: anthropic_usage.input_tokens,
                completion_tokens: anthropic_usage.output_tokens,
                total_tokens: anthropic_usage.input_tokens + anthropic_usage.output_tokens,
                completion_tokens_details: None,
                prompt_tokens_details: if cached_tokens > 0 {
                    Some(crate::chat::PromptTokensDetails {
                        cached_tokens: Some(cached_tokens),
                        audio_tokens: None,
                    })
                } else {
                    None
                },
            }
        })
    }
}

impl Anthropic {
    /// Converts a slice of ChatMessage into Anthropic's message format.
    ///
    /// This helper method handles all message types including text, images, PDFs,
    /// tool use, and tool results.
    fn convert_messages_to_anthropic<'a>(messages: &'a [ChatMessage]) -> Vec<AnthropicMessage<'a>> {
        messages
            .iter()
            .map(|m| AnthropicMessage {
                role: match m.role {
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                },
                content: match &m.message_type {
                    MessageType::Text => vec![MessageContent {
                        message_type: Some("text"),
                        text: Some(&m.content),
                        image_url: None,
                        source: None,
                        tool_use_id: None,
                        tool_input: None,
                        tool_name: None,
                        tool_result_id: None,
                        tool_output: None,
                    }],
                    MessageType::Pdf(raw_bytes) => {
                        vec![MessageContent {
                            message_type: Some("document"),
                            text: None,
                            image_url: None,
                            source: Some(ImageSource {
                                source_type: "base64",
                                media_type: "application/pdf",
                                data: BASE64.encode(raw_bytes),
                            }),
                            tool_use_id: None,
                            tool_input: None,
                            tool_name: None,
                            tool_result_id: None,
                            tool_output: None,
                        }]
                    }
                    MessageType::Image((image_mime, raw_bytes)) => {
                        vec![MessageContent {
                            message_type: Some("image"),
                            text: None,
                            image_url: None,
                            source: Some(ImageSource {
                                source_type: "base64",
                                media_type: image_mime.mime_type(),
                                data: BASE64.encode(raw_bytes),
                            }),
                            tool_use_id: None,
                            tool_input: None,
                            tool_name: None,
                            tool_result_id: None,
                            tool_output: None,
                        }]
                    }
                    MessageType::ImageURL(ref url) => vec![MessageContent {
                        message_type: Some("image_url"),
                        text: None,
                        image_url: Some(ImageUrlContent { url }),
                        source: None,
                        tool_use_id: None,
                        tool_input: None,
                        tool_name: None,
                        tool_result_id: None,
                        tool_output: None,
                    }],
                    MessageType::Audio(_) => vec![],
                    MessageType::ToolUse(calls) => calls
                        .iter()
                        .map(|c| MessageContent {
                            message_type: Some("tool_use"),
                            text: None,
                            image_url: None,
                            source: None,
                            tool_use_id: Some(c.id.clone()),
                            tool_input: Some(
                                serde_json::from_str(&c.function.arguments)
                                    .unwrap_or(c.function.arguments.clone().into()),
                            ),
                            tool_name: Some(c.function.name.clone()),
                            tool_result_id: None,
                            tool_output: None,
                        })
                        .collect(),
                    MessageType::ToolResult(responses) => responses
                        .iter()
                        .map(|r| MessageContent {
                            message_type: Some("tool_result"),
                            text: None,
                            image_url: None,
                            source: None,
                            tool_use_id: None,
                            tool_input: None,
                            tool_name: None,
                            tool_result_id: Some(r.id.clone()),
                            tool_output: Some(r.function.arguments.clone()),
                        })
                        .collect(),
                },
            })
            .collect()
    }

    /// Prepares Anthropic tools and tool_choice from the provided tools and instance configuration.
    ///
    /// Returns a tuple of (anthropic_tools, final_tool_choice) ready for the API request.
    fn prepare_tools_and_choice<'a>(
        tools: Option<&'a [Tool]>,
        instance_tools: Option<&'a [Tool]>,
        tool_choice: &Option<ToolChoice>,
    ) -> (
        Option<Vec<AnthropicTool<'a>>>,
        Option<HashMap<String, String>>,
    ) {
        let maybe_tool_slice: Option<&[Tool]> = tools.or(instance_tools);
        let anthropic_tools = maybe_tool_slice.map(|slice| {
            slice
                .iter()
                .map(|tool| AnthropicTool {
                    name: &tool.function.name,
                    description: &tool.function.description,
                    schema: &tool.function.parameters,
                    cache_control: tool.cache_control.as_ref(),
                })
                .collect::<Vec<_>>()
        });

        let tool_choice_map = match tool_choice {
            Some(ToolChoice::Auto) => {
                Some(HashMap::from([("type".to_string(), "auto".to_string())]))
            }
            Some(ToolChoice::Any) => Some(HashMap::from([("type".to_string(), "any".to_string())])),
            Some(ToolChoice::Tool(ref tool_name)) => Some(HashMap::from([
                ("type".to_string(), "tool".to_string()),
                ("name".to_string(), tool_name.clone()),
            ])),
            Some(ToolChoice::None) => {
                Some(HashMap::from([("type".to_string(), "none".to_string())]))
            }
            None => None,
        };

        let final_tool_choice = if anthropic_tools.is_some() {
            tool_choice_map
        } else {
            None
        };

        (anthropic_tools, final_tool_choice)
    }

    /// Converts a SystemPrompt to the request format
    fn system_to_request(system: &SystemPrompt) -> RequestSystemPrompt<'_> {
        match system {
            SystemPrompt::String(s) => RequestSystemPrompt::String(s),
            SystemPrompt::Messages(msgs) => RequestSystemPrompt::Messages(msgs),
        }
    }

    /// Builds an owned system prompt with a schema instruction appended, for
    /// the prompt-based structured output fallback (see `with_structured_output`).
    fn system_with_schema_instruction(system: &SystemPrompt, instruction: &str) -> OwnedSystemPrompt {
        match system {
            SystemPrompt::String(s) => OwnedSystemPrompt::String(format!("{s}\n\n{instruction}")),
            SystemPrompt::Messages(msgs) => {
                let mut msgs = msgs.clone();
                msgs.push(SystemContent::text(instruction.to_string()));
                OwnedSystemPrompt::Messages(msgs)
            }
        }
    }

    /// Creates a new Anthropic client with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `api_key` - Anthropic API key for authentication
    /// * `model` - Model identifier (defaults to "claude-3-sonnet-20240229")
    /// * `max_tokens` - Maximum tokens in response (defaults to 300)
    /// * `temperature` - Sampling temperature; omitted from the request entirely when unset
    /// * `timeout_seconds` - Request timeout in seconds (defaults to 30)
    /// * `system` - System prompt (defaults to "You are a helpful assistant.")
    /// * `thinking_budget_tokens` - Budget tokens for thinking (optional)
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        api_key: impl Into<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        system: Option<SystemPrompt>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        tools: Option<Vec<Tool>>,
        tool_choice: Option<ToolChoice>,
        reasoning: Option<bool>,
        thinking_budget_tokens: Option<u32>,
    ) -> Self {
        let timeout = timeout_seconds.unwrap_or(30);
        let mut builder = Client::builder();
        if timeout > 0 {
            builder = builder.timeout(std::time::Duration::from_secs(timeout));
        }
        Self::with_client(
            builder.build().expect("Failed to build reqwest Client"),
            api_key,
            model,
            max_tokens,
            temperature,
            timeout_seconds,
            system,
            top_p,
            top_k,
            tools,
            tool_choice,
            reasoning,
            thinking_budget_tokens,
        )
    }

    /// Creates a new Anthropic client with a custom HTTP client.
    ///
    /// This constructor allows sharing a pre-configured `reqwest::Client` across
    /// multiple provider instances, enabling connection pooling and custom
    /// HTTP settings.
    ///
    /// # Arguments
    ///
    /// * `client` - A pre-configured `reqwest::Client` for HTTP requests
    /// * `api_key` - Anthropic API key for authentication
    /// * `model` - Model identifier (defaults to "claude-3-sonnet-20240229")
    /// * `max_tokens` - Maximum tokens in response (defaults to 300)
    /// * `temperature` - Sampling temperature; omitted from the request entirely when unset
    /// * `timeout_seconds` - Request timeout in seconds (defaults to 30)
    /// * `system` - System prompt (defaults to "You are a helpful assistant.")
    /// * `thinking_budget_tokens` - Budget tokens for thinking (optional)
    #[allow(clippy::too_many_arguments)]
    pub fn with_client(
        client: Client,
        api_key: impl Into<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        system: Option<SystemPrompt>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        tools: Option<Vec<Tool>>,
        tool_choice: Option<ToolChoice>,
        reasoning: Option<bool>,
        thinking_budget_tokens: Option<u32>,
    ) -> Self {
        Self {
            config: Arc::new(AnthropicConfig {
                api_key: api_key.into(),
                model: model.unwrap_or_else(|| "claude-3-sonnet-20240229".to_string()),
                max_tokens: max_tokens.unwrap_or(300),
                temperature,
                system: system.unwrap_or_else(|| {
                    SystemPrompt::String("You are a helpful assistant.".to_string())
                }),
                timeout_seconds: timeout_seconds.unwrap_or(30),
                top_p,
                top_k,
                tools,
                tool_choice,
                reasoning: reasoning.unwrap_or(false),
                thinking_budget_tokens,
                output_config: None,
                output_validator: None,
                output_schema_instruction: None,
            }),
            client,
        }
    }

    /// Configure native JSON outputs. String/array upper bounds unsupported by
    /// Anthropic are moved into descriptions and enforced locally against the
    /// original schema. Adapted outputs are buffered before streaming to callers.
    ///
    /// Schemas that use a construct Anthropic's native `output_config` cannot
    /// express at all - an empty or boolean sub-schema, a free-form object
    /// (no `properties`), `patternProperties`, or `additionalProperties` set
    /// to anything but `false` - fall back to a prompt-based JSON instruction
    /// instead of a 400 or a silently narrowed schema (forcing
    /// `additionalProperties: false` onto a caller-intended free-form object
    /// would make it accept only `{}`).
    pub fn with_structured_output(
        mut self,
        format: crate::chat::StructuredOutputFormat,
    ) -> Result<Self, LLMError> {
        let schema = format.schema.ok_or_else(|| {
            LLMError::InvalidRequest(
                "Anthropic structured output requires an explicit JSON schema".to_string(),
            )
        })?;
        let config = Arc::make_mut(&mut self.config);
        if let Some(reason) = structured_output::unsupported_native_construct(&schema) {
            log::debug!(
                "Anthropic native structured output disabled - schema contains {reason}; \
                 falling back to prompt-based JSON output"
            );
            let validator = Arc::new(jsonschema::JSONSchema::compile(&schema).map_err(
                |error| LLMError::InvalidRequest(format!("Invalid Anthropic output schema: {error}")),
            )?);
            config.output_config = None;
            config.output_validator = Some(validator);
            config.output_schema_instruction =
                Some(structured_output::schema_prompt_instruction(&schema));
        } else {
            let wire_schema = structured_output::adapt_schema(&schema);
            let validator = if wire_schema != schema {
                Some(Arc::new(jsonschema::JSONSchema::compile(&schema).map_err(
                    |error| {
                        LLMError::InvalidRequest(format!("Invalid Anthropic output schema: {error}"))
                    },
                )?))
            } else {
                None
            };
            config.output_config = Some(serde_json::json!({
                "format": {"type": "json_schema", "schema": wire_schema}
            }));
            config.output_validator = validator;
            config.output_schema_instruction = None;
        }
        Ok(self)
    }

    async fn parse_completion_response(
        &self,
        response: reqwest::Response,
    ) -> Result<AnthropicCompleteResponse, LLMError> {
        let status = response.status();
        let request_id = response
            .headers()
            .get("request-id")
            .and_then(|value| value.to_str().ok())
            .unwrap_or("unknown")
            .to_owned();
        let body = response.text().await?;
        if !status.is_success() {
            // Preserve the provider diagnosis before consuming the response;
            // retain HttpError so existing gateway status mapping stays intact.
            return Err(LLMError::HttpError(format!(
                "Anthropic API returned {status} (request-id: {request_id}): {body}"
            )));
        }
        let parsed: AnthropicCompleteResponse = serde_json::from_str(&body)
            .map_err(|error| LLMError::HttpError(format!("Failed to parse JSON: {error}")))?;
        if let Some(validator) = self.config.output_validator.as_ref() {
            // Tool calls are intermediate turns, not the final structured output.
            let has_tool_calls = parsed.tool_calls().is_some_and(|calls| !calls.is_empty());
            if !has_tool_calls {
                structured_output::validate_output(validator, &parsed.text().unwrap_or_default())?;
            }
        }
        Ok(parsed)
    }

    pub fn api_key(&self) -> &str {
        &self.config.api_key
    }

    pub fn model(&self) -> &str {
        &self.config.model
    }

    pub fn max_tokens(&self) -> u32 {
        self.config.max_tokens
    }

    pub fn temperature(&self) -> Option<f32> {
        self.config.temperature
    }

    pub fn timeout_seconds(&self) -> u64 {
        self.config.timeout_seconds
    }

    pub fn system(&self) -> &SystemPrompt {
        &self.config.system
    }

    pub fn top_p(&self) -> Option<f32> {
        self.config.top_p
    }

    pub fn top_k(&self) -> Option<u32> {
        self.config.top_k
    }

    pub fn tools(&self) -> Option<&[Tool]> {
        self.config.tools.as_deref()
    }

    pub fn tool_choice(&self) -> Option<&ToolChoice> {
        self.config.tool_choice.as_ref()
    }

    pub fn reasoning(&self) -> bool {
        self.config.reasoning
    }

    pub fn thinking_budget_tokens(&self) -> Option<u32> {
        self.config.thinking_budget_tokens
    }

    pub fn client(&self) -> &Client {
        &self.client
    }
}

const AUDIO_UNSUPPORTED: &str = "Audio messages are not supported by Anthropic chat";

#[async_trait]
impl ChatProvider for Anthropic {
    /// Sends a chat request to Anthropic's API.
    ///
    /// # Arguments
    ///
    /// * `messages` - Slice of chat messages representing the conversation
    /// * `tools` - Optional slice of tools to use in the chat
    ///
    /// # Returns
    ///
    /// The model's response text or an error
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        crate::chat::ensure_no_audio(messages, AUDIO_UNSUPPORTED)?;
        if self.config.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing Anthropic API key".to_string()));
        }

        let anthropic_messages = Self::convert_messages_to_anthropic(messages);
        let (anthropic_tools, final_tool_choice) = Self::prepare_tools_and_choice(
            tools,
            self.config.tools.as_deref(),
            &self.config.tool_choice,
        );

        let thinking = if self.config.reasoning {
            Some(ThinkingConfig {
                thinking_type: "enabled".to_string(),
                budget_tokens: self.config.thinking_budget_tokens.unwrap_or(16000),
            })
        } else {
            None
        };

        let owned_system_prompt = self
            .config
            .output_schema_instruction
            .as_deref()
            .map(|instruction| Self::system_with_schema_instruction(&self.config.system, instruction));
        let system_prompt = match &owned_system_prompt {
            Some(owned) => owned.as_request(),
            None => Self::system_to_request(&self.config.system),
        };

        let req_body = AnthropicCompleteRequest {
            messages: anthropic_messages,
            model: &self.config.model,
            max_tokens: Some(self.config.max_tokens),
            temperature: self.config.temperature,
            system: Some(system_prompt),
            stream: Some(false),
            top_p: self.config.top_p,
            top_k: self.config.top_k,
            tools: anthropic_tools,
            tool_choice: final_tool_choice,
            thinking,
            output_config: self.config.output_config.as_ref(),
        };

        let mut request = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.config.api_key)
            .header("Content-Type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .json(&req_body);

        if self.config.timeout_seconds > 0 {
            request = request.timeout(std::time::Duration::from_secs(self.config.timeout_seconds));
        }

        if log::log_enabled!(log::Level::Trace) {
            if let Ok(json) = serde_json::to_string(&req_body) {
                log::trace!("Anthropic request payload: {}", json);
            }
        }

        log::debug!("Anthropic request: POST /v1/messages");
        let resp = request.send().await?;
        log::debug!("Anthropic HTTP status: {}", resp.status());

        let json_resp = self.parse_completion_response(resp).await?;
        Ok(Box::new(json_resp))
    }

    /// Sends a chat request to Anthropic's API.
    ///
    /// # Arguments
    ///
    /// * `messages` - The conversation history as a slice of chat messages
    ///
    /// # Returns
    ///
    /// The provider's response text or an error
    async fn chat(&self, messages: &[ChatMessage]) -> Result<Box<dyn ChatResponse>, LLMError> {
        self.chat_with_tools(messages, None).await
    }

    /// Sends a streaming chat request to Anthropic's API.
    ///
    /// # Arguments
    ///
    /// * `messages` - Slice of chat messages representing the conversation
    ///
    /// # Returns
    ///
    /// A stream of text tokens or an error
    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError>
    {
        crate::chat::ensure_no_audio(messages, AUDIO_UNSUPPORTED)?;
        if self.config.output_validator.is_some() {
            let response = self.chat_with_tools(messages, None).await?;
            return Ok(Box::pin(futures::stream::iter(vec![Ok(response
                .text()
                .unwrap_or_default())])));
        }
        if self.config.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing Anthropic API key".to_string()));
        }

        let anthropic_messages: Vec<AnthropicMessage> = messages
            .iter()
            .map(|m| AnthropicMessage {
                role: match m.role {
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                },
                content: match &m.message_type {
                    MessageType::Text => vec![MessageContent {
                        message_type: Some("text"),
                        text: Some(&m.content),
                        image_url: None,
                        source: None,
                        tool_use_id: None,
                        tool_input: None,
                        tool_name: None,
                        tool_result_id: None,
                        tool_output: None,
                    }],
                    MessageType::Pdf(_) => unimplemented!(),
                    MessageType::Image((image_mime, raw_bytes)) => {
                        vec![MessageContent {
                            message_type: Some("image"),
                            text: None,
                            image_url: None,
                            source: Some(ImageSource {
                                source_type: "base64",
                                media_type: image_mime.mime_type(),
                                data: BASE64.encode(raw_bytes),
                            }),
                            tool_use_id: None,
                            tool_input: None,
                            tool_name: None,
                            tool_result_id: None,
                            tool_output: None,
                        }]
                    }
                    _ => vec![MessageContent {
                        message_type: Some("text"),
                        text: Some(&m.content),
                        image_url: None,
                        source: None,
                        tool_use_id: None,
                        tool_input: None,
                        tool_name: None,
                        tool_result_id: None,
                        tool_output: None,
                    }],
                },
            })
            .collect();

        let system_prompt = Self::system_to_request(&self.config.system);

        let req_body = AnthropicCompleteRequest {
            messages: anthropic_messages,
            model: &self.config.model,
            max_tokens: Some(self.config.max_tokens),
            temperature: self.config.temperature,
            system: Some(system_prompt),
            stream: Some(true),
            top_p: self.config.top_p,
            top_k: self.config.top_k,
            tools: None,
            tool_choice: None,
            thinking: None,
            output_config: self.config.output_config.as_ref(),
        };

        let mut request = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.config.api_key)
            .header("Content-Type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .json(&req_body);

        if self.config.timeout_seconds > 0 {
            request = request.timeout(std::time::Duration::from_secs(self.config.timeout_seconds));
        }

        let response = request.send().await?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("Anthropic API returned error status: {status}"),
                raw_response: error_text,
            });
        }
        Ok(crate::chat::create_sse_stream(
            response,
            parse_anthropic_sse_chunk,
        ))
    }

    /// Sends a streaming chat request with tool support.
    ///
    /// # Arguments
    ///
    /// * `messages` - Slice of chat messages representing the conversation
    /// * `tools` - Optional slice of tools available for the model to use
    ///
    /// # Returns
    ///
    /// A stream of `StreamChunk` items or an error
    async fn chat_stream_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>>, LLMError>
    {
        if self.config.output_validator.is_some() {
            let response = self.chat_with_tools(messages, tools).await?;
            let mut chunks = vec![Ok(StreamChunk::Text(response.text().unwrap_or_default()))];
            let calls = response.tool_calls().unwrap_or_default();
            let stop_reason = if calls.is_empty() {
                "end_turn"
            } else {
                "tool_use"
            };
            for (index, tool_call) in calls.into_iter().enumerate() {
                chunks.push(Ok(StreamChunk::ToolUseStart {
                    index,
                    id: tool_call.id.clone(),
                    name: tool_call.function.name.clone(),
                }));
                chunks.push(Ok(StreamChunk::ToolUseInputDelta {
                    index,
                    partial_json: tool_call.function.arguments.clone(),
                }));
                chunks.push(Ok(StreamChunk::ToolUseComplete { index, tool_call }));
            }
            chunks.push(Ok(StreamChunk::Done {
                stop_reason: stop_reason.into(),
            }));
            return Ok(Box::pin(futures::stream::iter(chunks)));
        }
        if self.config.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing Anthropic API key".to_string()));
        }

        let anthropic_messages = Self::convert_messages_to_anthropic(messages);
        let (anthropic_tools, final_tool_choice) = Self::prepare_tools_and_choice(
            tools,
            self.config.tools.as_deref(),
            &self.config.tool_choice,
        );

        let system_prompt = Self::system_to_request(&self.config.system);

        let req_body = AnthropicCompleteRequest {
            messages: anthropic_messages,
            model: &self.config.model,
            max_tokens: Some(self.config.max_tokens),
            temperature: self.config.temperature,
            system: Some(system_prompt),
            stream: Some(true),
            top_p: self.config.top_p,
            top_k: self.config.top_k,
            tools: anthropic_tools,
            tool_choice: final_tool_choice,
            output_config: self.config.output_config.as_ref(),
            thinking: None, // Thinking not supported with streaming tools
        };

        let mut request = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.config.api_key)
            .header("Content-Type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .json(&req_body);

        if self.config.timeout_seconds > 0 {
            request = request.timeout(std::time::Duration::from_secs(self.config.timeout_seconds));
        }

        if log::log_enabled!(log::Level::Trace) {
            if let Ok(json) = serde_json::to_string(&req_body) {
                log::trace!("Anthropic streaming request payload: {}", json);
            }
        }

        log::debug!("Anthropic request: POST /v1/messages (streaming with tools)");
        let response = request.send().await?;
        log::debug!("Anthropic HTTP status: {}", response.status());

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("Anthropic API returned error status: {status}"),
                raw_response: error_text,
            });
        }

        Ok(create_anthropic_tool_stream(response))
    }
}

/// Creates an SSE stream that parses Anthropic tool use events into StreamChunk.
pub(crate) fn create_anthropic_tool_stream(
    response: reqwest::Response,
) -> std::pin::Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>> {
    use futures::stream::StreamExt;

    let stream = response
        .bytes_stream()
        .scan(
            (String::new(), Vec::new(), HashMap::new()),
            move |(buffer, utf8_buffer, tool_states), chunk| {
                let result = match chunk {
                    Ok(bytes) => {
                        utf8_buffer.extend_from_slice(&bytes);

                        match String::from_utf8(utf8_buffer.clone()) {
                            Ok(text) => {
                                buffer.push_str(&text);
                                utf8_buffer.clear();
                            }
                            Err(e) => {
                                let valid_up_to = e.utf8_error().valid_up_to();
                                if valid_up_to > 0 {
                                    let valid =
                                        String::from_utf8_lossy(&utf8_buffer[..valid_up_to]);
                                    buffer.push_str(&valid);
                                    utf8_buffer.drain(..valid_up_to);
                                }
                            }
                        }

                        let mut results = Vec::new();

                        while let Some(pos) = buffer.find("\n\n") {
                            let event = buffer[..pos + 2].to_string();
                            buffer.drain(..pos + 2);

                            match parse_anthropic_sse_chunk_with_tools(&event, tool_states) {
                                Ok(Some(chunk)) => results.push(Ok(chunk)),
                                Ok(None) => {}
                                Err(e) => results.push(Err(e)),
                            }
                        }

                        Some(results)
                    }
                    Err(e) => Some(vec![Err(LLMError::HttpError(e.to_string()))]),
                };

                async move { result }
            },
        )
        .flat_map(futures::stream::iter);

    Box::pin(stream)
}

#[async_trait]
impl CompletionProvider for Anthropic {
    /// Sends a completion request to Anthropic's API.
    ///
    /// Converts the completion request into a chat message format.
    async fn complete(&self, _req: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        unimplemented!()
    }
}

#[async_trait]
impl EmbeddingProvider for Anthropic {
    async fn embed(&self, _text: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        Err(LLMError::ProviderError(
            "Embedding not supported".to_string(),
        ))
    }
}

#[async_trait]
impl SpeechToTextProvider for Anthropic {
    async fn transcribe(&self, _audio: Vec<u8>) -> Result<String, LLMError> {
        Err(LLMError::ProviderError(
            "Speech to text not supported".to_string(),
        ))
    }
}

#[async_trait]
impl TextToSpeechProvider for Anthropic {}

#[derive(Clone, Debug, Deserialize)]
pub struct AnthropicModelListResponse {
    data: Vec<AnthropicModelEntry>,
}

impl ModelListResponse for AnthropicModelListResponse {
    fn get_models(&self) -> Vec<String> {
        self.data.iter().map(|m| m.id.clone()).collect()
    }

    fn get_models_raw(&self) -> Vec<Box<dyn ModelListRawEntry>> {
        self.data
            .iter()
            .map(|e| Box::new(e.clone()) as Box<dyn ModelListRawEntry>)
            .collect()
    }

    fn get_backend(&self) -> LLMBackend {
        LLMBackend::Anthropic
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct AnthropicModelEntry {
    created_at: DateTime<Utc>,
    id: String,
    #[serde(flatten)]
    extra: Value,
}

impl ModelListRawEntry for AnthropicModelEntry {
    fn get_id(&self) -> String {
        self.id.clone()
    }

    fn get_created_at(&self) -> DateTime<Utc> {
        self.created_at
    }

    fn get_raw(&self) -> Value {
        self.extra.clone()
    }
}

#[async_trait]
impl ModelsProvider for Anthropic {
    async fn list_models(
        &self,
        _request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, LLMError> {
        let resp = self
            .client
            .get("https://api.anthropic.com/v1/models")
            .header("x-api-key", &self.config.api_key)
            .header("Content-Type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .send()
            .await?;

        let result: AnthropicModelListResponse = resp.json().await?;

        Ok(Box::new(result))
    }
}

impl crate::LLMProvider for Anthropic {
    fn tools(&self) -> Option<&[Tool]> {
        self.config.tools.as_deref()
    }
}

/// Parses a Server-Sent Events (SSE) chunk from Anthropic's streaming API.
///
/// # Arguments
///
/// * `chunk` - The raw SSE chunk text
///
/// # Returns
///
/// * `Ok(Some(String))` - Content token if found
/// * `Ok(None)` - If chunk should be skipped (e.g., ping, done signal)
/// * `Err(LLMError)` - If parsing fails
pub(crate) fn parse_anthropic_sse_chunk(chunk: &str) -> Result<Option<String>, LLMError> {
    for line in chunk.lines() {
        let line = line.trim();
        if let Some(data) = line.strip_prefix("data: ") {
            match serde_json::from_str::<AnthropicStreamResponse>(data) {
                Ok(response) => {
                    if response.response_type == "content_block_delta" {
                        if let Some(delta) = response.delta {
                            if let Some(text) = delta.text {
                                return Ok(Some(text));
                            }
                        }
                    }
                    return Ok(None);
                }
                Err(_) => continue,
            }
        }
    }
    Ok(None)
}

/// State for tracking tool use blocks during streaming
#[derive(Debug, Default)]
pub(crate) struct ToolUseState {
    /// Tool ID
    id: String,
    /// Tool name
    name: String,
    /// Accumulated JSON input
    json_buffer: String,
}

/// Parses Anthropic SSE chunks with tool use support.
///
/// This parser handles all Anthropic streaming event types including:
/// - `content_block_start` with `type: "text"` or `type: "tool_use"`
/// - `content_block_delta` with `type: "text_delta"` or `type: "input_json_delta"`
/// - `content_block_stop`
/// - `message_delta` with `stop_reason`
///
/// # Arguments
///
/// * `chunk` - The raw SSE chunk text
/// * `tool_states` - Mutable reference to a map tracking tool use state by index
///
/// # Returns
///
/// * `Ok(Some(StreamChunk))` - A stream chunk if one was parsed
/// * `Ok(None)` - If chunk should be skipped
/// * `Err(LLMError)` - If parsing fails
pub(crate) fn parse_anthropic_sse_chunk_with_tools(
    chunk: &str,
    tool_states: &mut HashMap<usize, ToolUseState>,
) -> Result<Option<StreamChunk>, LLMError> {
    for line in chunk.lines() {
        let line = line.trim();
        if let Some(data) = line.strip_prefix("data: ") {
            match serde_json::from_str::<AnthropicStreamResponse>(data) {
                Ok(response) => {
                    match response.response_type.as_str() {
                        "content_block_start" => {
                            if let (Some(index), Some(content_block)) =
                                (response.index, response.content_block)
                            {
                                if content_block.block_type == "tool_use" {
                                    let id = content_block.id.unwrap_or_default();
                                    let name = content_block.name.unwrap_or_default();

                                    // Store state for this tool use block
                                    tool_states.insert(
                                        index,
                                        ToolUseState {
                                            id: id.clone(),
                                            name: name.clone(),
                                            json_buffer: String::new(),
                                        },
                                    );

                                    return Ok(Some(StreamChunk::ToolUseStart { index, id, name }));
                                }
                                // For text blocks, we just wait for content_block_delta
                            }
                        }
                        "content_block_delta" => {
                            if let (Some(index), Some(delta)) = (response.index, response.delta) {
                                // Check delta type
                                match delta.delta_type.as_deref() {
                                    Some("text_delta") => {
                                        if let Some(text) = delta.text {
                                            return Ok(Some(StreamChunk::Text(text)));
                                        }
                                    }
                                    Some("input_json_delta") => {
                                        if let Some(partial_json) = delta.partial_json {
                                            // Accumulate JSON in state
                                            if let Some(state) = tool_states.get_mut(&index) {
                                                state.json_buffer.push_str(&partial_json);
                                            }
                                            return Ok(Some(StreamChunk::ToolUseInputDelta {
                                                index,
                                                partial_json,
                                            }));
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                        "content_block_stop" => {
                            if let Some(index) = response.index {
                                // If we have tool state for this index, emit ToolUseComplete
                                if let Some(state) = tool_states.remove(&index) {
                                    // Anthropic API requires tool_use.input to be a JSON object.
                                    // If no input deltas were received (tool has no parameters),
                                    // default to empty object "{}" instead of empty string "".
                                    let arguments = if state.json_buffer.is_empty() {
                                        "{}".to_string()
                                    } else {
                                        state.json_buffer
                                    };
                                    let tool_call = ToolCall {
                                        id: state.id,
                                        call_type: "function".to_string(),
                                        function: FunctionCall {
                                            name: state.name,
                                            arguments,
                                        },
                                    };
                                    return Ok(Some(StreamChunk::ToolUseComplete {
                                        index,
                                        tool_call,
                                    }));
                                }
                            }
                        }
                        "message_delta" => {
                            if let Some(delta) = response.delta {
                                if let Some(stop_reason) = delta.stop_reason {
                                    return Ok(Some(StreamChunk::Done { stop_reason }));
                                }
                            }
                        }
                        _ => {}
                    }
                    return Ok(None);
                }
                Err(_) => continue,
            }
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounded_output_provider() -> Anthropic {
        Anthropic::new(
            "test-key", None, None, None, None, None, None, None, None, None, None, None,
        ).with_structured_output(crate::chat::StructuredOutputFormat {
            name: "draft".into(), description: None, strict: Some(true),
            schema: Some(serde_json::json!({
                "type": "object", "properties": {
                    "steps": {"type": "array", "maxItems": 1, "items": {"type": "string", "maxLength": 3}}
                }, "required": ["steps"], "additionalProperties": false
            })),
        }).unwrap()
    }

    fn completion_response(text: &str) -> reqwest::Response {
        http::Response::builder()
            .status(http::StatusCode::OK)
            .body(serde_json::json!({"content": [{"type": "text", "text": text}]}).to_string())
            .unwrap()
            .into()
    }

    #[tokio::test]
    async fn adapted_completion_is_validated_before_returning_to_caller() {
        let provider = bounded_output_provider();
        let valid = provider
            .parse_completion_response(completion_response(r#"{"steps":["ok"]}"#))
            .await
            .unwrap();
        assert_eq!(valid.text().unwrap(), r#"{"steps":["ok"]}"#);
        for invalid in [r#"{"steps":["too long"]}"#, r#"{"steps":["a","b"]}"#, "{"] {
            assert!(matches!(
                provider
                    .parse_completion_response(completion_response(invalid))
                    .await,
                Err(LLMError::ProviderError(_))
            ));
        }
    }

    #[tokio::test]
    async fn non_streaming_rejection_preserves_anthropic_diagnosis_and_request_id() {
        let body = serde_json::json!({"type": "error", "error": {
            "type": "invalid_request_error",
            "message": "output_config.format.schema: For 'array' type, property 'maxItems' is not supported"
        }}).to_string();
        let response = http::Response::builder()
            .status(http::StatusCode::BAD_REQUEST)
            .header("request-id", "req_test_schema")
            .body(body)
            .unwrap()
            .into();
        let error = bounded_output_provider()
            .parse_completion_response(response)
            .await
            .unwrap_err();
        assert!(matches!(error, LLMError::HttpError(_)));
        let message = error.to_string();
        for detail in [
            "400 Bad Request",
            "req_test_schema",
            "invalid_request_error",
            "maxItems",
        ] {
            assert!(message.contains(detail), "{message}");
        }
    }

    #[tokio::test]
    async fn adapted_output_allows_intermediate_tool_calls() {
        let body = serde_json::json!({"content": [{
            "type": "tool_use", "id": "tool_test", "name": "lookup", "input": {"query": "customer"}
        }]})
        .to_string();
        let response = http::Response::builder()
            .status(http::StatusCode::OK)
            .body(body)
            .unwrap()
            .into();
        let parsed = bounded_output_provider()
            .parse_completion_response(response)
            .await
            .unwrap();
        assert_eq!(parsed.tool_calls().unwrap()[0].function.name, "lookup");
    }

    #[test]
    fn replacing_adapted_schema_clears_buffered_validation() {
        let provider = bounded_output_provider();
        assert!(provider.config.output_validator.is_some());
        let original = provider.clone();
        let updated = provider.with_structured_output(crate::chat::StructuredOutputFormat {
            name: "plain".into(), description: None, strict: Some(true),
            schema: Some(serde_json::json!({"type": "object", "properties": {}, "additionalProperties": false})),
        }).unwrap();
        assert!(updated.config.output_validator.is_none());
        assert!(original.config.output_validator.is_some());
    }

    #[test]
    fn structured_output_preserves_nested_required_fields_on_wire() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"steps": {"type": "array", "items": {
                "type": "object",
                "properties": {"description": {"type": "string"}},
                "required": ["description"],
                "additionalProperties": false
            }}},
            "required": ["steps"],
            "additionalProperties": false
        });
        let provider = Anthropic::new(
            "test-key", None, None, None, None, None, None, None, None, None, None, None,
        )
        .with_structured_output(crate::chat::StructuredOutputFormat {
            name: "draft".to_string(),
            description: None,
            schema: Some(schema.clone()),
            strict: Some(true),
        })
        .unwrap();
        for streaming in [false, true] {
            let request = AnthropicCompleteRequest {
                messages: vec![],
                model: &provider.config.model,
                max_tokens: Some(provider.config.max_tokens),
                temperature: None,
                system: None,
                stream: Some(streaming),
                top_p: None,
                top_k: None,
                tools: None,
                tool_choice: None,
                thinking: None,
                output_config: provider.config.output_config.as_ref(),
            };
            let wire = serde_json::to_value(request).unwrap();
            assert_eq!(wire["output_config"]["format"]["type"], "json_schema");
            assert_eq!(wire["output_config"]["format"]["schema"], schema);
            assert!(wire.get("response_format").is_none());
        }
    }

    #[test]
    fn builder_rejects_schema_less_structured_output_instead_of_dropping_it() {
        let result = crate::builder::LLMBuilder::new()
            .backend(crate::builder::LLMBackend::Anthropic)
            .api_key("test-key")
            .schema(crate::chat::StructuredOutputFormat {
                name: "json_object".to_string(),
                description: None,
                schema: None,
                strict: Some(true),
            })
            .build();
        assert!(matches!(result, Err(LLMError::InvalidRequest(message))
            if message.contains("explicit JSON schema")));
    }

    fn free_form_planner_schema() -> serde_json::Value {
        // Mirrors the planner schema shape from inxm-ai/app-tgi-core#72: an
        // open `value: {}` and a free-form `arguments` object.
        serde_json::json!({
            "type": "object",
            "properties": {
                "value": {},
                "arguments": {"type": "object"}
            },
            "required": ["value", "arguments"],
            "additionalProperties": false
        })
    }

    #[test]
    fn schema_anthropic_cannot_express_falls_back_to_prompt_based_output() {
        let provider = Anthropic::new(
            "test-key", None, None, None, None, None, None, None, None, None, None, None,
        )
        .with_structured_output(crate::chat::StructuredOutputFormat {
            name: "planner".into(),
            description: None,
            strict: Some(true),
            schema: Some(free_form_planner_schema()),
        })
        .unwrap();
        assert!(provider.config.output_config.is_none());
        assert!(provider.config.output_validator.is_some());
        let instruction = provider
            .config
            .output_schema_instruction
            .as_ref()
            .expect("fallback should carry a schema instruction");
        assert!(instruction.contains("\"arguments\""));
    }

    #[tokio::test]
    async fn fallback_output_is_still_validated_locally_against_the_original_schema() {
        let provider = Anthropic::new(
            "test-key", None, None, None, None, None, None, None, None, None, None, None,
        )
        .with_structured_output(crate::chat::StructuredOutputFormat {
            name: "planner".into(),
            description: None,
            strict: Some(true),
            schema: Some(free_form_planner_schema()),
        })
        .unwrap();
        let valid = provider
            .parse_completion_response(completion_response(
                r#"{"value": 3, "arguments": {"anything": "goes"}}"#,
            ))
            .await
            .unwrap();
        assert_eq!(
            valid.text().unwrap(),
            r#"{"value": 3, "arguments": {"anything": "goes"}}"#
        );
        let invalid = provider
            .parse_completion_response(completion_response(r#"{"value": 3}"#))
            .await;
        assert!(matches!(invalid, Err(LLMError::ProviderError(_))));
    }

    #[test]
    fn fallback_schema_instruction_is_appended_to_string_system_prompt() {
        let provider = Anthropic::new(
            "test-key", None, None, None, None,
            Some(SystemPrompt::String("You are the planner.".into())),
            None, None, None, None, None, None,
        )
        .with_structured_output(crate::chat::StructuredOutputFormat {
            name: "planner".into(),
            description: None,
            strict: Some(true),
            schema: Some(free_form_planner_schema()),
        })
        .unwrap();
        let instruction = provider.config.output_schema_instruction.as_deref().unwrap();
        let owned = Anthropic::system_with_schema_instruction(&provider.config.system, instruction);
        match owned {
            OwnedSystemPrompt::String(s) => {
                assert!(s.starts_with("You are the planner."));
                assert!(s.contains("JSON Schema"));
            }
            OwnedSystemPrompt::Messages(_) => panic!("expected a string system prompt"),
        }
    }

    #[test]
    fn test_parse_stream_text_delta() {
        let chunk = r#"event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}

"#;
        let mut tool_states = HashMap::new();
        let result = parse_anthropic_sse_chunk_with_tools(chunk, &mut tool_states).unwrap();

        match result {
            Some(StreamChunk::Text(text)) => assert_eq!(text, "Hello"),
            _ => panic!("Expected Text chunk, got {:?}", result),
        }
    }

    #[test]
    fn test_parse_stream_tool_use_start() {
        let chunk = r#"event: content_block_start
data: {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_01ABC", "name": "get_weather", "input": {}}}

"#;
        let mut tool_states = HashMap::new();
        let result = parse_anthropic_sse_chunk_with_tools(chunk, &mut tool_states).unwrap();

        match result {
            Some(StreamChunk::ToolUseStart { index, id, name }) => {
                assert_eq!(index, 1);
                assert_eq!(id, "toolu_01ABC");
                assert_eq!(name, "get_weather");
            }
            _ => panic!("Expected ToolUseStart chunk, got {:?}", result),
        }

        // Verify state was stored
        assert!(tool_states.contains_key(&1));
        assert_eq!(tool_states[&1].id, "toolu_01ABC");
        assert_eq!(tool_states[&1].name, "get_weather");
    }

    #[test]
    fn test_parse_stream_tool_use_input_delta() {
        let chunk = r#"event: content_block_delta
data: {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"location\":"}}

"#;
        let mut tool_states = HashMap::new();
        // Pre-populate state as if tool_use_start was already processed
        tool_states.insert(
            1,
            ToolUseState {
                id: "toolu_01ABC".to_string(),
                name: "get_weather".to_string(),
                json_buffer: String::new(),
            },
        );

        let result = parse_anthropic_sse_chunk_with_tools(chunk, &mut tool_states).unwrap();

        match result {
            Some(StreamChunk::ToolUseInputDelta {
                index,
                partial_json,
            }) => {
                assert_eq!(index, 1);
                assert_eq!(partial_json, "{\"location\":");
            }
            _ => panic!("Expected ToolUseInputDelta chunk, got {:?}", result),
        }

        // Verify JSON was accumulated
        assert_eq!(tool_states[&1].json_buffer, "{\"location\":");
    }

    #[test]
    fn test_parse_stream_tool_use_complete() {
        let chunk = r#"event: content_block_stop
data: {"type": "content_block_stop", "index": 1}

"#;
        let mut tool_states = HashMap::new();
        // Pre-populate state with accumulated JSON
        tool_states.insert(
            1,
            ToolUseState {
                id: "toolu_01ABC".to_string(),
                name: "get_weather".to_string(),
                json_buffer: r#"{"location": "Paris"}"#.to_string(),
            },
        );

        let result = parse_anthropic_sse_chunk_with_tools(chunk, &mut tool_states).unwrap();

        match result {
            Some(StreamChunk::ToolUseComplete { index, tool_call }) => {
                assert_eq!(index, 1);
                assert_eq!(tool_call.id, "toolu_01ABC");
                assert_eq!(tool_call.function.name, "get_weather");
                assert_eq!(tool_call.function.arguments, r#"{"location": "Paris"}"#);
            }
            _ => panic!("Expected ToolUseComplete chunk, got {:?}", result),
        }

        // Verify state was removed
        assert!(!tool_states.contains_key(&1));
    }

    #[test]
    fn test_parse_stream_tool_use_complete_empty_arguments() {
        // Regression test: tools with no parameters should return "{}" not ""
        // Empty string arguments cause Anthropic API to reject with:
        // "tool_use.input: Input should be a valid dictionary"
        let chunk = r#"event: content_block_stop
data: {"type": "content_block_stop", "index": 1}

"#;
        let mut tool_states = HashMap::new();
        // Pre-populate state with EMPTY json_buffer (no input_json_delta events received)
        tool_states.insert(
            1,
            ToolUseState {
                id: "toolu_01XYZ".to_string(),
                name: "get_current_time".to_string(),
                json_buffer: String::new(), // Empty - tool has no parameters
            },
        );

        let result = parse_anthropic_sse_chunk_with_tools(chunk, &mut tool_states).unwrap();

        match result {
            Some(StreamChunk::ToolUseComplete { index, tool_call }) => {
                assert_eq!(index, 1);
                assert_eq!(tool_call.id, "toolu_01XYZ");
                assert_eq!(tool_call.function.name, "get_current_time");
                // CRITICAL: arguments must be "{}" not "" for Anthropic API compatibility
                assert_eq!(
                    tool_call.function.arguments, "{}",
                    "Empty arguments should default to '{{}}' not empty string"
                );
            }
            _ => panic!("Expected ToolUseComplete chunk, got {:?}", result),
        }

        // Verify state was removed
        assert!(!tool_states.contains_key(&1));
    }

    #[test]
    fn test_parse_stream_done_tool_use() {
        let chunk = r#"event: message_delta
data: {"type": "message_delta", "delta": {"stop_reason": "tool_use"}}

"#;
        let mut tool_states = HashMap::new();
        let result = parse_anthropic_sse_chunk_with_tools(chunk, &mut tool_states).unwrap();

        match result {
            Some(StreamChunk::Done { stop_reason }) => {
                assert_eq!(stop_reason, "tool_use");
            }
            _ => panic!("Expected Done chunk, got {:?}", result),
        }
    }

    #[test]
    fn test_parse_stream_done_end_turn() {
        let chunk = r#"event: message_delta
data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}

"#;
        let mut tool_states = HashMap::new();
        let result = parse_anthropic_sse_chunk_with_tools(chunk, &mut tool_states).unwrap();

        match result {
            Some(StreamChunk::Done { stop_reason }) => {
                assert_eq!(stop_reason, "end_turn");
            }
            _ => panic!("Expected Done chunk, got {:?}", result),
        }
    }

    #[test]
    fn test_parse_stream_full_tool_use_sequence() {
        let mut tool_states = HashMap::new();

        // 1. Tool use start
        let start_chunk = r#"event: content_block_start
data: {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_01ABC", "name": "get_weather", "input": {}}}

"#;
        let result = parse_anthropic_sse_chunk_with_tools(start_chunk, &mut tool_states).unwrap();
        assert!(matches!(result, Some(StreamChunk::ToolUseStart { .. })));

        // 2. Input JSON deltas
        let delta1 = r#"event: content_block_delta
data: {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"loc"}}

"#;
        let _ = parse_anthropic_sse_chunk_with_tools(delta1, &mut tool_states).unwrap();

        let delta2 = r#"event: content_block_delta
data: {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "ation\": \"Paris\"}"}}

"#;
        let _ = parse_anthropic_sse_chunk_with_tools(delta2, &mut tool_states).unwrap();

        // Verify accumulated JSON
        assert_eq!(tool_states[&1].json_buffer, "{\"location\": \"Paris\"}");

        // 3. Content block stop
        let stop_chunk = r#"event: content_block_stop
data: {"type": "content_block_stop", "index": 1}

"#;
        let result = parse_anthropic_sse_chunk_with_tools(stop_chunk, &mut tool_states).unwrap();

        match result {
            Some(StreamChunk::ToolUseComplete { tool_call, .. }) => {
                assert_eq!(tool_call.function.arguments, "{\"location\": \"Paris\"}");
            }
            _ => panic!("Expected ToolUseComplete"),
        }

        // 4. Message delta with stop reason
        let done_chunk = r#"event: message_delta
data: {"type": "message_delta", "delta": {"stop_reason": "tool_use"}}

"#;
        let result = parse_anthropic_sse_chunk_with_tools(done_chunk, &mut tool_states).unwrap();
        assert!(matches!(
            result,
            Some(StreamChunk::Done {
                stop_reason
            }) if stop_reason == "tool_use"
        ));
    }

    #[test]
    fn test_parse_stream_mixed_text_and_tool() {
        let mut tool_states = HashMap::new();

        // Text delta first
        let text_chunk = r#"event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "I'll check the weather"}}

"#;
        let result = parse_anthropic_sse_chunk_with_tools(text_chunk, &mut tool_states).unwrap();
        assert!(matches!(result, Some(StreamChunk::Text(t)) if t == "I'll check the weather"));

        // Then tool use
        let tool_start = r#"event: content_block_start
data: {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_01XYZ", "name": "weather", "input": {}}}

"#;
        let result = parse_anthropic_sse_chunk_with_tools(tool_start, &mut tool_states).unwrap();
        assert!(
            matches!(result, Some(StreamChunk::ToolUseStart { name, .. }) if name == "weather")
        );
    }

    #[test]
    fn test_parse_stream_ignores_message_start() {
        let chunk = r#"event: message_start
data: {"type": "message_start", "message": {"id": "msg_123", "type": "message", "role": "assistant"}}

"#;
        let mut tool_states = HashMap::new();
        let result = parse_anthropic_sse_chunk_with_tools(chunk, &mut tool_states).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_stream_ignores_ping() {
        let chunk = r#"event: ping
data: {"type": "ping"}

"#;
        let mut tool_states = HashMap::new();
        let result = parse_anthropic_sse_chunk_with_tools(chunk, &mut tool_states).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_anthropic_tool_serialization_without_cache_control() {
        let schema = serde_json::json!({"type": "object", "properties": {}});
        let tool = AnthropicTool {
            name: "my_tool",
            description: "desc",
            schema: &schema,
            cache_control: None,
        };

        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(json.get("name").unwrap(), "my_tool");
        assert!(json.get("cache_control").is_none());
    }

    #[test]
    fn test_anthropic_tool_serialization_with_cache_control() {
        let schema = serde_json::json!({"type": "object", "properties": {}});
        let cc = serde_json::json!({"type": "ephemeral"});
        let tool = AnthropicTool {
            name: "my_tool",
            description: "desc",
            schema: &schema,
            cache_control: Some(&cc),
        };

        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(
            json.get("cache_control").unwrap(),
            &serde_json::json!({"type": "ephemeral"})
        );
    }

    #[test]
    fn test_prepare_tools_maps_cache_control() {
        let tools = vec![Tool {
            tool_type: "function".to_string(),
            function: crate::chat::FunctionTool {
                name: "get_weather".to_string(),
                description: "Get weather".to_string(),
                parameters: serde_json::json!({"type": "object", "properties": {}}),
            },
            cache_control: Some(serde_json::json!({"type": "ephemeral"})),
        }];

        let (anthropic_tools, _) = Anthropic::prepare_tools_and_choice(Some(&tools), None, &None);

        let anthropic_tools = anthropic_tools.expect("tools should be present");
        assert_eq!(anthropic_tools.len(), 1);
        assert_eq!(
            anthropic_tools[0].cache_control,
            Some(&serde_json::json!({"type": "ephemeral"}))
        );
    }

    #[test]
    fn test_prepare_tools_none_cache_control() {
        let tools = vec![Tool {
            tool_type: "function".to_string(),
            function: crate::chat::FunctionTool {
                name: "get_weather".to_string(),
                description: "Get weather".to_string(),
                parameters: serde_json::json!({"type": "object", "properties": {}}),
            },
            cache_control: None,
        }];

        let (anthropic_tools, _) = Anthropic::prepare_tools_and_choice(Some(&tools), None, &None);

        let anthropic_tools = anthropic_tools.expect("tools should be present");
        assert!(anthropic_tools[0].cache_control.is_none());
    }

    #[test]
    fn request_omits_temperature_top_p_top_k_when_unset() {
        let req_body = AnthropicCompleteRequest {
            messages: vec![],
            model: "claude-sonnet-5",
            max_tokens: Some(64),
            temperature: None,
            system: None,
            stream: Some(false),
            top_p: None,
            top_k: None,
            tools: None,
            tool_choice: None,
            thinking: None,
            output_config: None,
        };

        let json = serde_json::to_value(&req_body).unwrap();
        assert!(json.get("temperature").is_none());
        assert!(json.get("top_p").is_none());
        assert!(json.get("top_k").is_none());
        assert!(json.get("output_config").is_none());
    }

    #[test]
    fn request_includes_temperature_top_p_top_k_when_explicitly_set() {
        let req_body = AnthropicCompleteRequest {
            messages: vec![],
            model: "claude-haiku-4-5-20251001",
            max_tokens: Some(64),
            temperature: Some(0.0),
            system: None,
            stream: Some(false),
            top_p: Some(0.5),
            top_k: Some(40),
            tools: None,
            tool_choice: None,
            thinking: None,
            output_config: None,
        };

        let json = serde_json::to_value(&req_body).unwrap();
        assert_eq!(json.get("temperature").unwrap(), 0.0);
        assert_eq!(json.get("top_p").unwrap(), 0.5);
        assert_eq!(json.get("top_k").unwrap(), 40);
    }
}
