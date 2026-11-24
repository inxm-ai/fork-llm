use super::*;
use crate::backends::openai::OpenAITool;
use crate::chat::{ChatMessage, FunctionTool, StructuredOutputFormat, Tool, ToolChoice};
use crate::providers::openai_compatible::OpenAICompatibleProviderConfig;
use reqwest::Url;
use serde_json::json;

fn base_config() -> OpenAICompatibleProviderConfig {
    OpenAICompatibleProviderConfig {
        api_key: "key".to_string(),
        base_url: Url::parse("https://api.openai.com/v1/").unwrap(),
        model: "gpt-4.1".to_string(),
        max_tokens: Some(120),
        temperature: Some(0.7),
        system: Some("system".to_string()),
        timeout_seconds: Some(30),
        top_p: Some(0.9),
        top_k: Some(40),
        tools: None,
        tool_choice: None,
        reasoning_effort: None,
        json_schema: None,
        voice: None,
        extra_body: serde_json::Map::new(),
        parallel_tool_calls: false,
        embedding_encoding_format: None,
        embedding_dimensions: None,
        normalize_response: false,
    }
}

fn sample_tool() -> Tool {
    Tool {
        tool_type: "function".to_string(),
        function: FunctionTool {
            name: "get_weather".to_string(),
            description: "Get weather".to_string(),
            parameters: json!({"type": "object"}),
        },
        cache_control: None,
    }
}

fn config_with_schema() -> OpenAICompatibleProviderConfig {
    let mut config = base_config();
    config.tool_choice = Some(ToolChoice::Auto);
    config.json_schema = Some(StructuredOutputFormat {
        name: "result".to_string(),
        description: None,
        schema: None,
        strict: Some(true),
    });
    config
}

fn assert_function_tool(tool: &OpenAITool, name: &str) {
    match tool {
        OpenAITool::Function {
            tool_type,
            name: tool_name,
            ..
        } => {
            assert_eq!(tool_type, "function");
            assert_eq!(tool_name, name);
        }
        _ => panic!("expected function tool"),
    }
}

#[test]
fn build_responses_request_maps_tools_and_text() {
    let config = config_with_schema();
    let message = ChatMessage::user().content("Hi").build();
    let tools = vec![sample_tool()];
    let params = ResponsesRequestParams {
        config: &config,
        messages: &[message],
        tools: Some(&tools),
        stream: false,
    };
    let request = build_responses_request(params).unwrap();

    assert_eq!(request.instructions, Some("system".to_string()));
    let tool = request.tools.as_ref().unwrap().first().unwrap();
    assert_function_tool(tool, "get_weather");
    assert!(request.tool_choice.is_some());
    assert_eq!(
        request.text.as_ref().unwrap().format.format_type,
        "json_schema"
    );
}

#[test]
fn structured_output_is_a_flat_text_format() {
    let mut config = base_config();
    config.json_schema = Some(StructuredOutputFormat {
        name: "plan".to_string(),
        description: Some("The plan".to_string()),
        schema: Some(serde_json::json!({
            "type": "object",
            "properties": {"steps": {"type": "array"}}
        })),
        strict: Some(true),
    });
    let message = ChatMessage::user().content("Hi").build();
    let request = build_responses_request(ResponsesRequestParams {
        config: &config,
        messages: &[message],
        tools: None,
        stream: false,
    })
    .unwrap();

    let body = serde_json::to_value(&request).unwrap();
    assert_eq!(
        body["text"]["format"],
        serde_json::json!({
            "type": "json_schema",
            "name": "plan",
            "description": "The plan",
            "schema": {
                "type": "object",
                "properties": {"steps": {"type": "array"}},
                "additionalProperties": false
            },
            "strict": true
        })
    );
}

#[test]
fn structured_output_without_a_name_still_has_one() {
    let mut config = base_config();
    config.json_schema = Some(StructuredOutputFormat {
        name: String::new(),
        description: None,
        schema: Some(serde_json::json!({"type": "object"})),
        strict: None,
    });
    let message = ChatMessage::user().content("Hi").build();
    let request = build_responses_request(ResponsesRequestParams {
        config: &config,
        messages: &[message],
        tools: None,
        stream: false,
    })
    .unwrap();

    let body = serde_json::to_value(&request).unwrap();
    assert_eq!(body["text"]["format"]["name"], "response");
    assert!(body["text"]["format"].get("json_schema").is_none());
}

#[test]
fn build_responses_request_for_input_allows_overrides() {
    let mut config = base_config();
    config.tool_choice = Some(ToolChoice::Auto);

    let params = ResponsesInputRequestParams {
        config: &config,
        input: ResponsesInput::Text("Hello".to_string()),
        tools: None,
        stream: false,
        instructions: None,
        text: None,
    };
    let request = build_responses_request_for_input(params);

    assert!(request.instructions.is_none());
    assert!(request.tool_choice.is_none());
    match request.input {
        ResponsesInput::Text(text) => assert_eq!(text, "Hello"),
        _ => panic!("expected text input"),
    }
}

fn text_format_for(schema: serde_json::Value, strict: Option<bool>) -> serde_json::Value {
    let mut config = base_config();
    config.json_schema = Some(StructuredOutputFormat {
        name: "r".to_string(),
        description: None,
        schema: Some(schema),
        strict,
    });
    let message = ChatMessage::user().content("Hi").build();
    let request = build_responses_request(ResponsesRequestParams {
        config: &config,
        messages: &[message],
        tools: None,
        stream: false,
    })
    .unwrap();
    serde_json::to_value(&request).unwrap()["text"]["format"].clone()
}

#[test]
fn a_non_object_root_schema_is_left_as_it_is() {
    for schema in [
        json!({"type": "string"}),
        json!({"type": "array", "items": {"type": "string"}}),
    ] {
        let format = text_format_for(schema.clone(), Some(true));
        assert_eq!(format["schema"], schema);
    }
}

#[test]
fn strict_schemas_close_every_object_inside() {
    let format = text_format_for(
        json!({
            "type": "object",
            "properties": {
                "steps": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}}}},
                "owner": {"anyOf": [{"type": "object", "properties": {"name": {"type": "string"}}}, {"type": "null"}]},
                "open": {"type": "object", "additionalProperties": true}
            },
            "$defs": {"ref": {"type": "object", "properties": {"x": {"type": "number"}}}}
        }),
        Some(true),
    );
    let schema = &format["schema"];
    assert_eq!(schema["additionalProperties"], false);
    assert_eq!(
        schema["properties"]["steps"]["items"]["additionalProperties"],
        false
    );
    assert_eq!(
        schema["properties"]["owner"]["anyOf"][0]["additionalProperties"],
        false
    );
    assert!(schema["properties"]["owner"]["anyOf"][1]
        .get("additionalProperties")
        .is_none());
    assert_eq!(schema["properties"]["open"]["additionalProperties"], true);
    assert_eq!(schema["$defs"]["ref"]["additionalProperties"], false);
    assert!(schema["properties"]["steps"]
        .get("additionalProperties")
        .is_none());
}

#[test]
fn non_strict_schemas_only_close_the_root() {
    let format = text_format_for(
        json!({"type": "object", "properties": {"inner": {"type": "object", "properties": {}}}}),
        None,
    );
    assert_eq!(format["schema"]["additionalProperties"], false);
    assert!(format["schema"]["properties"]["inner"]
        .get("additionalProperties")
        .is_none());
}
