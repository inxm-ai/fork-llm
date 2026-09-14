use serde_json::Value;

use crate::error::LLMError;

const DESCRIPTION: &str = "description";
const UNSUPPORTED_BOUNDS: &[&str] = &["maxLength", "maxItems"];
const SCHEMA_MAPS: &[&str] = &[
    "properties",
    "patternProperties",
    "$defs",
    "definitions",
    "dependentSchemas",
];
const SCHEMA_ARRAYS: &[&str] = &["allOf", "anyOf", "oneOf", "prefixItems"];
const CHILD_SCHEMAS: &[&str] = &[
    "items",
    "additionalItems",
    "additionalProperties",
    "contains",
    "not",
    "if",
    "then",
    "else",
    "propertyNames",
    "unevaluatedItems",
    "unevaluatedProperties",
];

/// Adapt schema positions only: property names, enum values, and examples are data.
/// The original schema must still be enforced locally after generation.
pub(super) fn adapt_schema(schema: &Value) -> Value {
    let Value::Object(fields) = schema else {
        return schema.clone();
    };
    let mut adapted = fields.clone();
    let constraints = UNSUPPORTED_BOUNDS
        .iter()
        .filter_map(|key| adapted.remove(*key).map(|bound| format!("{key} = {bound}")))
        .collect::<Vec<_>>();
    if !constraints.is_empty() {
        let description = fields
            .get(DESCRIPTION)
            .and_then(Value::as_str)
            .unwrap_or_default();
        adapted.insert(
            DESCRIPTION.into(),
            Value::String(
                format!(
                    "{description} Required constraints: {}.",
                    constraints.join(", ")
                )
                .trim()
                .into(),
            ),
        );
    }
    for key in SCHEMA_MAPS {
        if let Some(Value::Object(children)) = fields.get(*key) {
            adapted.insert(
                (*key).into(),
                Value::Object(
                    children
                        .iter()
                        .map(|(name, child)| (name.clone(), adapt_schema(child)))
                        .collect(),
                ),
            );
        }
    }
    for key in SCHEMA_ARRAYS {
        if let Some(Value::Array(children)) = fields.get(*key) {
            adapted.insert(
                (*key).into(),
                Value::Array(children.iter().map(adapt_schema).collect()),
            );
        }
    }
    for key in CHILD_SCHEMAS {
        if let Some(child) = fields.get(*key) {
            let child = match child {
                Value::Array(children) => Value::Array(children.iter().map(adapt_schema).collect()),
                _ => adapt_schema(child),
            };
            adapted.insert((*key).into(), child);
        }
    }
    // Anthropic's native structured output rejects any object schema whose
    // additionalProperties isn't explicitly false (400: "'additionalProperties'
    // must be explicitly set to false"), regardless of what the caller's
    // original schema declared. Force it after the recursive walk above so it
    // overrides whatever CHILD_SCHEMAS just adapted additionalProperties to.
    let is_object = fields.get("type").and_then(Value::as_str) == Some("object")
        || fields.contains_key("properties");
    if is_object {
        adapted.insert("additionalProperties".into(), Value::Bool(false));
    }
    Value::Object(adapted)
}

pub(super) fn validate_output(
    validator: &jsonschema::JSONSchema,
    text: &str,
) -> Result<(), LLMError> {
    let value: Value = serde_json::from_str(text).map_err(|error| {
        LLMError::ProviderError(format!(
            "Anthropic structured output is not valid JSON: {error}"
        ))
    })?;
    if let Err(mut errors) = validator.validate(&value) {
        if let Some(error) = errors.next() {
            // Report paths, not the potentially sensitive generated value.
            return Err(LLMError::ProviderError(format!(
                "Anthropic structured output violates the original schema at {} (schema {})",
                error.instance_path, error.schema_path
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn adapts_draft_bounds_without_losing_structure_or_original_validation() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "maxLength": 3, "description": "Draft name"},
                "steps": {"type": "array", "minItems": 1, "maxItems": 1, "items": {
                    "type": "object", "properties": {"name": {"type": "string", "maxLength": 3}},
                    "required": ["name"], "additionalProperties": false
                }}
            }, "required": ["name", "steps"], "additionalProperties": false
        });
        let adapted = adapt_schema(&schema);
        assert!(adapted["properties"]["name"].get("maxLength").is_none());
        assert!(adapted["properties"]["steps"].get("maxItems").is_none());
        assert!(
            adapted["properties"]["steps"]["items"]["properties"]["name"]
                .get("maxLength")
                .is_none()
        );
        assert_eq!(adapted["properties"]["steps"]["minItems"], 1);
        assert_eq!(adapted["required"], schema["required"]);
        assert_eq!(adapted["additionalProperties"], false);
        assert_eq!(
            adapted["properties"]["name"]["description"],
            "Draft name Required constraints: maxLength = 3."
        );
        let validator = jsonschema::JSONSchema::compile(&schema).unwrap();
        assert!(
            validate_output(&validator, r#"{"name":"猫猫猫","steps":[{"name":"ok"}]}"#).is_ok()
        );
        for invalid in [
            r#"{"name":"long","steps":[{"name":"ok"}]}"#,
            r#"{"name":"ok","steps":[{"name":"ok"},{"name":"ok"}]}"#,
            r#"{"name":"ok","steps":[{"name":"long"}]}"#,
            r#"{"name":"ok"}"#,
            "not JSON",
        ] {
            assert!(validate_output(&validator, invalid).is_err(), "{invalid}");
        }
        assert_eq!(schema["properties"]["name"]["maxLength"], 3);
    }

    #[test]
    fn follows_definitions_and_unions_but_preserves_property_names_and_literal_data() {
        let schema = json!({
            "$defs": {"label": {"type": "string", "maxLength": 3}},
            "type": "object", "properties": {
                "maxItems": {"anyOf": [{"$ref": "#/$defs/label"}, {"type": "null"}]},
                "maxLength": {"type": "array", "maxItems": 1, "items": {"type": "string"}}
            },
            "examples": [{"maxLength": 17}], "const": {"maxItems": 9}
        });
        let adapted = adapt_schema(&schema);
        assert!(adapted["$defs"]["label"].get("maxLength").is_none());
        assert_eq!(
            adapted["properties"]["maxItems"],
            schema["properties"]["maxItems"]
        );
        assert!(adapted["properties"]["maxLength"].get("maxItems").is_none());
        assert_eq!(adapted["examples"], schema["examples"]);
        assert_eq!(adapted["const"], schema["const"]);
    }

    #[test]
    fn forces_additional_properties_false_on_every_object_even_when_absent_or_true() {
        let schema = json!({
            "type": "object",
            "properties": {
                "user": {
                    "properties": {"id": {"type": "string"}}
                },
                "meta": {"type": "object", "additionalProperties": true}
            }
        });
        let adapted = adapt_schema(&schema);
        assert_eq!(adapted["additionalProperties"], false);
        assert_eq!(adapted["properties"]["user"]["additionalProperties"], false);
        assert_eq!(adapted["properties"]["meta"]["additionalProperties"], false);
        // Non-object schemas are untouched.
        let string_schema = json!({"type": "string"});
        assert!(adapt_schema(&string_schema).get("additionalProperties").is_none());
    }
}
