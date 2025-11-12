package contextwindow

import (
	"github.com/anthropics/anthropic-sdk-go"
	"github.com/openai/openai-go/v2"
	"google.golang.org/genai"
)

type ParameterType string

const (
	ParameterTypeString  ParameterType = "string"
	ParameterTypeNumber  ParameterType = "number"
	ParameterTypeBoolean ParameterType = "boolean"
	ParameterTypeArray   ParameterType = "array"
	ParameterTypeObject  ParameterType = "object"
)

type Parameter struct {
	Name        string
	Type        ParameterType
	Description string
	Required    bool
	Items       *Parameter
	Properties  map[string]*Parameter
}

type ToolBuilder struct {
	name        string
	description string
	parameters  []*Parameter
}

func NewTool(name, description string) *ToolBuilder {
	return &ToolBuilder{
		name:        name,
		description: description,
		parameters:  make([]*Parameter, 0),
	}
}

func (tb *ToolBuilder) AddStringParameter(name, description string, required bool) *ToolBuilder {
	tb.parameters = append(tb.parameters, &Parameter{
		Name:        name,
		Type:        ParameterTypeString,
		Description: description,
		Required:    required,
	})
	return tb
}

func (tb *ToolBuilder) AddNumberParameter(name, description string, required bool) *ToolBuilder {
	tb.parameters = append(tb.parameters, &Parameter{
		Name:        name,
		Type:        ParameterTypeNumber,
		Description: description,
		Required:    required,
	})
	return tb
}

func (tb *ToolBuilder) AddBooleanParameter(name, description string, required bool) *ToolBuilder {
	tb.parameters = append(tb.parameters, &Parameter{
		Name:        name,
		Type:        ParameterTypeBoolean,
		Description: description,
		Required:    required,
	})
	return tb
}

func (tb *ToolBuilder) AddArrayParameter(name, description string, required bool, itemType ParameterType) *ToolBuilder {
	tb.parameters = append(tb.parameters, &Parameter{
		Name:        name,
		Type:        ParameterTypeArray,
		Description: description,
		Required:    required,
		Items: &Parameter{
			Type: itemType,
		},
	})
	return tb
}

func (tb *ToolBuilder) AddObjectParameter(name, description string, required bool, properties map[string]*Parameter) *ToolBuilder {
	tb.parameters = append(tb.parameters, &Parameter{
		Name:        name,
		Type:        ParameterTypeObject,
		Description: description,
		Required:    required,
		Properties:  properties,
	})
	return tb
}

func (tb *ToolBuilder) ToOpenAI() openai.FunctionDefinitionParam {
	properties := make(map[string]any)
	required := make([]string, 0)

	for _, param := range tb.parameters {
		properties[param.Name] = tb.parameterToOpenAISchema(param)
		if param.Required {
			required = append(required, param.Name)
		}
	}

	return openai.FunctionDefinitionParam{
		Name:        tb.name,
		Description: openai.String(tb.description),
		Parameters: openai.FunctionParameters{
			"type":       "object",
			"properties": properties,
			"required":   required,
		},
	}
}

func (tb *ToolBuilder) parameterToOpenAISchema(param *Parameter) map[string]any {
	schema := map[string]any{
		"type": string(param.Type),
	}

	if param.Description != "" {
		schema["description"] = param.Description
	}

	switch param.Type {
	case ParameterTypeArray:
		if param.Items != nil {
			schema["items"] = tb.parameterToOpenAISchema(param.Items)
		}
	case ParameterTypeObject:
		if param.Properties != nil {
			properties := make(map[string]any)
			required := make([]string, 0)
			for name, prop := range param.Properties {
				properties[name] = tb.parameterToOpenAISchema(prop)
				if prop.Required {
					required = append(required, name)
				}
			}
			schema["properties"] = properties
			if len(required) > 0 {
				schema["required"] = required
			}
		}
	}

	return schema
}

func (tb *ToolBuilder) ToClaude() anthropic.ToolParam {
	properties := make(map[string]interface{})
	required := make([]string, 0)

	for _, p := range tb.parameters {
		properties[p.Name] = tb.parameterToClaudeSchema(p)
		if p.Required {
			required = append(required, p.Name)
		}
	}

	return anthropic.ToolParam{
		Name: tb.name,
		Description: anthropic.String(tb.description),
		InputSchema: anthropic.ToolInputSchemaParam{
			Properties: properties,
			Required:   required,
		},
	}
}

func (tb *ToolBuilder) parameterToClaudeSchema(param *Parameter) map[string]interface{} {
	schema := map[string]interface{}{
		"type": string(param.Type),
	}

	if param.Description != "" {
		schema["description"] = param.Description
	}

	switch param.Type {
	case ParameterTypeArray:
		if param.Items != nil {
			schema["items"] = tb.parameterToClaudeSchema(param.Items)
		}
	case ParameterTypeObject:
		if param.Properties != nil {
			properties := make(map[string]interface{})
			required := make([]string, 0)
			for name, prop := range param.Properties {
				properties[name] = tb.parameterToClaudeSchema(prop)
				if prop.Required {
					required = append(required, name)
				}
			}
			schema["properties"] = properties
			if len(required) > 0 {
				schema["required"] = required
			}
		}
	}

	return schema
}

func (tb *ToolBuilder) ToGemini() *genai.FunctionDeclaration {
	properties := make(map[string]*genai.Schema)
	var required []string

	for _, p := range tb.parameters {
		properties[p.Name] = tb.parameterToGeminiSchema(p)
		if p.Required {
			required = append(required, p.Name)
		}
	}

	return &genai.FunctionDeclaration{
		Name:        tb.name,
		Description: tb.description,
		Parameters: &genai.Schema{
			Type:       genai.TypeObject,
			Properties: properties,
			Required:   required,
		},
	}
}

func (tb *ToolBuilder) parameterToGeminiSchema(param *Parameter) *genai.Schema {
	schema := &genai.Schema{
		Description: param.Description,
	}

	switch param.Type {
	case ParameterTypeString:
		schema.Type = genai.TypeString
	case ParameterTypeNumber:
		schema.Type = genai.TypeNumber
	case ParameterTypeBoolean:
		schema.Type = genai.TypeBoolean
	case ParameterTypeArray:
		schema.Type = genai.TypeArray
		if param.Items != nil {
			schema.Items = tb.parameterToGeminiSchema(param.Items)
		}
	case ParameterTypeObject:
		schema.Type = genai.TypeObject
		if param.Properties != nil {
			properties := make(map[string]*genai.Schema)
			var required []string
			for name, prop := range param.Properties {
				properties[name] = tb.parameterToGeminiSchema(prop)
				if prop.Required {
					required = append(required, name)
				}
			}
			schema.Properties = properties
			schema.Required = required
		}
	}

	return schema
}

func (cw *ContextWindow) AddTool(tool *ToolBuilder, runner ToolRunner) error {
	// Store the ToolBuilder itself, not the converted format
	// This allows different models to convert it as needed
	return cw.RegisterTool(tool.name, tool, runner)
}

func (cw *ContextWindow) AddToolFromJSON(name string, jsonDefinition interface{}, runner ToolRunner) error {
	return cw.RegisterTool(name, jsonDefinition, runner)
}
