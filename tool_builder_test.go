package contextwindow

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewTool(t *testing.T) {
	tool := NewTool("test_tool", "A test tool")

	assert.Equal(t, "test_tool", tool.name)
	assert.Equal(t, "A test tool", tool.description)
	assert.Empty(t, tool.parameters)
}

func TestAddStringParameter(t *testing.T) {
	tool := NewTool("test", "test").
		AddStringParameter("name", "A string parameter", true)

	assert.Len(t, tool.parameters, 1)
	param := tool.parameters[0]
	assert.Equal(t, "name", param.Name)
	assert.Equal(t, ParameterTypeString, param.Type)
	assert.Equal(t, "A string parameter", param.Description)
	assert.True(t, param.Required)
}

func TestAddNumberParameter(t *testing.T) {
	tool := NewTool("test", "test").
		AddNumberParameter("count", "A number parameter", false)

	assert.Len(t, tool.parameters, 1)
	param := tool.parameters[0]
	assert.Equal(t, "count", param.Name)
	assert.Equal(t, ParameterTypeNumber, param.Type)
	assert.Equal(t, "A number parameter", param.Description)
	assert.False(t, param.Required)
}

func TestAddBooleanParameter(t *testing.T) {
	tool := NewTool("test", "test").
		AddBooleanParameter("flag", "A boolean parameter", true)

	assert.Len(t, tool.parameters, 1)
	param := tool.parameters[0]
	assert.Equal(t, "flag", param.Name)
	assert.Equal(t, ParameterTypeBoolean, param.Type)
	assert.Equal(t, "A boolean parameter", param.Description)
	assert.True(t, param.Required)
}

func TestAddArrayParameter(t *testing.T) {
	tool := NewTool("test", "test").
		AddArrayParameter("items", "An array parameter", false, ParameterTypeString)

	assert.Len(t, tool.parameters, 1)
	param := tool.parameters[0]
	assert.Equal(t, "items", param.Name)
	assert.Equal(t, ParameterTypeArray, param.Type)
	assert.Equal(t, "An array parameter", param.Description)
	assert.False(t, param.Required)
	assert.NotNil(t, param.Items)
	assert.Equal(t, ParameterTypeString, param.Items.Type)
}

func TestAddObjectParameter(t *testing.T) {
	properties := map[string]*Parameter{
		"field1": {
			Name:     "field1",
			Type:     ParameterTypeString,
			Required: true,
		},
		"field2": {
			Name:     "field2",
			Type:     ParameterTypeNumber,
			Required: false,
		},
	}

	tool := NewTool("test", "test").
		AddObjectParameter("config", "An object parameter", true, properties)

	assert.Len(t, tool.parameters, 1)
	param := tool.parameters[0]
	assert.Equal(t, "config", param.Name)
	assert.Equal(t, ParameterTypeObject, param.Type)
	assert.Equal(t, "An object parameter", param.Description)
	assert.True(t, param.Required)
	assert.Equal(t, properties, param.Properties)
}

func TestToOpenAI(t *testing.T) {
	tool := NewTool("example_tool", "An example tool for testing").
		AddStringParameter("query", "Search query", true).
		AddNumberParameter("limit", "Result limit", false).
		AddBooleanParameter("verbose", "Verbose output", false)

	openaiDef := tool.ToOpenAI()

	assert.Equal(t, "example_tool", openaiDef.Name)
	assert.Equal(t, "An example tool for testing", openaiDef.Description.Value)

	params := openaiDef.Parameters
	assert.Equal(t, "object", params["type"])

	properties := params["properties"].(map[string]any)
	assert.Contains(t, properties, "query")
	assert.Contains(t, properties, "limit")
	assert.Contains(t, properties, "verbose")

	queryProp := properties["query"].(map[string]any)
	assert.Equal(t, "string", queryProp["type"])
	assert.Equal(t, "Search query", queryProp["description"])

	limitProp := properties["limit"].(map[string]any)
	assert.Equal(t, "number", limitProp["type"])
	assert.Equal(t, "Result limit", limitProp["description"])

	verboseProp := properties["verbose"].(map[string]any)
	assert.Equal(t, "boolean", verboseProp["type"])
	assert.Equal(t, "Verbose output", verboseProp["description"])

	required := params["required"].([]string)
	assert.Contains(t, required, "query")
	assert.NotContains(t, required, "limit")
	assert.NotContains(t, required, "verbose")
}

func TestToOpenAIWithArray(t *testing.T) {
	tool := NewTool("array_tool", "Tool with array parameter").
		AddArrayParameter("tags", "List of tags", true, ParameterTypeString)

	openaiDef := tool.ToOpenAI()
	properties := openaiDef.Parameters["properties"].(map[string]any)
	tagsProp := properties["tags"].(map[string]any)

	assert.Equal(t, "array", tagsProp["type"])
	assert.Equal(t, "List of tags", tagsProp["description"])

	items := tagsProp["items"].(map[string]any)
	assert.Equal(t, "string", items["type"])
}

func TestToOpenAIWithObject(t *testing.T) {
	properties := map[string]*Parameter{
		"name": {
			Type:        ParameterTypeString,
			Description: "Person's name",
			Required:    true,
		},
		"age": {
			Type:        ParameterTypeNumber,
			Description: "Person's age",
			Required:    false,
		},
	}

	tool := NewTool("object_tool", "Tool with object parameter").
		AddObjectParameter("person", "Person information", true, properties)

	openaiDef := tool.ToOpenAI()
	toolProps := openaiDef.Parameters["properties"].(map[string]any)
	personProp := toolProps["person"].(map[string]any)

	assert.Equal(t, "object", personProp["type"])
	assert.Equal(t, "Person information", personProp["description"])

	personProperties := personProp["properties"].(map[string]any)
	assert.Contains(t, personProperties, "name")
	assert.Contains(t, personProperties, "age")

	nameProp := personProperties["name"].(map[string]any)
	assert.Equal(t, "string", nameProp["type"])
	assert.Equal(t, "Person's name", nameProp["description"])

	ageProp := personProperties["age"].(map[string]any)
	assert.Equal(t, "number", ageProp["type"])
	assert.Equal(t, "Person's age", ageProp["description"])

	personRequired := personProp["required"].([]string)
	assert.Contains(t, personRequired, "name")
	assert.NotContains(t, personRequired, "age")
}

func TestMethodChaining(t *testing.T) {
	tool := NewTool("chained_tool", "Testing method chaining").
		AddStringParameter("param1", "First param", true).
		AddNumberParameter("param2", "Second param", false).
		AddBooleanParameter("param3", "Third param", true)

	assert.Len(t, tool.parameters, 3)
	assert.Equal(t, "param1", tool.parameters[0].Name)
	assert.Equal(t, "param2", tool.parameters[1].Name)
	assert.Equal(t, "param3", tool.parameters[2].Name)
}

func TestContextWindowAddTool(t *testing.T) {
	db, err := NewContextDB(":memory:")
	assert.NoError(t, err)
	defer db.Close()

	cw, err := NewContextWindow(db, &dummyModel{}, "test_context")
	assert.NoError(t, err)

	tool := NewTool("test_tool", "A test tool").
		AddStringParameter("input", "Test input", true)

	runner := ToolRunnerFunc(func(ctx context.Context, args json.RawMessage) (string, error) {
		return "test result", nil
	})

	err = cw.AddTool(tool, runner)
	assert.NoError(t, err)

	registeredRunner, exists := cw.GetTool("test_tool")
	assert.True(t, exists)
	assert.NotNil(t, registeredRunner)

	result, err := registeredRunner.Run(context.Background(), json.RawMessage(`{"input":"test"}`))
	assert.NoError(t, err)
	assert.Equal(t, "test result", result)
}

func TestContextWindowAddToolFromJSON(t *testing.T) {
	db, err := NewContextDB(":memory:")
	assert.NoError(t, err)
	defer db.Close()

	cw, err := NewContextWindow(db, &dummyModel{}, "test_context")
	assert.NoError(t, err)

	jsonDef := map[string]interface{}{
		"name":        "json_tool",
		"description": "A tool defined with JSON",
		"parameters": map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"value": map[string]string{"type": "string"},
			},
			"required": []string{"value"},
		},
	}

	runner := ToolRunnerFunc(func(ctx context.Context, args json.RawMessage) (string, error) {
		return "json result", nil
	})

	err = cw.AddToolFromJSON("json_tool", jsonDef, runner)
	assert.NoError(t, err)

	registeredRunner, exists := cw.GetTool("json_tool")
	assert.True(t, exists)
	assert.NotNil(t, registeredRunner)
}
