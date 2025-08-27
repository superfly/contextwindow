package contextwindow

import (
	"context"
	"encoding/json"
	"fmt"
)

// TODO(tqbf): this is all pretty gnarly and half-baked, but comes of having
// only implemented this for OpenAI's client library; it'll stay gnarly until
// I do something with Claude.

// ToolDefinition represents a tool that can be called by the model.
type ToolDefinition struct {
	Name       string      `json:"name"`
	Definition interface{} `json:"definition"` // Model-specific tool definition (e.g., OpenAI FunctionDefinitionParam)
}

// ToolRunner defines the interface for executing a tool.
type ToolRunner interface {
	Run(ctx context.Context, args json.RawMessage) (string, error)
}

// ToolRunnerFunc allows functions to implement ToolRunner.
type ToolRunnerFunc func(ctx context.Context, args json.RawMessage) (string, error)

func (f ToolRunnerFunc) Run(ctx context.Context, args json.RawMessage) (string, error) {
	return f(ctx, args)
}

// ToolExecutor can execute tools by name and provide access to tool definitions.
type ToolExecutor interface {
	ExecuteTool(ctx context.Context, name string, args json.RawMessage) (string, error)
	GetRegisteredTools() []ToolDefinition
}

// RegisterTool registers a tool with this ContextWindow instance and stores the tool name as a hint in the database.
func (cw *ContextWindow) RegisterTool(name string, definition interface{}, runner ToolRunner) error {
	cw.registeredTools[name] = ToolDefinition{
		Name:       name,
		Definition: definition,
	}
	cw.toolRunners[name] = runner

	// Store the tool name in the database as a hint
	contextID, err := getContextIDByName(cw.db, cw.currentContext)
	if err != nil {
		return fmt.Errorf("register tool: %w", err)
	}
	_, err = AddContextTool(cw.db, contextID, name)
	if err != nil {
		return fmt.Errorf("register tool: %w", err)
	}
	return nil
}

// GetTool retrieves a registered tool runner by name.
func (cw *ContextWindow) GetTool(name string) (ToolRunner, bool) {
	runner, exists := cw.toolRunners[name]
	return runner, exists
}

// ExecuteTool implements ToolExecutor interface.
func (cw *ContextWindow) ExecuteTool(ctx context.Context, name string, args json.RawMessage) (string, error) {
	runner, exists := cw.toolRunners[name]
	if !exists {
		return "", fmt.Errorf("tool '%s' not registered", name)
	}
	return runner.Run(ctx, args)
}

// GetRegisteredTools returns all registered tool definitions.
func (cw *ContextWindow) GetRegisteredTools() []ToolDefinition {
	var tools []ToolDefinition
	for _, toolDef := range cw.registeredTools {
		tools = append(tools, toolDef)
	}
	return tools
}

// ListTools returns the names of all tools available in this context.
func (cw *ContextWindow) ListTools() ([]string, error) {
	contextID, err := getContextIDByName(cw.db, cw.currentContext)
	if err != nil {
		return nil, fmt.Errorf("list tools: %w", err)
	}
	return ListContextToolNames(cw.db, contextID)
}

// HasTool checks if a tool name is available in this context.
func (cw *ContextWindow) HasTool(name string) (bool, error) {
	contextID, err := getContextIDByName(cw.db, cw.currentContext)
	if err != nil {
		return false, fmt.Errorf("has tool: %w", err)
	}
	return HasContextTool(cw.db, contextID, name)
}
