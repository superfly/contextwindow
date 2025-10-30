package contextwindow

import (
	"context"
	"fmt"
	"os"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

const (
	ModelClaudeHaiku45  = "claude-haiku-4-5"
	ModelClaudeSonnet45 = "claude-sonnet-4-5"
	ModelClaudeSonnet40 = "claude-sonnet-4-0"
	ModelClaudeOpus41   = "claude-opus-4-1"
)

type ClaudeModel struct {
	client       *anthropic.Client
	model        string
	middleware   []Middleware
	toolExecutor ToolExecutor
}

func NewClaudeModel(model string) (*ClaudeModel, error) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("ANTHROPIC_API_KEY not set")
	}
	client := anthropic.NewClient(option.WithAPIKey(apiKey))
	return &ClaudeModel{
		client: &client,
		model:  model,
	}, nil
}

func (c *ClaudeModel) MaxTokens() int {
	return 200_000
}

// SetMiddleware sets the middleware for the Claude model
func (c *ClaudeModel) SetMiddleware(middleware []Middleware) {
	c.middleware = middleware
}

// SetToolExecutor sets the tool executor for the Claude model
func (c *ClaudeModel) SetToolExecutor(executor ToolExecutor) {
	c.toolExecutor = executor
}

func (c *ClaudeModel) Call(
	ctx context.Context,
	inputs []Record,
) ([]Record, int, error) {
	return c.CallWithOpts(ctx, inputs, CallModelOpts{})
}

func (c *ClaudeModel) CallWithOpts(
	ctx context.Context,
	inputs []Record,
	opts CallModelOpts,
) ([]Record, int, error) {
	var availableTools []ToolDefinition
	if c.toolExecutor != nil && !opts.DisableTools {
		availableTools = c.toolExecutor.GetRegisteredTools()
	}

	var systemBlocks []anthropic.TextBlockParam
	var messages []anthropic.MessageParam

	for _, rec := range inputs {
		switch rec.Source {
		case SystemPrompt:
			systemBlocks = append(systemBlocks, anthropic.TextBlockParam{
				Text: rec.Content,
			})
		case Prompt:
			messages = append(messages, anthropic.NewUserMessage(
				anthropic.NewTextBlock(rec.Content),
			))
		case ModelResp:
			messages = append(messages, anthropic.NewAssistantMessage(
				anthropic.NewTextBlock(rec.Content),
			))
		case ToolCall, ToolOutput:
			// For now, we'll just put the raw content in a message.
			// This will need to be revisited.
			messages = append(messages, anthropic.NewUserMessage(
				anthropic.NewTextBlock(rec.Content),
			))
		}
	}

	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(c.model),
		MaxTokens: 4096,
		Messages:  messages,
	}

	if len(systemBlocks) > 0 {
		params.System = systemBlocks
	}

	if len(availableTools) > 0 {
		tools := getClaudeToolParams(availableTools)
		params.Tools = tools
	}

	resp, err := c.client.Messages.New(ctx, params)
	if err != nil {
		return nil, 0, fmt.Errorf("Claude API: %w", err)
	}

	var events []Record
	totalTokens := int(resp.Usage.InputTokens + resp.Usage.OutputTokens)

	for hasToolUse(resp.Content) {
		var assistantContent []anthropic.ContentBlockParamUnion

		for _, block := range resp.Content {
			if block.Type == "text" && block.Text != "" {
				assistantContent = append(assistantContent, anthropic.NewTextBlock(block.Text))
			} else if block.Type == "tool_use" {
				assistantContent = append(assistantContent, anthropic.NewToolUseBlock(
					block.ID,
					block.Input,
					block.Name,
				))
			}
		}

		messages = append(messages, anthropic.MessageParam{
			Role:    anthropic.MessageParamRoleAssistant,
			Content: assistantContent,
		})

		var toolResults []anthropic.ContentBlockParamUnion

		for _, block := range resp.Content {
			if block.Type == "tool_use" {
				inputStr := string(block.Input)
				for _, m := range c.middleware {
					m.OnToolCall(ctx, block.Name, inputStr)
				}

				out, err := c.toolExecutor.ExecuteTool(ctx, block.Name, block.Input)
				if err != nil {
					out = fmt.Sprintf("error: %s", err)
				}

				for _, m := range c.middleware {
					m.OnToolResult(ctx, block.Name, out, err)
				}

				call := fmt.Sprintf("%s(%s)", block.Name, inputStr)
				events = append(events, Record{
					Source:    ToolCall,
					Content:   call,
					Live:      true,
					EstTokens: tokenCount(call),
				})
				events = append(events, Record{
					Source:    ToolOutput,
					Content:   out,
					Live:      true,
					EstTokens: tokenCount(out),
				})

				toolResults = append(toolResults, anthropic.NewToolResultBlock(
					block.ID,
					out,
					err != nil, // isError
				))
			}
		}

		messages = append(messages, anthropic.NewUserMessage(toolResults...))

		params.Messages = messages
		resp, err = c.client.Messages.New(ctx, params)
		if err != nil {
			return nil, 0, fmt.Errorf("Claude API (tool continuation): %w", err)
		}

		totalTokens += int(resp.Usage.InputTokens + resp.Usage.OutputTokens)
	}

	var responseText string
	for _, block := range resp.Content {
		if block.Type == "text" && block.Text != "" {
			responseText += block.Text
		}
	}

	events = append(events, Record{
		Source:    ModelResp,
		Content:   responseText,
		Live:      true,
		EstTokens: tokenCount(responseText),
	})

	return events, totalTokens, nil
}

// CallWithThreading implements ServerSideThreadingCapable interface
func (c *ClaudeModel) CallWithThreading(
	ctx context.Context,
	useServerSideThreading bool,
	lastResponseID *string,
	inputs []Record,
) ([]Record, *string, int, error) {
	return c.CallWithThreadingAndOpts(ctx, useServerSideThreading, lastResponseID, inputs, CallModelOpts{})
}

// CallWithThreadingAndOpts implements CallOptsCapable interface
func (c *ClaudeModel) CallWithThreadingAndOpts(
	ctx context.Context,
	useServerSideThreading bool,
	lastResponseID *string,
	inputs []Record,
	opts CallModelOpts,
) ([]Record, *string, int, error) {
	// Claude doesn't support server-side threading, so we always use client-side
	events, tokensUsed, err := c.CallWithOpts(ctx, inputs, opts)
	return events, nil, tokensUsed, err
}

// hasToolUse checks if the response content contains any tool_use blocks
func hasToolUse(content []anthropic.ContentBlockUnion) bool {
	for _, block := range content {
		if block.Type == "tool_use" {
			return true
		}
	}
	return false
}

// getClaudeToolParams converts ToolDefinitions to Claude tool union parameters
func getClaudeToolParams(availableTools []ToolDefinition) []anthropic.ToolUnionParam {
	var toolParams []anthropic.ToolUnionParam
	for _, tool := range availableTools {
		if claudeTool, ok := tool.Definition.(anthropic.ToolParam); ok {
			toolParams = append(toolParams, anthropic.ToolUnionParamOfTool(
				claudeTool.InputSchema,
				claudeTool.Name,
			))
			continue
		}

		if builder, ok := tool.Definition.(*ToolBuilder); ok {
			claudeTool := builder.ToClaude()
			toolParams = append(toolParams, anthropic.ToolUnionParamOfTool(
				claudeTool.InputSchema,
				claudeTool.Name,
			))
			continue
		}

		panic(fmt.Sprintf("can't convert tool definition for %s to Claude format (type: %T)", tool.Name, tool.Definition))
	}
	return toolParams
}
