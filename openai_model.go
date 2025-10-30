package contextwindow

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
	"github.com/openai/openai-go/v2/shared"
)

const (
	ResponsesModelGPT5     shared.ResponsesModel = "gpt-5-2025-08-07"
	ResponsesModelGPT5Mini shared.ResponsesModel = "gpt-5-mini-2025-08-07"
	ResponsesModel4o       shared.ResponsesModel = "gpt-4o-2024-08-06"
	ResponsesModelO4Mini   shared.ResponsesModel = "o4-mini-2025-04-16"
)

type OpenAIModel struct {
	client       *openai.Client
	model        shared.ChatModel
	middleware   []Middleware
	toolExecutor ToolExecutor
}

type llmToolParam = openai.ChatCompletionToolUnionParam

func NewOpenAIModel(model shared.ChatModel) (*OpenAIModel, error) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		return nil, fmt.Errorf("OPENAI_API_KEY not set")
	}
	client := openai.NewClient(option.WithAPIKey(os.Getenv("OPENAI_API_KEY")))
	return &OpenAIModel{client: &client, model: model}, nil
}

func (o *OpenAIModel) MaxTokens() int {
	return 128_000
}

// SetMiddleware sets the middleware for the OpenAI model
func (o *OpenAIModel) SetMiddleware(middleware []Middleware) {
	o.middleware = middleware
}

// SetToolExecutor sets the tool executor for the OpenAI model
func (o *OpenAIModel) SetToolExecutor(executor ToolExecutor) {
	o.toolExecutor = executor
}

func (o *OpenAIModel) Call(
	ctx context.Context,
	inputs []Record,
) ([]Record, int, error) {
	return o.CallWithOpts(ctx, inputs, CallModelOpts{})
}

func (o *OpenAIModel) CallWithOpts(
	ctx context.Context,
	inputs []Record,
	opts CallModelOpts,
) ([]Record, int, error) {
	var availableTools []ToolDefinition
	if o.toolExecutor != nil && !opts.DisableTools {
		availableTools = o.toolExecutor.GetRegisteredTools()
	}
	var messages []openai.ChatCompletionMessageParamUnion
	for _, rec := range inputs {
		switch rec.Source {
		case SystemPrompt:
			messages = append([]openai.ChatCompletionMessageParamUnion{openai.SystemMessage(rec.Content)}, messages...)
		case Prompt:
			messages = append(messages, openai.UserMessage(rec.Content))
		case ModelResp:
			messages = append(messages, openai.AssistantMessage(rec.Content))
		case ToolCall:
			// For now, we'll just put the raw content in a message.
			// This will need to be revisited.
			messages = append(messages, openai.AssistantMessage(rec.Content))
		case ToolOutput:
			messages = append(messages, openai.UserMessage(rec.Content))
		}
	}

	toolParams := getToolParamsFromDefinitions(availableTools)

	params := openai.ChatCompletionNewParams{
		Model:    o.model,
		Messages: messages,
		Tools:    toolParams,
	}
	resp, err := o.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, 0, fmt.Errorf("OpenAI chat: %w", err)
	}
	if len(resp.Choices) == 0 {
		return nil, 0, fmt.Errorf("no choices in response")
	}

	choice := resp.Choices[0].Message

	var events []Record
	for len(choice.ToolCalls) > 0 {
		messages = append(messages, choice.ToParam())

		for _, tc := range choice.ToolCalls {
			for _, m := range o.middleware {
				m.OnToolCall(ctx, tc.Function.Name, string(tc.Function.Arguments))
			}

			out, err := o.toolExecutor.ExecuteTool(ctx, tc.Function.Name, json.RawMessage(tc.Function.Arguments))
			if err != nil {
				out = fmt.Sprintf("error: %s", err)
			}

			for _, m := range o.middleware {
				m.OnToolResult(ctx, tc.Function.Name, out, err)
			}

			messages = append(messages, openai.ToolMessage(out, tc.ID))

			// Also record these events for persistence
			call := fmt.Sprintf("%s(%s)", tc.Function.Name, tc.Function.Arguments)
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
		}

		params.Messages = messages
		resp, err := o.client.Chat.Completions.New(ctx, params)
		if err != nil {
			return nil, 0, fmt.Errorf("OpenAI chat: %w", err)
		}
		if len(resp.Choices) == 0 {
			return nil, 0, fmt.Errorf("no choices in response")
		}

		choice = resp.Choices[0].Message
	}

	events = append(events, Record{
		Source:    ModelResp,
		Content:   choice.Content,
		Live:      true,
		EstTokens: tokenCount(choice.Content),
	})
	tokensUsed := int(resp.Usage.TotalTokens)
	return events, tokensUsed, nil
}

// CallWithThreading implements ServerSideThreadingCapable interface
func (o *OpenAIModel) CallWithThreading(
	ctx context.Context,
	useServerSideThreading bool,
	lastResponseID *string,
	inputs []Record,
) ([]Record, *string, int, error) {
	return o.CallWithThreadingAndOpts(ctx, useServerSideThreading, lastResponseID, inputs, CallModelOpts{})
}

// CallWithThreadingAndOpts implements CallOptsCapable interface
func (o *OpenAIModel) CallWithThreadingAndOpts(
	ctx context.Context,
	useServerSideThreading bool,
	lastResponseID *string,
	inputs []Record,
	opts CallModelOpts,
) ([]Record, *string, int, error) {
	if useServerSideThreading {
		return nil, nil, 0, fmt.Errorf("server-side threading not supported by OpenAI completions API")
	}

	// Fall back to regular client-side threading
	events, tokensUsed, err := o.CallWithOpts(ctx, inputs, opts)
	return events, nil, tokensUsed, err
}

// getToolParamsFromDefinitions converts ToolDefinitions to OpenAI tool parameters.
func getToolParamsFromDefinitions(availableTools []ToolDefinition) []llmToolParam {
	var toolParams []llmToolParam
	for _, tool := range availableTools {
		if funcDef, ok := tool.Definition.(openai.FunctionDefinitionParam); ok {
			toolParams = append(toolParams, openai.ChatCompletionFunctionTool(funcDef))
			continue
		}

		if builder, ok := tool.Definition.(*ToolBuilder); ok {
			toolParams = append(toolParams, openai.ChatCompletionFunctionTool(builder.ToOpenAI()))
			continue
		}

		panic(fmt.Sprintf("can't add tool definition for %s (type: %T)", tool.Name, tool.Definition))
	}
	return toolParams
}
