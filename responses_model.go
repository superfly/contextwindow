package contextwindow

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
	"github.com/openai/openai-go/v2/packages/param"
	"github.com/openai/openai-go/v2/responses"
	"github.com/openai/openai-go/v2/shared"
)

type OpenAIResponsesModel struct {
	client       *openai.Client
	model        shared.ResponsesModel
	middleware   []Middleware
	toolExecutor ToolExecutor
}

func NewOpenAIResponsesModel(model shared.ResponsesModel) (*OpenAIResponsesModel, error) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		return nil, fmt.Errorf("OPENAI_API_KEY not set")
	}
	client := openai.NewClient(option.WithAPIKey(os.Getenv("OPENAI_API_KEY")))
	return &OpenAIResponsesModel{client: &client, model: model}, nil
}

func (o *OpenAIResponsesModel) MaxTokens() int {
	return 128_000
}

func (o *OpenAIResponsesModel) SetMiddleware(middleware []Middleware) {
	o.middleware = middleware
}

func (o *OpenAIResponsesModel) SetToolExecutor(executor ToolExecutor) {
	o.toolExecutor = executor
}

func encodeMessage(msg, src string) responses.ResponseInputItemUnionParam {
	// this is fucking satanic
	ricups := []responses.ResponseInputContentUnionParam{}
	ricups = append(ricups, responses.ResponseInputContentParamOfInputText(msg))
	return responses.ResponseInputItemParamOfInputMessage(ricups, src)
}

// 	content := make([]responses.ResponseInputContentUnionParam, 0, len(message.Content))

// 	for _, block := range message.Content {
// 		item, err := EncodeUserContentBlock(block)
// 		if err != nil {
// 			return responses.ResponseInputItemUnionParam{}, fmt.Errorf("failed to encode content block: %w", err)
// 		}
// 		if item != nil {
// 			content = append(content, *item)
// 		}
// 	}

// 	return responses.ResponseInputItemParamOfInputMessage(content, "user"), nil
// }

func (o *OpenAIResponsesModel) Call(
	ctx context.Context,
	inputs []Record,
) ([]Record, int, error) {
	return o.CallWithOpts(ctx, inputs, CallModelOpts{})
}

func (o *OpenAIResponsesModel) CallWithOpts(
	ctx context.Context,
	inputs []Record,
	opts CallModelOpts,
) ([]Record, int, error) {
	events, _, tokensUsed, err := o.CallWithThreadingAndOpts(ctx, false, nil, inputs, opts)
	return events, tokensUsed, err
}

// callLLM is a wrapper helper for all LLM calls that handles threading logic
func (o *OpenAIResponsesModel) callLLM(
	ctx context.Context,
	fullMessageHistory string,
	toolParams []responses.ToolUnionParam,
	previousResponseID *string,
) (*responses.Response, error) {
	params := responses.ResponseNewParams{
		Model:             o.model,
		Tools:             toolParams,
		ParallelToolCalls: param.NewOpt(true),
	}

	if previousResponseID != nil {
		// Server-side threading: extract just the latest prompt
		latestPrompt := o.extractLatestPromptFromHistory(fullMessageHistory)
		if latestPrompt != "" {
			params.Input = responses.ResponseNewParamsInputUnion{
				OfString: param.NewOpt(latestPrompt),
			}
			params.PreviousResponseID = param.NewOpt(*previousResponseID)
		} else {
			// Fallback if can't extract prompt
			params.Input = responses.ResponseNewParamsInputUnion{
				OfString: param.NewOpt(fullMessageHistory),
			}
		}
	} else {
		// Client-side threading: use full message history
		params.Input = responses.ResponseNewParamsInputUnion{
			OfString: param.NewOpt(fullMessageHistory),
		}
	}

	resp, err := o.client.Responses.New(ctx, params)
	if err != nil && previousResponseID != nil {
		// If server-side threading failed, try falling back to client-side
		params.Input = responses.ResponseNewParamsInputUnion{
			OfString: param.NewOpt(fullMessageHistory),
		}
		params.PreviousResponseID = param.Null[string]()
		resp, err = o.client.Responses.New(ctx, params)
		if err != nil {
			return nil, fmt.Errorf("OpenAI responses (fallback): %w", err)
		}
	} else if err != nil {
		return nil, fmt.Errorf("OpenAI responses: %w", err)
	}

	return resp, nil
}

func (o *OpenAIResponsesModel) CallWithThreading(
	ctx context.Context,
	useServerSideThreading bool,
	lastResponseID *string,
	inputs []Record,
) ([]Record, *string, int, error) {
	return o.CallWithThreadingAndOpts(ctx, useServerSideThreading, lastResponseID, inputs, CallModelOpts{})
}

func (o *OpenAIResponsesModel) CallWithThreadingAndOpts(
	ctx context.Context,
	useServerSideThreading bool,
	lastResponseID *string,
	inputs []Record,
	opts CallModelOpts,
) ([]Record, *string, int, error) {
	var availableTools []ToolDefinition
	if o.toolExecutor != nil && !opts.DisableTools {
		availableTools = o.toolExecutor.GetRegisteredTools()
	}

	toolParams := getResponsesToolParamsFromDefinitions(availableTools)

	// Always prepare full conversation history
	fullMessageHistory := o.convertRecordsToInput(inputs)

	// Determine if we should use server-side threading
	var previousResponseID *string
	if useServerSideThreading && lastResponseID != nil && o.canUseServerSideThreading(inputs) {
		previousResponseID = lastResponseID
	}

	// Make the LLM call through our wrapper
	resp, err := o.callLLM(ctx, fullMessageHistory, toolParams, previousResponseID)
	if err != nil {
		return nil, nil, 0, err
	}

	var (
		events         []Record
		toolCallsFound = false
	)

	hasToolCall := func(r []responses.ResponseOutputItemUnion) bool {
		for _, it := range r {
			if it.Type == "function_call" {
				return true
			}
		}

		return false
	}

	toolCallsFound = hasToolCall(resp.Output)

	rawToolCall := func(it *responses.ResponseOutputItemUnion) string {
		for _, m := range o.middleware {
			m.OnToolCall(ctx, it.Name, it.Arguments)
		}

		out, err := o.toolExecutor.ExecuteTool(ctx, it.Name, json.RawMessage(it.Arguments))
		if err != nil {
			out = fmt.Sprintf("error: %s", err)
		}

		for _, m := range o.middleware {
			m.OnToolResult(ctx, it.Name, out, err)
		}

		return out
	}

	type responseItem = responses.ResponseOutputItemUnion
	type toolResult struct {
		item   responseItem
		output string
	}

	// Handle tool calls - build up the conversation history progressively
	currentHistory := fullMessageHistory

	for toolCallsFound {
		toolResults := []toolResult{}
		var toolCallsText []string

		for _, lastResponseItem := range resp.Output {
			if lastResponseItem.Type == "function_call" {
				var (
					out  = rawToolCall(&lastResponseItem)
					call = fmt.Sprintf("%s(%s)", lastResponseItem.Name, lastResponseItem.Arguments)
				)

				// save the tool call & output to the database
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

				toolResults = append(toolResults, toolResult{
					item:   lastResponseItem,
					output: out,
				})

				// Add to text representation
				toolCallsText = append(toolCallsText, "Tool Call: "+call)
				toolCallsText = append(toolCallsText, "Tool Output: "+out)
			}
		}

		// Update the conversation history with tool interactions
		for _, toolText := range toolCallsText {
			currentHistory += "\n" + toolText
		}

		// For tool calls, always use client-side threading (full history)
		// because tool call state is complex
		resp, err = o.callLLM(ctx, currentHistory, toolParams, nil)
		if err != nil {
			return nil, nil, 0, fmt.Errorf("tool call response: %w", err)
		}

		toolCallsFound = hasToolCall(resp.Output)
	}

	content := resp.OutputText()
	events = append(events, Record{
		Source:     ModelResp,
		Content:    content,
		Live:       true,
		EstTokens:  tokenCount(content),
		ResponseID: &resp.ID,
	})

	tokensUsed := int(resp.Usage.TotalTokens)
	return events, &resp.ID, tokensUsed, nil
}

func (o *OpenAIResponsesModel) convertRecordsToInput(inputs []Record) string {
	var parts []string
	for _, rec := range inputs {
		switch rec.Source {
		case SystemPrompt:
			parts = append([]string{"System: " + rec.Content}, parts...)
		case Prompt:
			parts = append(parts, "User: "+rec.Content)
		case ModelResp:
			parts = append(parts, "Assistant: "+rec.Content)
		case ToolCall:
			parts = append(parts, "Tool Call: "+rec.Content)
		case ToolOutput:
			parts = append(parts, "Tool Output: "+rec.Content)
		}
	}
	return strings.Join(parts, "\n")
}

func getResponsesToolParamsFromDefinitions(availableTools []ToolDefinition) []responses.ToolUnionParam {
	var toolParams []responses.ToolUnionParam
	for _, tool := range availableTools {
		if funcDef, ok := tool.Definition.(openai.FunctionDefinitionParam); ok {
			functionTool := responses.FunctionToolParam{
				Name:       funcDef.Name,
				Parameters: funcDef.Parameters,
				Strict:     funcDef.Strict,
			}
			if funcDef.Description.Valid() {
				functionTool.Description = funcDef.Description
			}
			toolParams = append(toolParams, responses.ToolUnionParam{
				OfFunction: &functionTool,
			})
			continue
		}

		if builder, ok := tool.Definition.(*ToolBuilder); ok {
			funcDef := builder.ToOpenAI()
			functionTool := responses.FunctionToolParam{
				Name:       funcDef.Name,
				Parameters: funcDef.Parameters,
				Strict:     funcDef.Strict,
			}
			if funcDef.Description.Valid() {
				functionTool.Description = funcDef.Description
			}
			toolParams = append(toolParams, responses.ToolUnionParam{
				OfFunction: &functionTool,
			})
			continue
		}

		panic(fmt.Sprintf("can't convert tool definition for %s to Responses format (type: %T)", tool.Name, tool.Definition))
	}
	return toolParams
}

// canUseServerSideThreading checks if server-side threading is safe to use
// Returns false if there are gaps in response IDs, tool calls, or other issues
func (o *OpenAIResponsesModel) canUseServerSideThreading(inputs []Record) bool {
	// Check for any tool calls in the conversation
	// Tool calls create complex state that may not thread properly
	for _, input := range inputs {
		if input.Source == ToolCall || input.Source == ToolOutput {
			return false
		}
	}

	// Look for the most recent model response to check for gaps
	var lastModelResponse *Record
	for i := len(inputs) - 1; i >= 0; i-- {
		if inputs[i].Source == ModelResp {
			lastModelResponse = &inputs[i]
			break
		}
	}

	// If there's a model response but no response ID, we have a gap
	if lastModelResponse != nil && (lastModelResponse.ResponseID == nil || *lastModelResponse.ResponseID == "") {
		return false
	}

	// Check for mixed response ID presence (some have IDs, some don't)
	hasResponseIDs := false
	hasMissingResponseIDs := false
	for _, input := range inputs {
		if input.Source == ModelResp {
			if input.ResponseID != nil && *input.ResponseID != "" {
				hasResponseIDs = true
			} else {
				hasMissingResponseIDs = true
			}
		}
	}

	// If we have mixed response ID state, don't use server-side threading
	if hasResponseIDs && hasMissingResponseIDs {
		return false
	}

	return true
}

// extractLatestPromptFromHistory extracts the most recent "User:" line from full message history
func (o *OpenAIResponsesModel) extractLatestPromptFromHistory(fullHistory string) string {
	lines := strings.Split(fullHistory, "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		line := strings.TrimSpace(lines[i])
		if strings.HasPrefix(line, "User: ") {
			return line
		}
	}
	return ""
}

// getLatestPrompt extracts the most recent user prompt from inputs for server-side threading
func (o *OpenAIResponsesModel) getLatestPrompt(inputs []Record) string {
	for i := len(inputs) - 1; i >= 0; i-- {
		if inputs[i].Source == Prompt {
			return "User: " + inputs[i].Content
		}
	}
	return ""
}
