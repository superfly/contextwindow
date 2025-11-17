package contextwindow

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

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

// CallStreaming implements StreamingCapable interface
func (o *OpenAIModel) CallStreaming(
	ctx context.Context,
	inputs []Record,
	callback StreamCallback,
) ([]Record, int, error) {
	return o.CallStreamingWithOpts(ctx, inputs, CallModelOpts{}, callback)
}

// CallStreamingWithOpts implements StreamingOptsCapable interface
func (o *OpenAIModel) CallStreamingWithOpts(
	ctx context.Context,
	inputs []Record,
	opts CallModelOpts,
	callback StreamCallback,
) ([]Record, int, error) {
	var availableTools []ToolDefinition
	if o.toolExecutor != nil && !opts.DisableTools {
		availableTools = o.toolExecutor.GetRegisteredTools()
	}

	// Convert Records to messages (reuse existing logic)
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
			messages = append(messages, openai.AssistantMessage(rec.Content))
		case ToolOutput:
			messages = append(messages, openai.UserMessage(rec.Content))
		}
	}

	// Build tool parameters (reuse existing logic)
	toolParams := getToolParamsFromDefinitions(availableTools)

	var events []Record
	var totalTokensUsed int

	// Handle tool calls in a loop (similar to CallWithOpts)
	for {
		// Create ChatCompletionNewParams (streaming is enabled by calling NewStreaming)
		params := openai.ChatCompletionNewParams{
			Model:    o.model,
			Messages: messages,
			Tools:    toolParams,
		}

		// Call NewStreaming
		stream := o.client.Chat.Completions.NewStreaming(ctx, params)

		// Accumulate deltas in buffer
		// Pre-allocate with 4KB capacity to reduce reallocations for typical responses
		// Profile with: go test -bench=. -benchmem -cpuprofile=cpu.prof -memprofile=mem.prof
		contentBuilder := strings.Builder{}
		contentBuilder.Grow(4096) // Pre-allocate 4KB buffer
		var toolCalls []openai.ChatCompletionChunkChoiceDeltaToolCall
		var finishReason string
		var usage openai.CompletionUsage
		var hasUsage bool

		// Iterate over stream chunks
		for stream.Next() {
			// Check for context cancellation before processing chunk
			select {
			case <-ctx.Done():
				// Context was cancelled - handle partial response
				partialContent := contentBuilder.String()
				errChunk := StreamChunk{
					Error: fmt.Errorf("stream cancelled: %w", ctx.Err()),
					Done:  true,
				}
				if callback != nil {
					_ = callback(errChunk) // Best effort to notify callback
				}
				// Return partial response if we have any content
				if partialContent != "" {
					events = append(events, Record{
						Source:    ModelResp,
						Content:   partialContent,
						Live:      true,
						EstTokens: tokenCount(partialContent),
					})
					return events, tokenCount(partialContent), fmt.Errorf("OpenAI streaming cancelled (partial response saved): %w", ctx.Err())
				}
				return nil, 0, fmt.Errorf("OpenAI streaming cancelled: %w", ctx.Err())
			default:
				// Continue processing
			}

			chunk := stream.Current()

			// Check for errors in the stream
			if stream.Err() != nil {
				streamErr := stream.Err()
				partialContent := contentBuilder.String()

				// Wrap error with provider context and error type
				var wrappedErr error
				if isNetworkError(streamErr) {
					wrappedErr = fmt.Errorf("OpenAI streaming network error (partial response saved): %w", streamErr)
				} else {
					wrappedErr = fmt.Errorf("OpenAI streaming error (partial response saved): %w", streamErr)
				}

				errChunk := StreamChunk{
					Error: wrappedErr,
					Done:  true,
				}
				if callback != nil {
					if err := callback(errChunk); err != nil {
						// If callback also errors, return both errors
						return nil, 0, fmt.Errorf("callback error during stream error: %w (original: %w)", err, wrappedErr)
					}
				}

				// Return partial response if we have any content
				if partialContent != "" {
					events = append(events, Record{
						Source:    ModelResp,
						Content:   partialContent,
						Live:      true,
						EstTokens: tokenCount(partialContent),
					})
					return events, tokenCount(partialContent), wrappedErr
				}
				return nil, 0, wrappedErr
			}

			if len(chunk.Choices) == 0 {
				continue
			}

			choice := chunk.Choices[0]

			// Handle finish reason
			if choice.FinishReason != "" {
				finishReason = choice.FinishReason
			}

			// Handle usage information
			if chunk.JSON.Usage.Valid() {
				usage = chunk.Usage
				hasUsage = true
			}

			// Handle content deltas
			if choice.Delta.JSON.Content.Valid() {
				delta := choice.Delta.Content
				contentBuilder.WriteString(delta)

				// Invoke callback for each chunk (minimize overhead by checking nil first)
				if callback != nil {
					// Reuse metadata map to reduce allocations
					metadata := make(map[string]any, 1)
					if finishReason != "" {
						metadata["finish_reason"] = finishReason
					}
					chunk := StreamChunk{
						Delta:    delta,
						Done:     false,
						Metadata: metadata,
					}
					if err := callback(chunk); err != nil {
						// Callback requested cancellation - save partial response
						partialContent := contentBuilder.String()
						if partialContent != "" {
							events = append(events, Record{
								Source:    ModelResp,
								Content:   partialContent,
								Live:      true,
								EstTokens: tokenCount(partialContent),
							})
							return events, tokenCount(partialContent), fmt.Errorf("callback error (partial response saved): %w", err)
						}
						return nil, 0, fmt.Errorf("callback error: %w", err)
					}
				}
			}

			// Handle tool call deltas - buffer until complete
			if len(choice.Delta.ToolCalls) > 0 {
				toolCalls = append(toolCalls, choice.Delta.ToolCalls...)
			}
		}

		// Check for stream errors after iteration
		if stream.Err() != nil {
			streamErr := stream.Err()
			partialContent := contentBuilder.String()

			// Wrap error with provider context and error type
			var wrappedErr error
			if isNetworkError(streamErr) {
				wrappedErr = fmt.Errorf("OpenAI streaming network error (partial response saved): %w", streamErr)
			} else {
				wrappedErr = fmt.Errorf("OpenAI streaming error (partial response saved): %w", streamErr)
			}

			errChunk := StreamChunk{
				Error: wrappedErr,
				Done:  true,
			}
			if callback != nil {
				if err := callback(errChunk); err != nil {
					// If callback also errors, return both errors
					return nil, 0, fmt.Errorf("callback error during stream error: %w (original: %w)", err, wrappedErr)
				}
			}

			// Return partial response if we have any content
			if partialContent != "" {
				events = append(events, Record{
					Source:    ModelResp,
					Content:   partialContent,
					Live:      true,
					EstTokens: tokenCount(partialContent),
				})
				return events, tokenCount(partialContent), wrappedErr
			}
			return nil, 0, wrappedErr
		}

		// Check for context cancellation after stream completes
		select {
		case <-ctx.Done():
			partialContent := contentBuilder.String()
			if partialContent != "" {
				events = append(events, Record{
					Source:    ModelResp,
					Content:   partialContent,
					Live:      true,
					EstTokens: tokenCount(partialContent),
				})
				return events, tokenCount(partialContent), fmt.Errorf("OpenAI streaming cancelled after stream (partial response saved): %w", ctx.Err())
			}
			return nil, 0, fmt.Errorf("OpenAI streaming cancelled: %w", ctx.Err())
		default:
			// Continue
		}

		// Handle tool calls if present
		if len(toolCalls) > 0 {
			// Reconstruct complete tool calls from deltas
			// OpenAI streams tool calls as deltas, so we need to accumulate them
			completeToolCalls := o.reconstructToolCalls(toolCalls)

			// Add assistant message with tool calls to conversation
			assistantParam := openai.ChatCompletionAssistantMessageParam{
				ToolCalls: completeToolCalls,
			}
			messages = append(messages, openai.ChatCompletionMessageParamUnion{
				OfAssistant: &assistantParam,
			})

			// Execute tools
			for _, tc := range completeToolCalls {
				if tc.OfFunction == nil {
					continue
				}
				tcFunc := tc.OfFunction

				for _, m := range o.middleware {
					m.OnToolCall(ctx, tcFunc.Function.Name, string(tcFunc.Function.Arguments))
				}

				out, err := o.toolExecutor.ExecuteTool(ctx, tcFunc.Function.Name, json.RawMessage(tcFunc.Function.Arguments))
				if err != nil {
					out = fmt.Sprintf("error: %s", err)
				}

				for _, m := range o.middleware {
					m.OnToolResult(ctx, tcFunc.Function.Name, out, err)
				}

				messages = append(messages, openai.ToolMessage(out, tcFunc.ID))

				// Record tool call and output events
				call := fmt.Sprintf("%s(%s)", tcFunc.Function.Name, tcFunc.Function.Arguments)
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

				// Stream tool result if callback provided
				if callback != nil {
					// Check for context cancellation before streaming tool result
					select {
					case <-ctx.Done():
						partialContent := contentBuilder.String()
						if partialContent != "" {
							events = append(events, Record{
								Source:    ModelResp,
								Content:   partialContent,
								Live:      true,
								EstTokens: tokenCount(partialContent),
							})
							return events, tokenCount(partialContent), fmt.Errorf("OpenAI streaming cancelled during tool execution (partial response saved): %w", ctx.Err())
						}
						return nil, 0, fmt.Errorf("OpenAI streaming cancelled: %w", ctx.Err())
					default:
						// Continue
					}

					// Pre-allocate metadata map to reduce allocations
					metadata := make(map[string]any, 1)
					metadata["tool_call"] = tcFunc.Function.Name
					toolResultChunk := StreamChunk{
						Delta:    fmt.Sprintf("\n[Tool: %s returned: %s]\n", tcFunc.Function.Name, out),
						Done:     false,
						Metadata: metadata,
					}
					if err := callback(toolResultChunk); err != nil {
						// Callback requested cancellation - save partial response
						partialContent := contentBuilder.String()
						if partialContent != "" {
							events = append(events, Record{
								Source:    ModelResp,
								Content:   partialContent,
								Live:      true,
								EstTokens: tokenCount(partialContent),
							})
							return events, tokenCount(partialContent), fmt.Errorf("callback error during tool execution (partial response saved): %w", err)
						}
						return nil, 0, fmt.Errorf("callback error: %w", err)
					}
				}
			}

			// Continue loop to get next response after tool calls
			toolCalls = nil
			contentBuilder.Reset()
			contentBuilder.Grow(4096) // Re-allocate buffer for next iteration
			continue
		}

		// No tool calls, we have the final response
		content := contentBuilder.String()

		// Send done chunk
		if callback != nil {
			// Pre-allocate metadata map to reduce allocations
			metadata := make(map[string]any, 1)
			if finishReason != "" {
				metadata["finish_reason"] = finishReason
			}
			doneChunk := StreamChunk{
				Delta:    "",
				Done:     true,
				Metadata: metadata,
			}
			if err := callback(doneChunk); err != nil {
				return nil, 0, fmt.Errorf("callback error: %w", err)
			}
		}

		// Record final response
		events = append(events, Record{
			Source:    ModelResp,
			Content:   content,
			Live:      true,
			EstTokens: tokenCount(content),
		})

		// Get token count from usage or estimate
		if hasUsage {
			totalTokensUsed = int(usage.TotalTokens)
		} else {
			// Estimate if usage not available
			totalTokensUsed = tokenCount(content)
		}

		return events, totalTokensUsed, nil
	}
}

// CallStreamingWithThreadingAndOpts implements streaming with threading support
func (o *OpenAIModel) CallStreamingWithThreadingAndOpts(
	ctx context.Context,
	useServerSideThreading bool,
	lastResponseID *string,
	inputs []Record,
	opts CallModelOpts,
	callback StreamCallback,
) ([]Record, int, error) {
	if useServerSideThreading {
		return nil, 0, fmt.Errorf("server-side threading not supported by OpenAI completions API with streaming")
	}

	// Fall back to client-side streaming
	return o.CallStreamingWithOpts(ctx, inputs, opts, callback)
}

// reconstructToolCalls reconstructs complete tool calls from streaming deltas
func (o *OpenAIModel) reconstructToolCalls(deltas []openai.ChatCompletionChunkChoiceDeltaToolCall) []openai.ChatCompletionMessageToolCallUnionParam {
	// Map to accumulate tool calls by index
	toolCallMap := make(map[int64]*openai.ChatCompletionMessageFunctionToolCallParam)

	for _, delta := range deltas {
		idx := delta.Index
		if toolCallMap[idx] == nil {
			toolCallMap[idx] = &openai.ChatCompletionMessageFunctionToolCallParam{
				ID:   delta.ID,
				Type: "function",
			}
		}

		tc := toolCallMap[idx]

		// Accumulate function name
		if delta.Function.JSON.Name.Valid() {
			tc.Function.Name += delta.Function.Name
		}

		// Accumulate function arguments
		if delta.Function.JSON.Arguments.Valid() {
			tc.Function.Arguments += delta.Function.Arguments
		}
	}

	// Convert map to slice in order
	var maxIdx int64
	for idx := range toolCallMap {
		if idx > maxIdx {
			maxIdx = idx
		}
	}

	result := make([]openai.ChatCompletionMessageToolCallUnionParam, 0, maxIdx+1)
	for i := int64(0); i <= maxIdx; i++ {
		if tc, ok := toolCallMap[i]; ok {
			result = append(result, openai.ChatCompletionMessageToolCallUnionParam{
				OfFunction: tc,
			})
		}
	}

	return result
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
