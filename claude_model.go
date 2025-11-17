package contextwindow

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

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
		//lint:ignore ST1005 - Proper noun error message
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
			//lint:ignore ST1005 - Proper noun error message
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

// CallStreaming implements StreamingCapable interface
func (c *ClaudeModel) CallStreaming(
	ctx context.Context,
	inputs []Record,
	callback StreamCallback,
) ([]Record, int, error) {
	return c.CallStreamingWithOpts(ctx, inputs, CallModelOpts{}, callback)
}

// CallStreamingWithOpts implements StreamingOptsCapable interface
func (c *ClaudeModel) CallStreamingWithOpts(
	ctx context.Context,
	inputs []Record,
	opts CallModelOpts,
	callback StreamCallback,
) ([]Record, int, error) {
	var availableTools []ToolDefinition
	if c.toolExecutor != nil && !opts.DisableTools {
		availableTools = c.toolExecutor.GetRegisteredTools()
	}

	// Convert Records to messages (reuse existing logic)
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
			messages = append(messages, anthropic.NewUserMessage(
				anthropic.NewTextBlock(rec.Content),
			))
		}
	}

	// Build system blocks (reuse existing logic)
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

	var events []Record
	var totalTokensUsed int

	// Handle tool calls in a loop (similar to CallWithOpts)
	for {
		// Call NewStreaming (streaming is enabled by calling NewStreaming)
		stream := c.client.Messages.NewStreaming(ctx, params)

		// Accumulate deltas in buffer
		// Pre-allocate with 4KB capacity to reduce reallocations for typical responses
		// Profile with: go test -bench=. -benchmem -cpuprofile=cpu.prof -memprofile=mem.prof
		contentBuilder := strings.Builder{}
		contentBuilder.Grow(4096) // Pre-allocate 4KB buffer
		var toolUseBlocks []anthropic.ContentBlockUnion
		var currentToolUseBlock *anthropic.ToolUseBlock
		var usage anthropic.MessageDeltaUsage
		var hasUsage bool

		// Iterate over stream events
	streamLoop:
		for stream.Next() {
			// Check for context cancellation before processing event
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
					//lint:ignore ST1005 - Proper noun error message
					return events, tokenCount(partialContent), fmt.Errorf(
						"Claude streaming cancelled (partial response saved): %w", ctx.Err())
				}
				//lint:ignore ST1005 - Proper noun error message
				return nil, 0, fmt.Errorf("Claude streaming cancelled: %w", ctx.Err())
			default:
				// Continue processing
			}

			event := stream.Current()

			// Check for errors in the stream
			if stream.Err() != nil {
				streamErr := stream.Err()
				partialContent := contentBuilder.String()

				// Wrap error with provider context and error type
				var wrappedErr error
				if isNetworkError(streamErr) {
					//lint:ignore ST1005 - Proper noun error message
					wrappedErr = fmt.Errorf(
						"Claude streaming network error (partial response saved): %w", streamErr)
				} else {
					//lint:ignore ST1005 - Proper noun error message
					wrappedErr = fmt.Errorf(
						"Claude streaming error (partial response saved): %w", streamErr)
				}

				errChunk := StreamChunk{
					Error: wrappedErr,
					Done:  true,
				}
				if callback != nil {
					if err := callback(errChunk); err != nil {
						// If callback also errors, return both errors
						return nil, 0, fmt.Errorf(
							"callback error during stream error: %w (original: %w)", err, wrappedErr)
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

			// Handle different event types
			switch event.Type {
			case "message_start":
				// Initialize - message start event
				// No action needed, just acknowledge

			case "content_block_start":
				// Handle tool use start
				startEvent := event.AsContentBlockStart()
				if startEvent.ContentBlock.Type == "tool_use" {
					// Start accumulating a tool use block
					toolUse := startEvent.ContentBlock.AsToolUse()
					currentToolUseBlock = &anthropic.ToolUseBlock{
						ID:    toolUse.ID,
						Name:  toolUse.Name,
						Input: json.RawMessage("{}"), // Initialize empty
					}
				}

			case "content_block_delta":
				// Accumulate text or tool use deltas
				deltaEvent := event.AsContentBlockDelta()
				delta := deltaEvent.Delta
				if delta.Type == "text_delta" {
					// Text content delta
					if delta.JSON.Text.Valid() {
						text := delta.Text
						contentBuilder.WriteString(text)

						// Invoke callback for text deltas (minimize overhead by checking nil first)
						if callback != nil {
							// Pre-allocate metadata map with known size to reduce allocations
							metadata := make(map[string]any, 2)
							metadata["event_type"] = "content_block_delta"
							metadata["index"] = deltaEvent.Index
							chunk := StreamChunk{
								Delta:    text,
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
									return events, tokenCount(partialContent), fmt.Errorf(
										"callback error (partial response saved): %w", err)
								}
								return nil, 0, fmt.Errorf("callback error: %w", err)
							}
						}
					}
				} else if delta.Type == "input_json_delta" {
					// Tool use input delta - accumulate JSON
					if currentToolUseBlock != nil && delta.JSON.PartialJSON.Valid() {
						// Accumulate partial JSON for tool input
						currentInput := string(currentToolUseBlock.Input)
						if currentInput == "{}" {
							currentInput = ""
						}
						currentInput += delta.PartialJSON
						currentToolUseBlock.Input = json.RawMessage(currentInput)
					}
				}

			case "content_block_stop":
				// Tool use block is complete
				if currentToolUseBlock != nil {
					toolUseBlocks = append(toolUseBlocks, anthropic.ContentBlockUnion{
						Type:  "tool_use",
						ID:    currentToolUseBlock.ID,
						Name:  currentToolUseBlock.Name,
						Input: currentToolUseBlock.Input,
					})
					currentToolUseBlock = nil
				}

			case "message_delta":
				// Handle usage updates
				deltaEvent := event.AsMessageDelta()
				if deltaEvent.JSON.Usage.Valid() {
					usage = deltaEvent.Usage
					hasUsage = true
				}

			case "message_stop":
				// Finalize - stream is complete
				break streamLoop
			}
		}

		// Check for stream errors after iteration
		if stream.Err() != nil {
			streamErr := stream.Err()
			partialContent := contentBuilder.String()

			// Wrap error with provider context and error type
			var wrappedErr error
			if isNetworkError(streamErr) {
				//lint:ignore ST1005 - Proper noun error message
				wrappedErr = fmt.Errorf(
					"Claude streaming network error (partial response saved): %w", streamErr)
			} else {
				//lint:ignore ST1005 - Proper noun error message
				wrappedErr = fmt.Errorf(
					"Claude streaming error (partial response saved): %w", streamErr)
			}

			errChunk := StreamChunk{
				Error: wrappedErr,
				Done:  true,
			}
			if callback != nil {
				if err := callback(errChunk); err != nil {
					// If callback also errors, return both errors
					return nil, 0, fmt.Errorf(
						"callback error during stream error: %w (original: %w)", err, wrappedErr)
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
				//lint:ignore ST1005 - Proper noun error message
				return events, tokenCount(partialContent), fmt.Errorf(
					"Claude streaming cancelled after stream (partial response saved): %w", ctx.Err())
			}
			//lint:ignore ST1005 - Proper noun error message
			return nil, 0, fmt.Errorf("Claude streaming cancelled: %w", ctx.Err())
		default:
			// Continue
		}

		// Handle tool calls if present
		if len(toolUseBlocks) > 0 {
			// Add assistant message with tool calls to conversation
			var assistantContent []anthropic.ContentBlockParamUnion
			for _, block := range toolUseBlocks {
				if block.Type == "tool_use" {
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

			// Execute tools
			var toolResults []anthropic.ContentBlockParamUnion
			for _, block := range toolUseBlocks {
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
								//lint:ignore ST1005 - Proper noun error message
								return events, tokenCount(partialContent), fmt.Errorf(
									"Claude streaming cancelled during tool execution (partial response saved): %w", ctx.Err())
							}
							//lint:ignore ST1005 - Proper noun error message
							return nil, 0, fmt.Errorf("Claude streaming cancelled: %w", ctx.Err())
						default:
							// Continue
						}

						// Pre-allocate metadata map to reduce allocations
						metadata := make(map[string]any, 1)
						metadata["tool_call"] = block.Name
						toolResultChunk := StreamChunk{
							Delta:    fmt.Sprintf("\n[Tool: %s returned: %s]\n", block.Name, out),
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
			}

			messages = append(messages, anthropic.NewUserMessage(toolResults...))
			params.Messages = messages

			// Continue loop to get next response after tool calls
			toolUseBlocks = nil
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
			metadata["event_type"] = "message_stop"
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
			totalTokensUsed = int(usage.InputTokens + usage.OutputTokens)
		} else {
			// Estimate if usage not available
			totalTokensUsed = tokenCount(content)
		}

		return events, totalTokensUsed, nil
	}
}

// CallStreamingWithThreadingAndOpts implements streaming with threading support.
// Claude does not support server-side threading, so this always uses client-side streaming.
func (c *ClaudeModel) CallStreamingWithThreadingAndOpts(
	ctx context.Context,
	_ bool,
	_ *string,
	inputs []Record,
	opts CallModelOpts,
	callback StreamCallback,
) ([]Record, int, error) {
	return c.CallStreamingWithOpts(ctx, inputs, opts, callback)
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
