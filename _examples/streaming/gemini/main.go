package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/superfly/contextwindow"
)

func main() {
	ctx := context.Background()

	// Example 1: Basic streaming
	fmt.Println("=== Example 1: Basic Streaming ===")
	basicStreamingExample(ctx)
	fmt.Println()

	// Example 2: Advanced streaming with progress tracking
	fmt.Println("=== Example 2: Advanced Streaming with Progress Tracking ===")
	advancedStreamingExample(ctx)
	fmt.Println()

	// Example 3: Streaming with tool calls
	fmt.Println("=== Example 3: Streaming with Tool Calls ===")
	toolCallStreamingExample(ctx)
}

// basicStreamingExample demonstrates the simplest streaming usage.
func basicStreamingExample(ctx context.Context) {
	// reads GOOGLE_GENAI_API_KEY or GEMINI_API_KEY
	model, err := contextwindow.NewGeminiModel(contextwindow.ModelGemini20Flash)
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	db, err := contextwindow.NewContextDB(":memory:")
	if err != nil {
		log.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	cw, err := contextwindow.NewContextWindow(db, model, "")
	if err != nil {
		log.Fatalf("Failed to create context window: %v", err)
	}
	defer cw.Close()

	if err := cw.AddPrompt("Write a short haiku about programming."); err != nil {
		log.Fatalf("Failed to add prompt: %v", err)
	}

	// Basic streaming callback: print deltas as they arrive
	callback := func(chunk contextwindow.StreamChunk) error {
		if !chunk.Done {
			fmt.Print(chunk.Delta)
		}
		return nil
	}

	response, err := cw.CallModelStreaming(ctx, callback)
	if err != nil {
		log.Fatalf("Failed to call model: %v", err)
	}

	fmt.Printf("\n\n[Complete response: %s]\n", response)
}

// advancedStreamingExample demonstrates streaming with progress tracking and metadata.
func advancedStreamingExample(ctx context.Context) {
	// reads GOOGLE_GENAI_API_KEY or GEMINI_API_KEY
	model, err := contextwindow.NewGeminiModel(contextwindow.ModelGemini20Flash)
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	db, err := contextwindow.NewContextDB(":memory:")
	if err != nil {
		log.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	cw, err := contextwindow.NewContextWindow(db, model, "")
	if err != nil {
		log.Fatalf("Failed to create context window: %v", err)
	}
	defer cw.Close()

	if err := cw.AddPrompt("Explain quantum computing in simple terms, in about 100 words."); err != nil {
		log.Fatalf("Failed to add prompt: %v", err)
	}

	var (
		chunkCount  int
		startTime   = time.Now()
		lastUpdate  = time.Now()
		accumulated strings.Builder
	)

	// Advanced callback with progress tracking
	callback := func(chunk contextwindow.StreamChunk) error {
		chunkCount++

		// Accumulate text
		if chunk.Delta != "" {
			accumulated.WriteString(chunk.Delta)
			fmt.Print(chunk.Delta)
		}

		// Check for errors
		if chunk.Error != nil {
			return fmt.Errorf("streaming error: %w", chunk.Error)
		}

		// Update progress every 500ms
		now := time.Now()
		if now.Sub(lastUpdate) > 500*time.Millisecond {
			elapsed := now.Sub(startTime)
			text := accumulated.String()
			words := len(strings.Fields(text))
			fmt.Fprintf(os.Stderr, "\n[Progress: %d chunks, %d words, %.2fs elapsed]\n", chunkCount, words, elapsed.Seconds())
			lastUpdate = now
		}

		// Check metadata if available (example of accessing provider-specific metadata)
		if chunk.Metadata != nil {
			if tokens, ok := chunk.Metadata["tokens"].(int); ok {
				_ = tokens // tokens available in metadata if provided by model
			}
		}

		// Stream is complete
		if chunk.Done {
			elapsed := time.Since(startTime)
			fmt.Fprintf(os.Stderr,
				"\n[Stream complete: %d chunks, %d words, %.2fs total]\n",
				chunkCount, len(strings.Fields(accumulated.String())), elapsed.Seconds())
		}

		return nil
	}

	response, err := cw.CallModelStreaming(ctx, callback)
	if err != nil {
		log.Fatalf("Failed to call model: %v", err)
	}

	fmt.Printf("\n\n[Final response length: %d characters]\n", len(response))
}

// toolCallStreamingExample demonstrates streaming with tool calls.
func toolCallStreamingExample(ctx context.Context) {
	// reads GOOGLE_GENAI_API_KEY or GEMINI_API_KEY
	model, err := contextwindow.NewGeminiModel(contextwindow.ModelGemini20Flash)
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	db, err := contextwindow.NewContextDB(":memory:")
	if err != nil {
		log.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	cw, err := contextwindow.NewContextWindow(db, model, "")
	if err != nil {
		log.Fatalf("Failed to create context window: %v", err)
	}
	defer cw.Close()

	// Add a tool for getting the current time
	cw.AddTool(
		contextwindow.NewTool("get_current_time", "Gets the current time in a specified timezone").
			AddStringParameter("timezone", "The timezone (e.g., 'America/New_York', 'UTC')", false),
		contextwindow.ToolRunnerFunc(func(ctx context.Context, args json.RawMessage) (string, error) {
			var params struct {
				Timezone string `json:"timezone"`
			}
			if err := json.Unmarshal(args, &params); err != nil {
				return "", err
			}

			loc := time.UTC
			if params.Timezone != "" {
				var err error
				loc, err = time.LoadLocation(params.Timezone)
				if err != nil {
					return "", fmt.Errorf("invalid timezone: %w", err)
				}
			}

			return time.Now().In(loc).Format(time.RFC3339), nil
		}),
	)

	// Add middleware to track tool calls
	cw.AddMiddleware(&toolCallMiddleware{})

	if err := cw.AddPrompt("What time is it in New York? Use the get_current_time tool."); err != nil {
		log.Fatalf("Failed to add prompt: %v", err)
	}

	var accumulated strings.Builder

	// Streaming callback that handles tool calls
	callback := func(chunk contextwindow.StreamChunk) error {
		// Check if this chunk is related to a tool call
		if chunk.Metadata != nil {
			if toolName, ok := chunk.Metadata["tool_call"].(string); ok {
				fmt.Fprintf(os.Stderr, "\n[Tool call detected: %s]\n", toolName)
			}
		}

		// Print deltas
		if chunk.Delta != "" {
			accumulated.WriteString(chunk.Delta)
			fmt.Print(chunk.Delta)
		}

		// Handle errors
		if chunk.Error != nil {
			return fmt.Errorf("streaming error: %w", chunk.Error)
		}

		// Stream complete
		if chunk.Done {
			fmt.Fprintf(os.Stderr, "\n[Stream complete]\n")
		}

		return nil
	}

	response, err := cw.CallModelStreaming(ctx, callback)
	if err != nil {
		log.Fatalf("Failed to call model: %v", err)
	}

	fmt.Printf("\n\n[Final response: %s]\n", response)
}

// toolCallMiddleware implements Middleware to track tool calls during streaming.
type toolCallMiddleware struct{}

func (m *toolCallMiddleware) OnToolCall(ctx context.Context, name, args string) {
	fmt.Fprintf(os.Stderr, "[Middleware] Tool called: %s with args: %s\n", name, args)
}

func (m *toolCallMiddleware) OnToolResult(ctx context.Context, name, result string, err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "[Middleware] Tool %s returned error: %v\n", name, err)
	} else {
		fmt.Fprintf(os.Stderr, "[Middleware] Tool %s returned: %s\n", name, result)
	}
}
