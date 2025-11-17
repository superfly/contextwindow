# ContextWindow: a low-level Go agentic LLM session library

[![GoDoc](https://godoc.org/github.com/superfly/contextwindow?status.svg)](https://godoc.org/github.com/superfly/contextwindow)

**ContextWindow** is a straightforward library for managing conversations with LLMs. It supports tool calls, token usage tracking, summary-based compression (with an arbitrary summary model), and automatically persists conversations in SQLite.

The impetus for this library is multi-context agents. Under the hood of an LLM agent, a "context window"
is simply a list of strings. LLM APIs generally don't have a notion of "sessions". As you build up a conversation, you simply send all prompts and responses to the LLM each time (we do support `previous_response_id`). This means there's no drama in having 2, 10, or 100 different "conversations" in an agent, and **ContextWindow** tries to make that straightforward to express.

## Simple usage

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/openai/openai-go/v2/shared"
	"github.com/superfly/contextwindow"
)

var prompt = `how's the weather over there?`

func main() {
	ctx := context.Background()

	// reads OPENAI_API_KEY
	model, err := contextwindow.NewOpenAIResponsesModel(shared.ChatModelGPT5Mini)
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	// Initialize the database (in-memory)
	// For persistence, use a file path instead of ":memory:"
	db, err := contextwindow.NewContextDB(":memory:")
	if err != nil {
		log.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	// Initialize the ContextWindow
	cw, err := contextwindow.NewContextWindow(db, model, "")
	if err != nil {
		log.Fatalf("Failed to create context window: %v", err)
	}
	defer cw.Close()

	if err := cw.AddPrompt(prompt); err != nil {
		log.Fatalf("Failed to add prompt: %v", err)
	}

	// Call the model to get a response
	response, err := cw.CallModel(ctx)
	if err != nil {
		log.Fatalf("Failed to call model: %v", err)
	}
	fmt.Printf("<- Model: %s\n", response)
}
```

## Tool calling

An LLM tool or function call is just a set of special JSON blobs the LLM is trained to parse
and emit instead of English text. In this library, they're simply function calls:

```go
lsTool := contextwindow.NewTool("list_files", `
  This tool lists files in the specified directory.
`).AddStringParameter("directory", "Directory to list", true)

cw.AddTool(lsTool, contextwindow.ToolRunnerFunc(func context.Context,
                                                args json.RawMessage) (string, error) {
  var treq struct {
    Dir string `json:"directory"`
  }
  json.Unmarshal(args, &treq)
  // actually run ls, or pretend to
  return "here\nare\nsome\nfiles.exe\n", nil
})
```

Arguments to function calls are passed in JSON messages; tool responses are passed as
string return values.

Attach different tools to different ContextWindow instances; that's kind of the point. If
one ContextWindow does something sensitive (like reading a database) and another reads untrusted
data (like incoming email addresses), check carefully before relaying information from the untrusted
side to the trusted side (for instance: instruct the untrusted side to provide responses in a
JSON message you can parse in your code before handing off).

## Streaming

ContextWindow supports streaming responses for real-time token display. Streaming allows you to
receive and display tokens as they arrive from the LLM provider, providing a better user experience
with faster perceived response times.

### Basic Streaming

The simplest streaming usage involves providing a callback function that receives chunks as they arrive:

```go
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
```

The callback receives `StreamChunk` objects containing:
- `Delta`: The incremental text/token content for this chunk
- `Done`: Whether the stream has completed
- `Metadata`: Provider-specific metadata (optional)
- `Error`: Any streaming error that occurred

### Callback Error Handling

If your callback returns a non-nil error, streaming will be stopped immediately. This allows you to
implement early cancellation:

```go
callback := func(chunk contextwindow.StreamChunk) error {
    if chunk.Error != nil {
        return chunk.Error
    }
    if !chunk.Done {
        fmt.Print(chunk.Delta)
    }
    // Return error to cancel stream early if needed
    return nil
}
```

### Fallback Behavior

If your model doesn't support streaming (doesn't implement `StreamingCapable` or `StreamingOptsCapable`),
`CallModelStreaming` automatically falls back to the buffered `CallModel` method. This ensures backward
compatibility - your code will work with both streaming and non-streaming models.

### Streaming with Options

You can use `CallModelStreamingWithOpts` to disable tools or pass other options:

```go
opts := contextwindow.CallModelOpts{
    DisableTools: true,
}

response, err := cw.CallModelStreamingWithOpts(ctx, opts, callback)
```

### Tool Calls During Streaming

Tool calls work seamlessly with streaming. When a tool is called during streaming, the tool execution
happens after the tool call is complete, and the tool result is streamed back to the callback. The
complete response (including tool results) is persisted after the stream finishes, just like with
non-streaming calls.

### Performance Characteristics

- **Latency**: Streaming provides faster time-to-first-token compared to buffered responses
- **Memory**: Streaming uses similar memory overhead as non-streaming (complete response is still accumulated)
- **Persistence**: The complete response is persisted after streaming completes, maintaining the same
  database consistency as non-streaming calls

### Provider-Specific Behaviors

Different providers handle streaming differently:

- **OpenAI**: Streams token deltas directly
- **Claude**: Uses event-based streaming (message_start, content_block_delta, etc.)
- **Gemini**: Streams content parts incrementally

The `StreamChunk.Metadata` field may contain provider-specific information. The library abstracts
these differences, so your callback code works the same across all providers.

See `_examples/streaming/main.go` for complete examples including progress tracking and tool calls.

## Context Management and Stats

For building table views or managing multiple conversations, you can efficiently get statistics
for any context without expensive operations:

```go
// Get all contexts
contexts, err := cw.ListContexts()
if err != nil {
    log.Fatalf("Failed to list contexts: %v", err)
}

// Get efficient stats for each context (no table scans)
for _, context := range contexts {
    stats, err := cw.GetContextStats(context)
    if err != nil {
        log.Printf("Failed to get stats for %s: %v", context.Name, err)
        continue
    }
    
    fmt.Printf("Context: %s\n", context.Name)
    fmt.Printf("  Created: %s\n", context.StartTime.Format("2006-01-02 15:04"))
    fmt.Printf("  Live tokens: %d\n", stats.LiveTokens)
    fmt.Printf("  Records: %d total, %d live\n", stats.TotalRecords, stats.LiveRecords)
    if stats.LastActivity != nil {
        fmt.Printf("  Last activity: %s\n", stats.LastActivity.Format("2006-01-02 15:04"))
    } else {
        fmt.Printf("  Last activity: never\n")
    }
}
```

The `GetContextStats` method uses a single aggregation query with existing indexes,
making it efficient even with many contexts and large conversation histories.

## Testing

This project uses Go build tags to separate unit tests from integration tests:

- **Unit tests**: Run with `go test` (default). These tests don't require API keys or make external API calls.
- **Integration tests**: Run with `go test -tags=integration`. These tests require API keys and make real API calls to LLM providers.

Integration tests are behind the `integration` build tag because they:
- Require API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_GENAI_API_KEY, etc.)
- Make real API calls that consume quota
- May be rate-limited or fail due to network issues
- Are slower and more expensive to run

For IDE support (VS Code, GoLand, etc.), configure your editor to use `-tags=integration` build flags so that integration test files are recognized and can be run individually.

## Maturity

This is alpha-quality code. Happy to get feedback or PRs or whatever. Publishing this more
as a statement than as an endorsement of the code itself. :)
