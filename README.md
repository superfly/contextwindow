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

	"github.com/superfly/contextwindow"
	"github.com/openai/openai-go/v2/shared"
)

func main() {
	ctx := context.Background()

        // reads OPENAI_API_KEY
        model, err := NewOpenAIResponsesModel(shared.ResponsesModel4o)
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	// Initialize the ContextWindow
	// This creates a new conversation history, backed by an in-memory SQLite DB.
	// For persistence, provide a file path: contextwindow.New(model, summarizerModel, "/path/to/your.db")
	cw, err := contextwindow.New(nil, model, "")
	if err != nil {
		log.Fatalf("Failed to create context window: %v", err)
	}
	defer cw.Close()

        if err := cw.AddPrompt(ctx, "how's the weather over there?")
	if err := cw.AddPrompt(ctx, prompt); err != nil {
		log.Fatalf("Failed to add prompt: %v", err)
	}

	// Call the model to get a response
	response, err := cw.CallModel(ctx)
	if err != nil {
		log.Fatalf("Failed to call model: %v", err)
	}
	fmt.Printf("<- Model: %s\n", response)
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

## Maturity

This is alpha-quality code. Happy to get feedback or PRs or whatever. Publishing this more
as a statement than as an endorsement of the code itself. :)
