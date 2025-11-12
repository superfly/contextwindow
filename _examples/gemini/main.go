package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	"github.com/superfly/contextwindow"
)

var prompt = `tell me the weather in a Louisville, CO, USA`

func main() {
	ctx := context.Background()

	// reads GOOGLE_GENAI_API_KEY or GEMINI_API_KEY
	model, err := contextwindow.NewGeminiModel(contextwindow.ModelGemini25FlashLite)
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

	cw.AddTool(
		contextwindow.NewTool("get_weather", "Gets the current weather for the given city").
			AddStringParameter("city", "The city to get the weather for", true).
			AddStringParameter("state", "The country of the city", false).
			AddStringParameter("country", "The country of the city", true),
		weatherTool{},
	)

	cw.AddMiddleware(middleware{})

	// Call the model to get a response
	response, err := cw.CallModel(ctx)
	if err != nil {
		log.Fatalf("Failed to call model: %v", err)
	}
	fmt.Printf("<- Model: %s\n", response)
}

type weatherTool struct{}

func (w weatherTool) Run(ctx context.Context, args json.RawMessage) (string, error) {
	var p struct {
		City    string `json:"city"`
		State   string `json:"state"`
		Country string `json:"country"`
	}
	if err := json.Unmarshal(args, &p); err != nil {
		return "", err
	}

	return fmt.Sprintf("It's sunny and 75 degrees in %s %s, %s", p.City, p.State, p.Country), nil
}

type middleware struct{}

func (m middleware) OnToolCall(ctx context.Context, name, args string) {
	fmt.Printf("<- Calling tool: %s with args: %s\n", name, args)
}

func (m middleware) OnToolResult(ctx context.Context, name, result string, err error) {
	if err != nil {
		fmt.Printf("-> Tool %s returned error: %v\n", name, err)
	} else {
		fmt.Printf("-> Tool %s returned result: %s\n", name, result)
	}
}
