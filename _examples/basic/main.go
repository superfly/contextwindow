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
