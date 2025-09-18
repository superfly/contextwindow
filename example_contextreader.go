package main

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/superfly/contextwindow"
)

// This example demonstrates the ContextReader for thread-safe access
func main() {
	// Create database and context window
	db, err := contextwindow.NewContextDB(":memory:")
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	// For this example, we'll use a simple mock model
	model := &ExampleModel{}
	cw, err := contextwindow.NewContextWindow(db, model, "demo")
	if err != nil {
		log.Fatal(err)
	}

	// Add some initial data
	cw.SetSystemPrompt("You are a helpful assistant")
	cw.AddPrompt("What is the weather like?")
	cw.AddPrompt("Tell me about Go programming")

	// Create a reader for thread-safe access
	reader := cw.Reader()

	fmt.Println("=== ContextReader Demo ===")

	// Simulate UI goroutines that need read access
	var wg sync.WaitGroup

	// UI updater goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 5; i++ {
			time.Sleep(100 * time.Millisecond)

			usage, _ := reader.TokenUsage()
			fmt.Printf("UI Update %d: Context=%s, Live=%d tokens, Total=%d tokens\n",
				i+1, reader.GetCurrentContext(), usage.Live, usage.Total)
		}
	}()

	// Metrics collector goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 3; i++ {
			time.Sleep(150 * time.Millisecond)

			contexts, _ := reader.ListContexts()
			liveTokens, _ := reader.LiveTokens()
			fmt.Printf("Metrics %d: Found %d contexts, %d live tokens\n",
				i+1, len(contexts), liveTokens)
		}
	}()

	// Meanwhile, main thread continues to modify the context
	go func() {
		time.Sleep(50 * time.Millisecond)
		cw.AddPrompt("Another question")

		time.Sleep(200 * time.Millisecond)
		cw.SetMaxTokens(8192)

		time.Sleep(200 * time.Millisecond)
		cw.AddPrompt("Final question")
	}()

	// Wait for all goroutines to complete
	wg.Wait()

	// Final state
	finalRecords, _ := reader.LiveRecords()
	fmt.Printf("\nFinal state: %d live records\n", len(finalRecords))

	for i, record := range finalRecords {
		recordType := "Unknown"
		switch record.Source {
		case contextwindow.SystemPrompt:
			recordType = "System"
		case contextwindow.Prompt:
			recordType = "User"
		case contextwindow.ModelResp:
			recordType = "Assistant"
		}
		fmt.Printf("  %d. [%s] %s\n", i+1, recordType, record.Content)
	}
}

// ExampleModel is a simple mock model for demonstration
type ExampleModel struct{}

func (m *ExampleModel) Call(ctx context.Context, inputs []contextwindow.Record) ([]contextwindow.Record, int, error) {
	return []contextwindow.Record{
		{
			Source:    contextwindow.ModelResp,
			Content:   "This is a mock response",
			Live:      true,
			EstTokens: 10,
		},
	}, 25, nil
}