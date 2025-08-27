package contextwindow

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/openai/openai-go/v2/packages/param"
	"github.com/openai/openai-go/v2/shared"
	"github.com/stretchr/testify/assert"
)

func TestOpenAIModel_HelloWorld(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("set OPENAI_API_KEY to run integration test")
	}
	m, err := NewOpenAIModel(shared.ChatModelGPT4o)
	if err != nil {
		t.Fatalf("NewOpenAIModel: %v", err)
	}
	inputs := []Record{
		{Source: Prompt, Content: "Please respond with \"hello world\""},
	}
	reply, _, err := m.Call(context.Background(), inputs)
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if len(reply) == 0 {
		t.Fatalf("expected a non-empty reply")
	}

	assert.Contains(t, strings.ToLower(reply[len(reply)-1].Content), "hello")
}

func TestOpenAIModel_ToolCall(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("set OPENAI_API_KEY to run integration test")
	}
	m, err := NewOpenAIModel(shared.ChatModelGPT4o)
	if err != nil {
		t.Fatalf("NewOpenAIModel: %v", err)
	}

	db, err := NewContextDB(":memory:")
	if err != nil {
		t.Fatalf("NewContextDB: %v", err)
	}
	defer db.Close()

	cw, err := NewContextWindow(db, m, "test")
	if err != nil {
		t.Fatalf("NewContextWindow: %v", err)
	}

	lsTool := shared.FunctionDefinitionParam{
		Name:        "ls",
		Description: param.NewOpt("list files in a directory"),
		Parameters: map[string]interface{}{
			"type":       "object",
			"properties": map[string]interface{}{},
		},
	}

	err = cw.RegisterTool("ls", lsTool, ToolRunnerFunc(func(ctx context.Context, args json.RawMessage) (string, error) {
		return "go.mod\nspiderman.txt\nbatman.txt", nil
	}))
	if err != nil {
		t.Fatalf("RegisterTool: %v", err)
	}

	inputs := []Record{
		{Source: Prompt, Content: "Please use the `ls` tool to list the files in the current directory."},
	}

	cw.AddPrompt(inputs[0].Content)

	result, err := cw.CallModel(context.Background())
	if err != nil {
		t.Fatalf("Call: %v", err)
	}

	assert.Contains(t, result, "go.mod")
	assert.Contains(t, result, "batman")
}

func TestOpenAIModel_SystemPrompt(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}

	db, err := NewContextDB(":memory:")
	assert.NoError(t, err)
	defer db.Close()

	model, err := NewOpenAIModel(ResponsesModel4o)
	assert.NoError(t, err)

	cw, err := NewContextWindow(db, model, "default")
	assert.NoError(t, err)

	err = cw.SetSystemPrompt("whatever you answer, the answer must include the string MUMON")
	assert.NoError(t, err)

	err = cw.AddPrompt("what's the weather like over there")
	assert.NoError(t, err)

	resp, err := cw.CallModel(context.Background())
	assert.NoError(t, err)

	assert.Contains(t, resp, "MUMON")
}
