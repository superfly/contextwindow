package contextwindow

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"google.golang.org/genai"
)

func TestGeminiModel_HelloWorld(t *testing.T) {
	if os.Getenv("GOOGLE_GENAI_API_KEY") == "" && os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("set GOOGLE_GENAI_API_KEY or GEMINI_API_KEY to run integration test")
	}
	m, err := NewGeminiModel(ModelGemini20Flash)
	if err != nil {
		t.Fatalf("NewGeminiModel: %v", err)
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

func TestGeminiModel_ToolCall(t *testing.T) {
	if os.Getenv("GOOGLE_GENAI_API_KEY") == "" && os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("set GOOGLE_GENAI_API_KEY or GEMINI_API_KEY to run integration test")
	}
	m, err := NewGeminiModel(ModelGemini20Flash)
	if err != nil {
		t.Fatalf("NewGeminiModel: %v", err)
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

	lsTool := &genai.FunctionDeclaration{
		Name:        "ls",
		Description: "list files in a directory",
		Parameters: &genai.Schema{
			Type:       genai.TypeObject,
			Properties: map[string]*genai.Schema{},
		},
	}

	err = cw.RegisterTool("ls", lsTool, ToolRunnerFunc(func(ctx context.Context, args json.RawMessage) (string, error) {
		return `{"files": ["go.mod", "spiderman.txt", "batman.txt"]}`, nil
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

func TestGeminiModel_SystemPrompt(t *testing.T) {
	if os.Getenv("GOOGLE_GENAI_API_KEY") == "" && os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("set GOOGLE_GENAI_API_KEY or GEMINI_API_KEY to run integration test")
	}

	db, err := NewContextDB(":memory:")
	assert.NoError(t, err)
	defer db.Close()

	model, err := NewGeminiModel(ModelGemini20Flash)
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

func TestNewGeminiModel_NoAPIKey(t *testing.T) {
	// Save current env vars
	oldKey1 := os.Getenv("GOOGLE_GENAI_API_KEY")
	oldKey2 := os.Getenv("GEMINI_API_KEY")
	defer func() {
		os.Setenv("GOOGLE_GENAI_API_KEY", oldKey1)
		os.Setenv("GEMINI_API_KEY", oldKey2)
	}()

	// Unset both keys
	os.Unsetenv("GOOGLE_GENAI_API_KEY")
	os.Unsetenv("GEMINI_API_KEY")

	_, err := NewGeminiModel(ModelGemini20Flash)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "GOOGLE_GENAI_API_KEY or GEMINI_API_KEY not set")
}

func TestGeminiModel_ToolBuilder(t *testing.T) {
	if os.Getenv("GOOGLE_GENAI_API_KEY") == "" && os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("set GOOGLE_GENAI_API_KEY or GEMINI_API_KEY to run integration test")
	}

	db, err := NewContextDB(":memory:")
	assert.NoError(t, err)
	defer db.Close()

	model, err := NewGeminiModel(ModelGemini20Flash)
	assert.NoError(t, err)

	cw, err := NewContextWindow(db, model, "test")
	assert.NoError(t, err)

	// Use ToolBuilder to create a tool
	calcTool := NewTool("calculate", "Performs a simple calculation").
		AddStringParameter("operation", "The operation to perform (add, subtract, multiply, divide)", true).
		AddNumberParameter("x", "First number", true).
		AddNumberParameter("y", "Second number", true)

	err = cw.AddTool(calcTool, ToolRunnerFunc(func(ctx context.Context, args json.RawMessage) (string, error) {
		var params struct {
			Operation string  `json:"operation"`
			X         float64 `json:"x"`
			Y         float64 `json:"y"`
		}
		if err := json.Unmarshal(args, &params); err != nil {
			return "", err
		}
		var result float64
		switch params.Operation {
		case "add":
			result = params.X + params.Y
		case "multiply":
			result = params.X * params.Y
		default:
			result = 0
		}
		data, err := json.Marshal(map[string]float64{"result": result})
		if err != nil {
			return "", err
		}
		return string(data), nil
	}))
	assert.NoError(t, err)

	err = cw.AddPrompt("Use the calculate tool to multiply 7 and 8")
	assert.NoError(t, err)

	resp, err := cw.CallModel(context.Background())
	assert.NoError(t, err)
	assert.Contains(t, resp, "56")
}

func TestAllGeminiModels_BasicCall(t *testing.T) {
	if os.Getenv("GOOGLE_GENAI_API_KEY") == "" && os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("set GOOGLE_GENAI_API_KEY or GEMINI_API_KEY to run integration test")
	}

	for _, model := range AllGeminiModels {
		t.Run(model, func(t *testing.T) {
			m, err := NewGeminiModel(model)
			if err != nil {
				t.Fatalf("NewGeminiModel(%s): %v", model, err)
			}

			inputs := []Record{
				{Source: Prompt, Content: "Say 'test passed' and nothing else"},
			}

			reply, tokensUsed, err := m.Call(context.Background(), inputs)
			if err != nil {
				t.Fatalf("Call(%s): %v", model, err)
			}

			assert.Greater(t, len(reply), 0, "expected non-empty reply from %s", model)
			assert.Greater(t, tokensUsed, 0, "expected token usage from %s", model)

			// Verify we got a model response
			hasModelResponse := false
			for _, rec := range reply {
				if rec.Source == ModelResp {
					hasModelResponse = true
					assert.NotEmpty(t, rec.Content, "model %s returned empty content", model)
					break
				}
			}
			assert.True(t, hasModelResponse, "no ModelResp record found for %s", model)

			t.Logf("✓ Model %s: %d tokens used, response: %.50s...", model, tokensUsed, reply[len(reply)-1].Content)
		})
	}
}
