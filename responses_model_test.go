package contextwindow

import (
	"context"
	"os"
	"testing"

	"github.com/openai/openai-go/v2/option"
	"github.com/openai/openai-go/v2/packages/param"
	"github.com/openai/openai-go/v2/responses"
	"github.com/openai/openai-go/v2/shared"
	"github.com/stretchr/testify/assert"
)

func TestStreamingAPI(t *testing.T) {
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		t.Skipf("no OPENAI_API_KEY set")
		return
	}

	if os.Getenv("RUN_STREAMING_TEST") == "" {
		t.Skipf("not running streaming test; WIP functionality")
	}

	params := responses.ResponseNewParams{
		Input: responses.ResponseNewParamsInputUnion{
			OfString: param.NewOpt("write a simple go program that applies the double-angle formula to simplify a trigonometry expression"),
			//			OfString: param.NewOpt("-sin^2 + cos^2 => -1, right? if not, break it down step by step for me"),
		},
		Model: ResponsesModelO4Mini,
		Reasoning: shared.ReasoningParam{
			Effort: shared.ReasoningEffortMedium,
		},
	}

	rsvc := responses.NewResponseService(option.WithAPIKey(key),
		option.WithBaseURL("https://api.openai.com/v1/"))
	stream := rsvc.NewStreaming(context.TODO(), params)

	for stream.Next() {
		item := stream.Current()
		// spew.Dump(item)

		t.Logf("%s", item.Type)
		//if item.Delta != "" {
		//	t.Logf("%s", item.Delta)
		//}
		if item.Text != "" {
			t.Logf("%s", item.Text)
		}
		//if len(item.Response.Output) > 0 && item.Response.Output[0].Type == "reasoning" {
		//	t.Logf("%s", item.Response.Output[0].Content[0].Text)
		//}
	}

	if stream.Err() != nil {
		t.Fatalf("%v", stream.Err())
	}
}

func TestNewOpenAIResponsesModel(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "test-key")

	model, err := NewOpenAIResponsesModel(shared.ResponsesModelO1Pro)
	assert.NoError(t, err)
	assert.NotNil(t, model)
	assert.Equal(t, 128_000, model.MaxTokens())
}

func TestNewOpenAIResponsesModel_NoAPIKey(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")

	_, err := NewOpenAIResponsesModel(shared.ResponsesModelO1Pro)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "OPENAI_API_KEY not set")
}

func TestOpenAIResponsesModel_ConvertRecordsToInput(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "test-key")

	model, err := NewOpenAIResponsesModel(shared.ResponsesModelO1Pro)
	assert.NoError(t, err)

	inputs := []Record{
		{Source: Prompt, Content: "Hello"},
		{Source: ModelResp, Content: "Hi there!"},
		{Source: ToolCall, Content: "search(\"test\")"},
		{Source: ToolOutput, Content: "search results"},
	}

	result := model.convertRecordsToInput(inputs)
	expected := "User: Hello\nAssistant: Hi there!\nTool Call: search(\"test\")\nTool Output: search results"
	assert.Equal(t, expected, result)
}

func TestOpenAIResponsesModel_SystemPrompt(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}

	db, err := NewContextDB(":memory:")
	assert.NoError(t, err)
	defer db.Close()

	model, err := NewOpenAIResponsesModel(ResponsesModel4o)
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

func TestCanUseServerSideThreading(t *testing.T) {
	model := &OpenAIResponsesModel{}

	// Test case 1: Clean conversation with response IDs - should work
	responseID := "resp_123"
	cleanInputs := []Record{
		{Source: SystemPrompt, Content: "System message"},
		{Source: Prompt, Content: "Hello"},
		{Source: ModelResp, Content: "Hi there", ResponseID: &responseID},
		{Source: Prompt, Content: "How are you?"},
	}
	assert.True(t, model.canUseServerSideThreading(cleanInputs))

	// Test case 2: Has tool calls - should not work
	toolInputs := []Record{
		{Source: Prompt, Content: "Hello"},
		{Source: ModelResp, Content: "Hi there", ResponseID: &responseID},
		{Source: ToolCall, Content: "some_tool({})"},
		{Source: ToolOutput, Content: "tool output"},
	}
	assert.False(t, model.canUseServerSideThreading(toolInputs))

	// Test case 3: Missing response ID - should not work
	missingIDInputs := []Record{
		{Source: Prompt, Content: "Hello"},
		{Source: ModelResp, Content: "Hi there", ResponseID: nil},
		{Source: Prompt, Content: "How are you?"},
	}
	assert.False(t, model.canUseServerSideThreading(missingIDInputs))

	// Test case 4: Mixed response ID state - should not work
	responseID2 := "resp_456"
	mixedInputs := []Record{
		{Source: Prompt, Content: "Hello"},
		{Source: ModelResp, Content: "Hi there", ResponseID: &responseID},
		{Source: Prompt, Content: "How are you?"},
		{Source: ModelResp, Content: "Fine", ResponseID: nil},
		{Source: Prompt, Content: "Good"},
		{Source: ModelResp, Content: "Yes", ResponseID: &responseID2},
	}
	assert.False(t, model.canUseServerSideThreading(mixedInputs))

	// Test case 5: No model responses - should work
	noResponseInputs := []Record{
		{Source: SystemPrompt, Content: "System message"},
		{Source: Prompt, Content: "Hello"},
	}
	assert.True(t, model.canUseServerSideThreading(noResponseInputs))
}

func TestGetLatestPrompt(t *testing.T) {
	model := &OpenAIResponsesModel{}

	// Test with multiple prompts
	inputs := []Record{
		{Source: SystemPrompt, Content: "System message"},
		{Source: Prompt, Content: "First prompt"},
		{Source: ModelResp, Content: "First response"},
		{Source: Prompt, Content: "Second prompt"},
		{Source: ModelResp, Content: "Second response"},
		{Source: Prompt, Content: "Latest prompt"},
	}

	result := model.getLatestPrompt(inputs)
	assert.Equal(t, "User: Latest prompt", result)

	// Test with no prompts
	noPromptInputs := []Record{
		{Source: SystemPrompt, Content: "System message"},
		{Source: ModelResp, Content: "Response"},
	}

	result = model.getLatestPrompt(noPromptInputs)
	assert.Equal(t, "", result)
}

func TestExtractLatestPromptFromHistory(t *testing.T) {
	model := &OpenAIResponsesModel{}

	// Test with multiple user prompts in history
	fullHistory := `System: You are a helpful assistant
User: Hello
Assistant: Hi there
User: How are you?
Assistant: I'm doing well
User: What's the weather?`

	result := model.extractLatestPromptFromHistory(fullHistory)
	assert.Equal(t, "User: What's the weather?", result)

	// Test with no user prompts
	noUserHistory := `System: You are a helpful assistant
Assistant: Hi there
Assistant: I'm doing well`

	result = model.extractLatestPromptFromHistory(noUserHistory)
	assert.Equal(t, "", result)

	// Test with user prompt mixed with tool calls
	complexHistory := `User: Hello
Assistant: Hi
Tool Call: get_weather({})
Tool Output: Sunny
User: Thanks`

	result = model.extractLatestPromptFromHistory(complexHistory)
	assert.Equal(t, "User: Thanks", result)
}
