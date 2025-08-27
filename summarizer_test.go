package contextwindow

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
)

type mockSummarizer struct {
	summaryText string
	tokensUsed  int
}

func (m *mockSummarizer) Call(
	ctx context.Context,
	inputs []Record,
) ([]Record, int, error) {
	return []Record{
		{
			Source:  ModelResp,
			Content: m.summaryText,
			Live:    true,
		},
	}, m.tokensUsed, nil
}

func setupTestDB(t *testing.T) *ContextWindow {
	path := filepath.Join(t.TempDir(), "cw.db")
	db, err := NewContextDB(path)
	assert.NoError(t, err)

	cw, err := NewContextWindow(db, &dummyModel{}, "")
	assert.NoError(t, err)
	return cw
}

func TestSummarizeLiveContext(t *testing.T) {
	cw := setupTestDB(t)
	defer cw.Close()

	summarizer := &mockSummarizer{
		summaryText: "This is a test summary",
		tokensUsed:  10,
	}
	cw.SetSummarizer(summarizer)

	err := cw.AddPrompt("Hello world")
	assert.NoError(t, err)

	err = cw.AddPrompt("How are you?")
	assert.NoError(t, err)

	err = cw.AddPrompt("Tell me about Go")
	assert.NoError(t, err)

	liveRecords, err := cw.LiveRecords()
	assert.NoError(t, err)
	assert.Len(t, liveRecords, 3)

	result, err := cw.SummarizeLiveContext(context.Background())
	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, "This is a test summary", result.Summary)
	assert.Len(t, result.Replaced, 3)
	assert.True(t, result.OrigCount > result.SummaryCount)
}

func TestSummarizeLiveContext_NoSummarizer(t *testing.T) {
	cw := setupTestDB(t)
	defer cw.Close()

	err := cw.AddPrompt("Hello world")
	assert.NoError(t, err)

	result, err := cw.SummarizeLiveContext(context.Background())
	assert.Error(t, err)
	assert.Nil(t, result)
	assert.Contains(t, err.Error(), "no summarizer configured")
}

func TestSummarizeLiveContext_NoLiveRecords(t *testing.T) {
	cw := setupTestDB(t)
	defer cw.Close()

	summarizer := &mockSummarizer{
		summaryText: "This is a test summary",
		tokensUsed:  10,
	}
	cw.SetSummarizer(summarizer)

	result, err := cw.SummarizeLiveContext(context.Background())
	assert.Error(t, err)
	assert.Nil(t, result)
	assert.Contains(t, err.Error(), "no live records to summarize")
}

func TestAcceptSummary(t *testing.T) {
	cw := setupTestDB(t)
	defer cw.Close()

	summarizer := &mockSummarizer{
		summaryText: "This is a test summary",
		tokensUsed:  10,
	}
	cw.SetSummarizer(summarizer)

	err := cw.AddPrompt("Hello world")
	assert.NoError(t, err)

	err = cw.AddPrompt("How are you?")
	assert.NoError(t, err)

	liveRecordsBefore, err := cw.LiveRecords()
	assert.NoError(t, err)
	assert.Len(t, liveRecordsBefore, 2)

	result, err := cw.SummarizeLiveContext(context.Background())
	assert.NoError(t, err)

	err = cw.AcceptSummary(result)
	assert.NoError(t, err)

	liveRecordsAfter, err := cw.LiveRecords()
	assert.NoError(t, err)
	assert.Len(t, liveRecordsAfter, 1)
	assert.Equal(t, "This is a test summary", liveRecordsAfter[0].Content)
	assert.Equal(t, ModelResp, liveRecordsAfter[0].Source)
	assert.True(t, liveRecordsAfter[0].Live)
}

func TestRejectSummary(t *testing.T) {
	cw := setupTestDB(t)
	defer cw.Close()

	summarizer := &mockSummarizer{
		summaryText: "This is a test summary",
		tokensUsed:  10,
	}
	cw.SetSummarizer(summarizer)

	err := cw.AddPrompt("Hello world")
	assert.NoError(t, err)

	err = cw.AddPrompt("How are you?")
	assert.NoError(t, err)

	liveRecordsBefore, err := cw.LiveRecords()
	assert.NoError(t, err)
	assert.Len(t, liveRecordsBefore, 2)

	result, err := cw.SummarizeLiveContext(context.Background())
	assert.NoError(t, err)

	cw.RejectSummary(result)

	liveRecordsAfter, err := cw.LiveRecords()
	assert.NoError(t, err)
	assert.Len(t, liveRecordsAfter, 2)
}

func TestDefaultSummarizerPrompt(t *testing.T) {
	assert.NotEmpty(t, defaultSummarizerPrompt)
	assert.Contains(t, defaultSummarizerPrompt, "summarizing")
	assert.Contains(t, defaultSummarizerPrompt, "concision")
}

func TestSetSummarizerPrompt(t *testing.T) {
	cw := setupTestDB(t)
	defer cw.Close()

	customPrompt := "Custom summarization prompt for testing"
	cw.SetSummarizerPrompt(customPrompt)

	summarizer := &mockSummarizer{
		summaryText: "Custom summary",
		tokensUsed:  15,
	}
	cw.SetSummarizer(summarizer)

	err := cw.AddPrompt("Test message")
	assert.NoError(t, err)

	result, err := cw.SummarizeLiveContext(context.Background())
	assert.NoError(t, err)
	assert.Equal(t, "Custom summary", result.Summary)
}

func TestDefaultPromptWhenNotSet(t *testing.T) {
	cw := setupTestDB(t)
	defer cw.Close()

	summarizer := &mockSummarizerWithInputCapture{
		summaryText: "Default summary",
		tokensUsed:  12,
	}
	cw.SetSummarizer(summarizer)

	err := cw.AddPrompt("Test message")
	assert.NoError(t, err)

	result, err := cw.SummarizeLiveContext(context.Background())
	assert.NoError(t, err)
	assert.Equal(t, "Default summary", result.Summary)

	assert.Len(t, summarizer.lastInputs, 2)
	assert.Equal(t, Prompt, summarizer.lastInputs[0].Source)
	assert.Equal(t, defaultSummarizerPrompt, summarizer.lastInputs[0].Content)
}

func TestCustomPromptUsed(t *testing.T) {
	cw := setupTestDB(t)
	defer cw.Close()

	customPrompt := "Please provide a very brief summary"
	cw.SetSummarizerPrompt(customPrompt)

	summarizer := &mockSummarizerWithInputCapture{
		summaryText: "Brief summary",
		tokensUsed:  8,
	}
	cw.SetSummarizer(summarizer)

	err := cw.AddPrompt("Test message for custom prompt")
	assert.NoError(t, err)

	result, err := cw.SummarizeLiveContext(context.Background())
	assert.NoError(t, err)
	assert.Equal(t, "Brief summary", result.Summary)

	assert.Len(t, summarizer.lastInputs, 2)
	assert.Equal(t, Prompt, summarizer.lastInputs[0].Source)
	assert.Equal(t, customPrompt, summarizer.lastInputs[0].Content)
}

type mockSummarizerWithInputCapture struct {
	summaryText string
	tokensUsed  int
	lastInputs  []Record
}

func (m *mockSummarizerWithInputCapture) Call(
	ctx context.Context,
	inputs []Record,
) ([]Record, int, error) {
	m.lastInputs = make([]Record, len(inputs))
	copy(m.lastInputs, inputs)
	return []Record{
		{
			Source:  ModelResp,
			Content: m.summaryText,
			Live:    true,
		},
	}, m.tokensUsed, nil
}
