package contextwindow

import (
	"context"
	_ "embed"
	"fmt"
)

//go:embed default_summarize.md
var defaultSummarizerPrompt string

type Summarizer interface {
	Model
}

type SummaryResult struct {
	Summary      string
	Replaced     []Record
	OrigCount    int
	SummaryCount int
}

func (cw *ContextWindow) SetSummarizer(summarizer Summarizer) {
	cw.summarizer = summarizer
}

func (cw *ContextWindow) SetSummarizerPrompt(prompt string) {
	cw.summarizerPrompt = prompt
}

func (cw *ContextWindow) SummarizeLiveContext(ctx context.Context) (*SummaryResult, error) {
	return cw.SummarizeLiveContextInContext(ctx, cw.currentContext)
}

func (cw *ContextWindow) SummarizeLiveContextInContext(
	ctx context.Context,
	contextName string,
) (*SummaryResult, error) {
	if cw.summarizer == nil {
		return nil, fmt.Errorf("no summarizer configured")
	}

	contextID, err := getContextIDByName(cw.db, contextName)
	if err != nil {
		return nil, fmt.Errorf("summarize context: %w", err)
	}

	liveRecords, err := ListLiveRecords(cw.db, contextID)
	if err != nil {
		return nil, fmt.Errorf("get live records: %w", err)
	}

	if len(liveRecords) == 0 {
		return nil, fmt.Errorf("no live records to summarize")
	}

	origCount := 0
	for _, r := range liveRecords {
		origCount += r.EstTokens
	}

	prompt := cw.summarizerPrompt
	if prompt == "" {
		prompt = defaultSummarizerPrompt
	}

	summaryInput := append([]Record{
		{
			Source:    Prompt,
			Content:   prompt,
			Live:      false,
			ContextID: contextID,
		},
	}, liveRecords...)

	events, _, err := cw.summarizer.Call(ctx, summaryInput)
	if err != nil {
		return nil, fmt.Errorf("summarizer call failed: %w", err)
	}

	if len(events) == 0 {
		return nil, fmt.Errorf("summarizer returned no events")
	}

	summary := events[len(events)-1].Content

	return &SummaryResult{
		Summary:      summary,
		Replaced:     liveRecords,
		OrigCount:    origCount,
		SummaryCount: tokenCount(summary),
	}, nil
}

func (cw *ContextWindow) AcceptSummary(
	result *SummaryResult,
) error {
	return cw.AcceptSummaryInContext(result, cw.currentContext)
}

func (cw *ContextWindow) AcceptSummaryInContext(
	result *SummaryResult,
	contextName string,
) error {
	contextID, err := getContextIDByName(cw.db, contextName)
	if err != nil {
		return fmt.Errorf("accept summary: %w", err)
	}

	tx, err := cw.db.Begin()
	if err != nil {
		return fmt.Errorf("begin transaction: %w", err)
	}
	defer tx.Rollback()

	for _, record := range result.Replaced {
		if err := markRecordNotAlive(tx, record.ID); err != nil {
			return fmt.Errorf("mark record %d not alive: %w", record.ID, err)
		}
	}

	_, err = insertRecordTx(
		tx,
		contextID,
		ModelResp,
		result.Summary,
		true,
	)
	if err != nil {
		return fmt.Errorf("insert summary: %w", err)
	}

	return tx.Commit()
}

func (cw *ContextWindow) RejectSummary(
	result *SummaryResult,
) {
}
