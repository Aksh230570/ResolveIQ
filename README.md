# ResolveIQ

### A Historically Grounded and Risk-Aware Customer Support Agent

ResolveIQ is an AI-powered customer-support system designed to:

- Classify customer-support intents.
- Reconstruct relevant conversation context.
- Retrieve similar historical support cases.
- Draft evidence-grounded responses.
- Decide whether a request should be auto-handled or escalated to a human.
- Provide a clear reason for every escalation decision.

## Current Dataset

This project uses the Kaggle Customer Support on Twitter dataset.

The initial brand selected for experimentation is **AmazonHelp**.

The raw dataset is not included in this repository because of its large size.

## Project Goals

1. Build reliable intent-classification baselines.
2. Create a historical support-case retrieval system.
3. Generate grounded customer-support replies.
4. Develop a risk-aware escalation policy.
5. Evaluate the system using both automated metrics and human review.

## Project Status

- [x] Dataset exploration
- [x] Brand discovery
- [x] GitHub repository setup
- [ ] AmazonHelp data extraction
- [ ] Conversation reconstruction
- [ ] Golden evaluation dataset
- [ ] Intent-classification baseline
- [ ] Historical retrieval
- [ ] Reply generation
- [ ] Escalation policy
- [ ] Evaluation harness
- [ ] Demo interface

## Local Setup

```bash
python -m venv .venv
```
