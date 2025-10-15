# Arcade Flow Analyzer - Solution

## Overview
This solution analyzes Arcade flow recordings using OpenAI's multimodal APIs with:
- Vision AI analysis of VIDEO steps using surrounding image and thumbnail context
- Clean list and summary of user actions
- Image prompt generation
- Multithreaded multi-image generation with AI-powered selection of strongest image candidate
- LLM API call caching system for cost efficiency during development

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Set up API key
# Create secrets.yaml with: openai-key: "your-key"

# Run analysis
python generate_report.py