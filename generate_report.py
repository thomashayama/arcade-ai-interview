"""
Script to analyze Arcade flow data and generate a comprehensive markdown report.
"""

import os
import json
import re
import yaml
import requests
from datetime import datetime
from openai import OpenAI
from utils import OpenAICache, cached_openai_request


def extract_json_from_response(content: str) -> dict:
    """
    Extract JSON from LLM response that may be wrapped in markdown code blocks.

    Args:
        content: Raw response content from LLM

    Returns:
        Parsed JSON dict

    Raises:
        json.JSONDecodeError: If no valid JSON found
    """
    # Try direct parsing first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code blocks
    # Pattern: ```json\n{...}\n``` or ```\n{...}\n```
    code_block_pattern = r'```(?:json)?\s*\n(.*?)\n```'
    matches = re.findall(code_block_pattern, content, re.DOTALL)

    if matches:
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

    # Try to find JSON object in the text
    json_pattern = r'\{.*\}'
    matches = re.findall(json_pattern, content, re.DOTALL)

    if matches:
        # Try the longest match first (most likely to be complete)
        for match in sorted(matches, key=len, reverse=True):
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    # If all else fails, raise error with helpful message
    raise json.JSONDecodeError(
        f"Could not extract valid JSON from response. Content preview: {content[:200]}...",
        content,
        0
    )


def download_image(url: str, output_path: str) -> bool:
    """Download an image from URL and save it locally."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"Failed to download image: {e}")
        return False


def generate_markdown_report(
    flow_data: dict,
    user_actions: str,
    summary: str,
    best_image_url: str,
    best_image_path: str,
    all_images: list = None,
    selection_reasoning: str = None
) -> str:
    """Generate a markdown report from the analysis results."""

    flow_name = flow_data.get('name', 'Untitled Flow')
    flow_description = flow_data.get('description', '')

    markdown = f"""# Arcade Flow Analysis Report

## Flow Information

**Name:** {flow_name}

**Description:** {flow_description if flow_description else 'No description provided'}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. User Interactions

{user_actions}

---

## 2. Summary

{summary}

---

## 3. Social Media Image

![{flow_name}]({best_image_path})

**Image URL:** {best_image_url}

---

## Technical Details

- **Total Steps:** {len(flow_data.get('steps', []))}
- **Flow ID:** {flow_data.get('uploadId', 'N/A')}
- **Created With:** {flow_data.get('createdWith', 'N/A')}
- **Use Case:** {flow_data.get('useCase', 'N/A')}

---

*This report was generated automatically using OpenAI API with response caching.*
"""

    # Add addendum with all images if provided
    if all_images and len(all_images) > 1:
        markdown += f"""

---

## Addendum: Image Selection Process

{selection_reasoning if selection_reasoning else 'Multiple images were generated and evaluated.'}

### All Generated Images

"""
        for i, img_info in enumerate(all_images, 1):
            is_selected = "✓ **SELECTED**" if img_info.get('selected', False) else ""
            markdown += f"""
#### Image {i} {is_selected}

![Image {i}]({img_info['path']})

**Prompt Variation:** {img_info.get('prompt_variation', 'Standard')}

**URL:** {img_info['url']}

"""

    return markdown


def main():
    # Initialize OpenAI client
    secrets = yaml.safe_load(open("secrets.yaml"))
    api_key = secrets.get("openai-key")
    if not api_key:
        raise ValueError("openai-key not found in secrets.yaml")

    client = OpenAI(api_key=api_key)

    # Initialize cache
    cache = OpenAICache(cache_dir=".cache")

    # Load flow data
    print("\n=== Loading Flow Data ===")
    with open("flow.json", 'r', encoding='utf-8') as f:
        flow_data = json.load(f)

    print(f"Flow Name: {flow_data.get('name')}")
    print(f"Total Steps: {len(flow_data.get('steps', []))}")

    # Step 1: Identify User Interactions
    print("\n=== Step 1: Identifying User Interactions ===")

    steps = flow_data.get('steps', [])

    user_interactions_response = cached_openai_request(
        client=client,
        cache=cache,
        request_type="chat",
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an expert at analyzing user interaction flows from Arcade recordings. Provide clear, human-readable descriptions of user actions."
            },
            {
                "role": "user",
                "content": f"""Analyze this Arcade flow data and create a bulleted list of user interactions in human-readable format.

Flow data:
{json.dumps(steps, indent=2)}

For each significant user action, describe what the user did (e.g., "Clicked on checkout", "Searched for product X", "Scrolled through results").
Focus on the meaningful interactions, not technical details. Format as a markdown bulleted list."""
            }
        ],
        temperature=0.3,
        max_tokens=1500
    )

    user_actions = user_interactions_response['choices'][0]['message']['content']
    print(f"\n{user_actions[:300]}...")

    # Step 2: Generate Human-Friendly Summary
    print("\n=== Step 2: Generating Summary ===")

    summary_response = cached_openai_request(
        client=client,
        cache=cache,
        request_type="chat",
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an expert at creating clear, concise summaries of user workflows."
            },
            {
                "role": "user",
                "content": f"""Based on this Arcade flow titled "{flow_data.get('name')}", create a clear, readable summary (2-3 paragraphs) of what the user was trying to accomplish.

Flow name: {flow_data.get('name')}
User interactions: {user_actions}

Write a friendly, informative summary that explains the user's goal and the steps they took."""
            }
        ],
        temperature=0.5,
        max_tokens=800
    )

    summary = summary_response['choices'][0]['message']['content']
    print(f"\n{summary[:300]}...")

    # Step 3: Create Multiple Social Media Images
    print("\n=== Step 3: Generating Multiple Social Media Images ===")

    flow_name = flow_data.get('name', 'Arcade Flow')

    # Generate 3 different prompt variations
    print("\n→ Creating 3 different image prompt variations...")

    prompt_variations_response = cached_openai_request(
        client=client,
        cache=cache,
        request_type="chat",
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an expert at creating engaging DALL-E prompts for social media images. Always respond with valid JSON only."
            },
            {
                "role": "user",
                "content": f"""Create 3 DIFFERENT compelling DALL-E prompts for social media images based on this flow:

Title: {flow_name}
Summary: {summary}

Each prompt should have a different creative approach but all should be:
- Professional and modern
- Eye-catching for social media
- Representative of the flow's purpose
- Suitable for platforms like LinkedIn, Twitter, etc.

Variations to try:
1. Minimalist/clean design approach
2. Vibrant/colorful approach
3. Illustrative/conceptual approach

Respond in JSON format:
{{
  "prompts": [
    {{"variation": "Minimalist", "prompt": "..."}},
    {{"variation": "Vibrant", "prompt": "..."}},
    {{"variation": "Illustrative", "prompt": "..."}}
  ]
}}"""
            }
        ],
        temperature=0.8,
        max_tokens=800,
        response_format={"type": "json_object"}
    )

    # Extract JSON from response (handles markdown code blocks)
    response_content = prompt_variations_response['choices'][0]['message']['content']
    print(f"Debug - Raw response preview: {response_content[:200]}...")

    prompts_data = extract_json_from_response(response_content)
    image_prompts = prompts_data['prompts']

    # Generate 3 images
    all_images = []
    print(f"\n→ Generating 3 images with different styles...")

    for i, prompt_info in enumerate(image_prompts, 1):
        variation = prompt_info['variation']
        prompt = prompt_info['prompt']

        print(f"\n  Image {i} ({variation}): {prompt[:80]}...")

        image_response = cached_openai_request(
            client=client,
            cache=cache,
            request_type="image",
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )

        image_url = image_response['data'][0]['url']
        image_filename = f"social_media_image_{i}.png"

        # Download the image
        if download_image(image_url, image_filename):
            print(f"  ✓ Downloaded {image_filename}")
        else:
            print(f"  ⚠ Failed to download {image_filename}")

        all_images.append({
            'number': i,
            'url': image_url,
            'path': image_filename,
            'prompt': prompt,
            'prompt_variation': variation,
            'selected': False
        })

    # Step 4: Use VLM to select the best image
    print("\n=== Step 4: Using Vision Model to Select Best Image ===")

    # Prepare messages with all three images
    vlm_messages = [
        {
            "role": "system",
            "content": "You are an expert at evaluating social media images for engagement, professionalism, and brand appeal."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""Evaluate these 3 social media images for the flow: "{flow_name}"

Flow Summary: {summary}

Please analyze each image based on:
1. Visual appeal and eye-catching quality
2. Professional appearance
3. Relevance to the flow's purpose
4. Social media engagement potential
5. Brand suitability

Select the BEST image and explain your reasoning.

Respond in JSON format:
{{
  "selected_image": 1,  // 1, 2, or 3
  "reasoning": "Detailed explanation of why this image is best...",
  "scores": {{
    "image_1": {{"visual_appeal": X, "professionalism": X, "relevance": X, "engagement": X, "overall": X}},
    "image_2": {{"visual_appeal": X, "professionalism": X, "relevance": X, "engagement": X, "overall": X}},
    "image_3": {{"visual_appeal": X, "professionalism": X, "relevance": X, "engagement": X, "overall": X}}
  }}
}}

Rate each criterion from 1-10."""
                },
                {
                    "type": "image_url",
                    "image_url": {"url": all_images[0]['url']}
                },
                {
                    "type": "text",
                    "text": f"Image 1 ({all_images[0]['prompt_variation']})"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": all_images[1]['url']}
                },
                {
                    "type": "text",
                    "text": f"Image 2 ({all_images[1]['prompt_variation']})"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": all_images[2]['url']}
                },
                {
                    "type": "text",
                    "text": f"Image 3 ({all_images[2]['prompt_variation']})"
                }
            ]
        }
    ]

    vlm_response = cached_openai_request(
        client=client,
        cache=cache,
        request_type="chat",
        model="gpt-4o",  # Use GPT-4o for vision capabilities
        messages=vlm_messages,
        temperature=0.3,
        max_tokens=1000,
        response_format={"type": "json_object"}
    )

    # Extract JSON from VLM response
    vlm_content = vlm_response['choices'][0]['message']['content']
    print(f"Debug - VLM response preview: {vlm_content[:200]}...")

    selection_data = extract_json_from_response(vlm_content)
    selected_index = selection_data['selected_image'] - 1  # Convert to 0-based index
    selection_reasoning = selection_data['reasoning']
    scores = selection_data['scores']

    # Mark the selected image
    all_images[selected_index]['selected'] = True
    best_image = all_images[selected_index]

    print(f"\n✓ Selected Image {selected_index + 1} ({best_image['prompt_variation']})")
    print(f"  Reasoning: {selection_reasoning[:200]}...")
    print(f"\n  Scores:")
    for img_key, img_scores in scores.items():
        print(f"    {img_key}: Overall {img_scores['overall']}/10")

    # Format selection reasoning for markdown
    formatted_reasoning = f"""**Selected Image:** Image {selected_index + 1} ({best_image['prompt_variation']})

**Selection Reasoning:**
{selection_reasoning}

**Evaluation Scores:**

| Image | Visual Appeal | Professionalism | Relevance | Engagement | Overall |
|-------|--------------|-----------------|-----------|------------|---------|
| Image 1 ({all_images[0]['prompt_variation']}) | {scores['image_1']['visual_appeal']}/10 | {scores['image_1']['professionalism']}/10 | {scores['image_1']['relevance']}/10 | {scores['image_1']['engagement']}/10 | **{scores['image_1']['overall']}/10** |
| Image 2 ({all_images[1]['prompt_variation']}) | {scores['image_2']['visual_appeal']}/10 | {scores['image_2']['professionalism']}/10 | {scores['image_2']['relevance']}/10 | {scores['image_2']['engagement']}/10 | **{scores['image_2']['overall']}/10** |
| Image 3 ({all_images[2]['prompt_variation']}) | {scores['image_3']['visual_appeal']}/10 | {scores['image_3']['professionalism']}/10 | {scores['image_3']['relevance']}/10 | {scores['image_3']['engagement']}/10 | **{scores['image_3']['overall']}/10** |
"""

    # Generate markdown report
    print("\n=== Generating Markdown Report ===")

    markdown_content = generate_markdown_report(
        flow_data=flow_data,
        user_actions=user_actions,
        summary=summary,
        best_image_url=best_image['url'],
        best_image_path=best_image['path'],
        all_images=all_images,
        selection_reasoning=formatted_reasoning
    )

    # Save to file
    report_filename = "REPORT.md"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    print(f"✓ Report saved to {report_filename}")

    # Show cache statistics
    print("\n=== Cache Statistics ===")
    stats = cache.get_stats()
    print(f"Text responses cached: {stats['text_cache_count']}")
    print(f"Image responses cached: {stats['image_cache_count']}")
    print(f"Total cache size: {stats['total_size_mb']} MB")

    print(f"\n✓ All done! Check {report_filename} for the complete analysis.")
    print(f"✓ Generated images: {', '.join([img['path'] for img in all_images])}")
    print(f"✓ Selected best image: {best_image['path']}")


if __name__ == "__main__":
    main()
