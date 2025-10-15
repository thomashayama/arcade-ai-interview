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
from utils import OpenAICache, cached_openai_request, download_image, generate_markdown_report, extract_json_from_response
from enhanced_video_analysis import create_user_interactions_with_videos

def get_surrounding_context(steps, video_index):
    """
    Get context from steps surrounding a VIDEO step.

    Returns dict with previous_action and next_action info.
    """
    context = {'previous_action': None, 'next_action': None}

    # Get previous IMAGE step
    if video_index > 0:
        prev_step = steps[video_index - 1]
        if prev_step.get('type') == 'IMAGE':
            click_ctx = prev_step.get('clickContext', {})
            context['previous_action'] = {
                'element': click_ctx.get('text', 'unknown'),
                'element_type': click_ctx.get('elementType', 'unknown'),
                'page_url': prev_step.get('pageContext', {}).get('url', '')
            }

    # Get next IMAGE step
    if video_index < len(steps) - 1:
        next_step = steps[video_index + 1]
        if next_step.get('type') == 'IMAGE':
            click_ctx = next_step.get('clickContext', {})
            context['next_action'] = {
                'element': click_ctx.get('text', 'unknown'),
                'element_type': click_ctx.get('elementType', 'unknown'),
                'page_url': next_step.get('pageContext', {}).get('url', '')
            }

    return context


def analyze_video_with_context(client, cache, step, context, captured_events):
    """
    Analyze a VIDEO step using vision AI with surrounding context.

    Returns a human-readable description of what happened in the video.
    """
    import base64

    thumbnail_url = step.get('videoThumbnailUrl')

    # Build comprehensive context from surrounding steps
    context_text = "Context:\n"
    if context['previous_action']:
        prev = context['previous_action']
        context_text += f"- Previous action: User clicked on '{prev['element']}' ({prev['element_type']})\n"
        if prev['page_url']:
            context_text += f"- Previous page: {prev['page_url']}\n"

    if context['next_action']:
        nxt = context['next_action']
        context_text += f"- Next action: User will click on '{nxt['element']}' ({nxt['element_type']})\n"
        if nxt['page_url']:
            context_text += f"- Next page: {nxt['page_url']}\n"

    # Summarize all captured events in the flow for general context
    events_summary = []
    for event in captured_events:
        event_type = event.get('type', 'unknown')
        if event_type not in [e['type'] for e in events_summary]:
            events_summary.append({'type': event_type})

    events_text = "Overall events in this flow include: "
    event_types = [e['type'] for e in events_summary]
    if 'typing' in event_types:
        events_text += "typing, "
    if 'scrolling' in event_types:
        events_text += "scrolling, "
    if 'click' in event_types:
        events_text += "clicking, "
    if 'dragging' in event_types:
        events_text += "dragging, "
    events_text = events_text.rstrip(', ')

    # Download thumbnail and convert to base64
    try:
        response = requests.get(thumbnail_url, timeout=10)
        response.raise_for_status()
        image_data = base64.b64encode(response.content).decode('utf-8')

        # Determine image type from URL or content
        if thumbnail_url.endswith('.png') or 'png' in thumbnail_url:
            image_type = 'png'
        elif thumbnail_url.endswith('.jpg') or thumbnail_url.endswith('.jpeg') or 'jpeg' in thumbnail_url:
            image_type = 'jpeg'
        else:
            image_type = 'png'  # Default

        image_url_data = f"data:image/{image_type};base64,{image_data}"

    except Exception as e:
        print(f"  Warning: Failed to download thumbnail, skipping vision analysis: {e}")
        # Fallback: describe based on context and events only
        if captured_events:
            event_types = [e.get('type') for e in captured_events]
            if 'typing' in event_types:
                return "Typed into a field"
            elif 'scrolling' in event_types:
                return "Scrolled the page"
            elif 'click' in event_types:
                return "Performed a click action"
        return "Performed an action"

    # Use vision model with context
    description_response = cached_openai_request(
        client=client,
        cache=cache,
        request_type="chat",
        model="gpt-4o",  # Vision-capable model
        messages=[
            {
                "role": "system",
                "content": "You are an expert at describing user actions in web interfaces. Be specific and concise."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Describe what the user is doing in this video segment.

{context_text}

{events_text}

Write ONE clear sentence describing the user's action (e.g., "Typed 'scooter' into the search bar", "Scrolled through the product results").
Respond with just the action description, no preamble."""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url_data,
                            "detail": "low"  # Use low detail for faster processing
                        }
                    }
                ]
            }
        ],
        temperature=0.2,
        max_tokens=100
    )
    return description_response['choices'][0]['message']['content'].strip()


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

    # Step 1: Identify User Interactions (with enriched video analysis)
    print("\n=== Step 1: Identifying User Interactions with Video Context ===")

    user_actions = create_user_interactions_with_videos(client, cache, flow_data)
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
                "content": "You are an expert at creating engaging DALL-E prompts for social media images."
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
        max_tokens=800
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

    # Convert local images to base64 for vision API
    import base64

    image_data_urls = []
    for img_info in all_images:
        try:
            with open(img_info['path'], 'rb') as f:
                image_bytes = f.read()
                image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                image_data_urls.append(f"data:image/png;base64,{image_b64}")
        except Exception as e:
            print(f"  Warning: Failed to read {img_info['path']}: {e}")
            image_data_urls.append(img_info['url'])  # Fallback to URL

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
                    "image_url": {"url": image_data_urls[0], "detail": "low"}
                },
                {
                    "type": "text",
                    "text": f"Image 1 ({all_images[0]['prompt_variation']})"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_urls[1], "detail": "low"}
                },
                {
                    "type": "text",
                    "text": f"Image 2 ({all_images[1]['prompt_variation']})"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_urls[2], "detail": "low"}
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
