"""
Script to analyze Arcade flow data and generate a comprehensive markdown report.
"""

import os
import json
import yaml
import requests
from datetime import datetime
from openai import OpenAI
from utils import OpenAICache, cached_openai_request


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


def generate_markdown_report(flow_data: dict, user_actions: str, summary: str, image_url: str, image_path: str) -> str:
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

![{flow_name}]({image_path})

**Image URL:** {image_url}

---

## Technical Details

- **Total Steps:** {len(flow_data.get('steps', []))}
- **Flow ID:** {flow_data.get('uploadId', 'N/A')}
- **Created With:** {flow_data.get('createdWith', 'N/A')}
- **Use Case:** {flow_data.get('useCase', 'N/A')}

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

    # Step 3: Create Social Media Image
    print("\n=== Step 3: Generating Social Media Image ===")

    flow_name = flow_data.get('name', 'Arcade Flow')

    image_prompt_response = cached_openai_request(
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
                "content": f"""Create a compelling DALL-E prompt for a social media image based on this flow:

Title: {flow_name}
Summary: {summary}

The image should be:
- Professional and modern
- Eye-catching for social media
- Representative of the flow's purpose
- Suitable for platforms like LinkedIn, Twitter, etc.

Provide only the DALL-E prompt, nothing else."""
            }
        ],
        temperature=0.7,
        max_tokens=300
    )

    image_prompt = image_prompt_response['choices'][0]['message']['content'].strip()
    print(f"\nImage prompt: {image_prompt}")

    image_response = cached_openai_request(
        client=client,
        cache=cache,
        request_type="image",
        model="dall-e-3",
        prompt=image_prompt,
        size="1024x1024",
        quality="standard",
        n=1
    )

    image_url = image_response['data'][0]['url']
    print(f"\nImage URL: {image_url}")

    # Download the image
    image_filename = "social_media_image.png"
    print(f"\nDownloading image to {image_filename}...")
    if download_image(image_url, image_filename):
        print(f"✓ Image saved successfully")
    else:
        print(f"⚠ Failed to download image, will use URL in report")

    # Generate markdown report
    print("\n=== Generating Markdown Report ===")

    markdown_content = generate_markdown_report(
        flow_data=flow_data,
        user_actions=user_actions,
        summary=summary,
        image_url=image_url,
        image_path=image_filename
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


if __name__ == "__main__":
    main()
