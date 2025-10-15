"""
Enhanced video analysis functions for combining context from surrounding steps.
"""

import json
from typing import List, Dict, Any


def get_surrounding_context(steps: List[Dict], video_index: int) -> Dict[str, Any]:
    """
    Get context from steps surrounding a VIDEO step.

    Args:
        steps: All steps in the flow
        video_index: Index of the VIDEO step

    Returns:
        Dict with previous_step, next_step, and context info
    """
    context = {
        'previous_step': None,
        'next_step': None,
        'previous_action': None,
        'next_action': None
    }

    # Get previous step (usually an IMAGE with click context)
    if video_index > 0:
        prev_step = steps[video_index - 1]
        context['previous_step'] = prev_step

        if prev_step.get('type') == 'IMAGE':
            click_ctx = prev_step.get('clickContext', {})
            context['previous_action'] = {
                'type': 'click',
                'element': click_ctx.get('text', 'unknown'),
                'element_type': click_ctx.get('elementType', 'unknown'),
                'page_url': prev_step.get('pageContext', {}).get('url', '')
            }

    # Get next step
    if video_index < len(steps) - 1:
        next_step = steps[video_index + 1]
        context['next_step'] = next_step

        if next_step.get('type') == 'IMAGE':
            click_ctx = next_step.get('clickContext', {})
            context['next_action'] = {
                'type': 'click',
                'element': click_ctx.get('text', 'unknown'),
                'element_type': click_ctx.get('elementType', 'unknown'),
                'page_url': next_step.get('pageContext', {}).get('url', '')
            }

    return context


def analyze_video_with_context(client, cache, step: Dict, context: Dict, captured_events: List[Dict]) -> str:
    """
    Analyze a VIDEO step with surrounding context.

    Args:
        client: OpenAI client
        cache: Cache instance
        step: The VIDEO step
        context: Surrounding context from get_surrounding_context()
        captured_events: All captured events

    Returns:
        Human-readable description of what happened in the video
    """
    from utils import cached_openai_request

    # Get thumbnail
    thumbnail_url = step.get('videoThumbnailUrl')

    # Get time range
    start_time = step['startTimeFrac'] * step['duration']
    end_time = step['endTimeFrac'] * step['duration']

    # Find events in this time range
    events_in_video = [
        e for e in captured_events
        if start_time <= e.get('timeMs', 0)/1000 <= end_time
    ]

    # Build context description
    context_text = "Context:\n"

    if context['previous_action']:
        prev = context['previous_action']
        context_text += f"- User just clicked on: '{prev['element']}' ({prev['element_type']})\n"
        context_text += f"- Previous page: {prev['page_url']}\n"

    if context['next_action']:
        nxt = context['next_action']
        context_text += f"- Next action will be: clicking '{nxt['element']}' ({nxt['element_type']})\n"
        context_text += f"- Next page: {nxt['page_url']}\n"

    # Format events
    events_text = "Events during this video:\n"
    for event in events_in_video:
        event_type = event.get('type', 'unknown')
        if event_type == 'typing':
            events_text += f"- User typed text\n"
        elif event_type == 'scrolling':
            events_text += f"- User scrolled the page\n"
        elif event_type == 'click':
            events_text += f"- User clicked\n"
        elif event_type == 'dragging':
            events_text += f"- User dragged\n"
        else:
            events_text += f"- {event_type}\n"

    # Use vision model with context
    description_response = cached_openai_request(
        client=client,
        cache=cache,
        request_type="chat",
        model="gpt-4o",  # Use vision-capable model
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

Based on the screenshot, context, and events, write a single clear sentence describing the user's action.
Example: "Typed 'scooter' into the search bar"
Example: "Scrolled through the search results"
Example: "Selected the blue color option"

Respond with just the action description, no preamble."""
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": thumbnail_url}
                    }
                ]
            }
        ],
        temperature=0.2,
        max_tokens=100
    )

    return description_response['choices'][0]['message']['content'].strip()


def create_enriched_flow_description(client, cache, steps: List[Dict], captured_events: List[Dict]) -> str:
    """
    Create a comprehensive flow description by analyzing all steps including videos.

    Args:
        client: OpenAI client
        cache: Cache instance
        steps: All flow steps
        captured_events: All captured events

    Returns:
        Enriched description with all steps described
    """
    from utils import cached_openai_request

    print("\n→ Analyzing flow steps with video context...")

    enriched_steps = []

    for i, step in enumerate(steps):
        step_type = step.get('type')

        if step_type == 'CHAPTER':
            # Chapter steps are intro/outro
            enriched_steps.append({
                'type': 'chapter',
                'title': step.get('title', ''),
                'subtitle': step.get('subtitle', '')
            })

        elif step_type == 'IMAGE':
            # Image steps have click context
            click_ctx = step.get('clickContext', {})
            page_ctx = step.get('pageContext', {})

            action_desc = None
            if click_ctx:
                element = click_ctx.get('text', 'element')
                element_type = click_ctx.get('elementType', 'unknown')
                action_desc = f"Clicked on '{element}' ({element_type})"

            enriched_steps.append({
                'type': 'image',
                'action': action_desc,
                'page_url': page_ctx.get('url', ''),
                'page_title': page_ctx.get('title', ''),
                'hotspot_label': step.get('hotspots', [{}])[0].get('label', '') if step.get('hotspots') else ''
            })

        elif step_type == 'VIDEO':
            # Analyze video with surrounding context
            context = get_surrounding_context(steps, i)
            video_description = analyze_video_with_context(client, cache, step, context, captured_events)

            print(f"  Video {len([s for s in enriched_steps if s['type'] == 'video']) + 1}: {video_description}")

            enriched_steps.append({
                'type': 'video',
                'action': video_description,
                'duration': (step['endTimeFrac'] - step['startTimeFrac']) * step['duration']
            })

    return enriched_steps


def create_user_interactions_with_videos(client, cache, flow_data: Dict) -> str:
    """
    Create user interactions list with VIDEO descriptions interwoven.

    Args:
        client: OpenAI client
        cache: Cache instance
        flow_data: Complete flow data

    Returns:
        Markdown bulleted list of user interactions
    """
    from utils import cached_openai_request

    steps = flow_data.get('steps', [])
    captured_events = flow_data.get('capturedEvents', [])

    # Get enriched step descriptions
    enriched_steps = create_enriched_flow_description(client, cache, steps, captured_events)

    # Build a narrative from enriched steps
    narrative_parts = []
    for step in enriched_steps:
        if step['type'] == 'chapter':
            if step['title'] and step.get('subtitle'):
                narrative_parts.append(f"**{step['title']}**: {step['subtitle']}")
        elif step['type'] == 'image' and step.get('action'):
            narrative_parts.append(step['action'])
        elif step['type'] == 'video' and step.get('action'):
            narrative_parts.append(step['action'])

    # Use LLM to organize into clean bulleted list
    interactions_response = cached_openai_request(
        client=client,
        cache=cache,
        request_type="chat",
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an expert at creating clear, bulleted lists of user actions."
            },
            {
                "role": "user",
                "content": f"""Convert these user actions into a clean, bulleted markdown list.

Actions:
{chr(10).join(f"- {part}" for part in narrative_parts if part)}

Create a well-organized bulleted list that:
1. List out the actions the user did in a human readable format (i.e. "Clicked on checkout", "Searched for X", "Typed Y into Z")
2. Uses clear, active voice
3. Maintains chronological order

Format as markdown with proper bullets."""
            }
        ],
        temperature=0.3,
        max_tokens=1000
    )

    return interactions_response['choices'][0]['message']['content']
