from __future__ import annotations
import gradio as gr
from modules.options import OptionInfo, options_section
from modules.cloud.registry import list_providers
from modules.cloud.google import reset_client as reset_google_client


def register_options(options_templates: dict) -> None:
    options_templates.update(options_section(('cloud_options', "Cloud Providers"), {
        "cloud_intro_sep":               OptionInfo("<h2>Cloud Providers</h2>", "", gr.HTML),
        "cloud_default_text_provider":   OptionInfo('google', "Default text provider", gr.Dropdown,
            lambda: {"choices": list_providers('text', only_enabled=False) or ['google']}),
        "cloud_default_vision_provider": OptionInfo('google', "Default vision provider", gr.Dropdown,
            lambda: {"choices": list_providers('vision', only_enabled=False) or ['google']}),
        "cloud_request_timeout":         OptionInfo(60, "Request timeout in seconds (non-streaming)", gr.Slider,
            {"minimum": 10, "maximum": 600, "step": 5}),
        "cloud_streaming_enabled":       OptionInfo(True, "Enable streaming for prompt-enhance"),

        "cloud_jobs_sep":                OptionInfo("<h2>Cloud Jobs</h2>", "", gr.HTML),
        "cloud_jobs_note":               OptionInfo("<i>Image and video providers run as background jobs. Settings below tune the job runner that powers /sdapi/v1/cloud/jobs and the WebSocket progress channel at /sdapi/v1/ws.</i>", "", gr.HTML),
        "cloud_job_poll_default":        OptionInfo(5, "Default poll interval (seconds)", gr.Slider,
            {"minimum": 1, "maximum": 60, "step": 1}),
        "cloud_job_max_duration":        OptionInfo(600, "Max job duration (seconds, auto-cancel)", gr.Slider,
            {"minimum": 60, "maximum": 3600, "step": 30}),
        "cloud_job_history_size":        OptionInfo(50, "Jobs history size (LRU)", gr.Slider,
            {"minimum": 10, "maximum": 500, "step": 10}),

        "cloud_google_sep":   OptionInfo("<h2>Google</h2>", "", gr.HTML),
        "google_use_vertexai": OptionInfo(False, "Google cloud use VertexAI endpoints", onchange=reset_google_client),
        "google_api_key":      OptionInfo("", "Google cloud API key", gr.Textbox, secret=True, env_var='GOOGLE_API_KEY', onchange=reset_google_client),
        "google_project_id":   OptionInfo("", "Google Cloud project ID", gr.Textbox, secret=True, env_var='GOOGLE_PROJECT_ID', onchange=reset_google_client),
        "google_location_id":  OptionInfo("", "Google Cloud location ID", gr.Textbox, onchange=reset_google_client),

        "cloud_openai_sep":     OptionInfo("<h2>OpenAI</h2>", "", gr.HTML),
        "openai_key":           OptionInfo("", "OpenAI API key", gr.Textbox, secret=True, env_var='OPENAI_API_KEY'),
        "openai_base_override": OptionInfo("", "OpenAI base URL override (optional)"),

        "cloud_anthropic_sep":  OptionInfo("<h2>Anthropic</h2>", "", gr.HTML),
        "anthropic_key":        OptionInfo("", "Anthropic API key", gr.Textbox, secret=True, env_var='ANTHROPIC_API_KEY'),

        "cloud_openrouter_sep": OptionInfo("<h2>OpenRouter</h2>", "", gr.HTML),
        "openrouter_key":       OptionInfo("", "OpenRouter API key", gr.Textbox, secret=True, env_var='OPENROUTER_API_KEY'),

        "cloud_nanogpt_sep":    OptionInfo("<h2>NanoGPT</h2>", "", gr.HTML),
        "nanogpt_key":          OptionInfo("", "NanoGPT API key", gr.Textbox, secret=True, env_var='NANOGPT_API_KEY'),

        "cloud_aihubmix_sep":   OptionInfo("<h2>AIHubMix</h2>", "", gr.HTML),
        "aihubmix_key":         OptionInfo("", "AIHubMix API key", gr.Textbox, secret=True, env_var='AIHUBMIX_API_KEY'),

        "cloud_huggingface_sep":  OptionInfo("<h2>HuggingFace Inference Providers</h2>", "", gr.HTML),
        "cloud_huggingface_note": OptionInfo("<i>Uses the HuggingFace token from System Paths section</i>", "", gr.HTML),

        "cloud_custom_sep":            OptionInfo("<h2>Custom OpenAI-compatible</h2>", "", gr.HTML),
        "openai_compat_custom_url":    OptionInfo("", "Custom OpenAI-compat base URL"),
        "openai_compat_custom_key":    OptionInfo("", "Custom OpenAI-compat API key", gr.Textbox, secret=True),
        "openai_compat_custom_models": OptionInfo("", "Custom model IDs (comma separated)"),
    }))
