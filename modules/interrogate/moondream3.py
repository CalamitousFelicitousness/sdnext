# Moondream 3 Preview VLM Implementation
# Source: https://huggingface.co/moondream/moondream3-preview
# Model: 9.3GB, gated (requires HuggingFace authentication)
# Architecture: Mixture-of-Experts (9B total params, 2B active)
#
# Phase 1: Full feature implementation including:
# - query() and caption() with streaming support
# - point() for object coordinate identification
# - detect() for bounding box detection
# - Image encoding cache for efficiency

import os
import re
import transformers
from PIL import Image
from modules import shared, devices, sd_models


# Debug logging - function-based to avoid circular import
debug_enabled = os.environ.get('SD_VQA_DEBUG', None) is not None

def debug(*args, **kwargs):
    if debug_enabled:
        shared.log.trace(*args, **kwargs)


# Global state
moondream3_model = None
loaded = None
image_cache = {}  # Cache encoded images for reuse


def get_settings():
    """
    Build settings dict for Moondream 3 API from global VQA options.
    Moondream 3 accepts: temperature, top_p, max_tokens
    """
    settings = {}
    if shared.opts.interrogate_vlm_max_length > 0:
        settings['max_tokens'] = shared.opts.interrogate_vlm_max_length
    if shared.opts.interrogate_vlm_temperature > 0:
        settings['temperature'] = shared.opts.interrogate_vlm_temperature
    if shared.opts.interrogate_vlm_top_p > 0:
        settings['top_p'] = shared.opts.interrogate_vlm_top_p
    return settings if settings else None


def load_model(repo: str):
    """Load and compile Moondream 3 model."""
    global moondream3_model, loaded  # pylint: disable=global-statement

    if moondream3_model is None or loaded != repo:
        shared.log.debug(f'Interrogate load: vlm="{repo}"')
        moondream3_model = None
        moondream3_model = transformers.AutoModelForCausalLM.from_pretrained(
            repo,
            trust_remote_code=True,
            torch_dtype=devices.dtype,
            cache_dir=shared.opts.hfcache_dir,
        )
        moondream3_model.eval()
        if 'LLM' in shared.opts.cuda_compile:
            debug('VQA interrogate: handler=moondream3 compiling model for fast decoding')
            moondream3_model.compile()  # Critical for fast decoding per moondream3 docs
        loaded = repo
        devices.torch_gc()

    # Move model to active device
    sd_models.move_model(moondream3_model, devices.device)
    return moondream3_model


def encode_image(image: Image.Image, cache_key: str = None):
    """
    Encode image for reuse across multiple queries.

    Args:
        image: PIL Image
        cache_key: Optional cache key for storing encoded image

    Returns:
        Encoded image tensor
    """
    if cache_key and cache_key in image_cache:
        debug(f'VQA interrogate: handler=moondream3 using cached encoding for cache_key="{cache_key}"')
        return image_cache[cache_key]

    model = load_model(loaded)

    with devices.inference_context():
        encoded = model.encode_image(image)

    if cache_key:
        image_cache[cache_key] = encoded
        debug(f'VQA interrogate: handler=moondream3 cached encoding cache_key="{cache_key}" cache_size={len(image_cache)}')

    return encoded


def query(image: Image.Image, question: str, repo: str, stream: bool = False,
          temperature: float = None, top_p: float = None, max_tokens: int = None,
          use_cache: bool = False):
    """
    Visual question answering with optional streaming.

    Args:
        image: PIL Image
        question: Question about the image
        repo: Model repository
        stream: Enable streaming output (generator)
        temperature: Sampling temperature (overrides global setting)
        top_p: Nucleus sampling parameter (overrides global setting)
        max_tokens: Maximum tokens to generate (overrides global setting)
        use_cache: Use cached image encoding if available

    Returns:
        Answer dict or string (or generator if stream=True)
    """
    model = load_model(repo)

    # Build settings - per-call parameters override global settings
    settings = get_settings() or {}
    if temperature is not None:
        settings['temperature'] = temperature
    if top_p is not None:
        settings['top_p'] = top_p
    if max_tokens is not None:
        settings['max_tokens'] = max_tokens

    debug(f'VQA interrogate: handler=moondream3 method=query question="{question}" stream={stream} settings={settings}')

    # Use cached encoding if requested
    if use_cache:
        cache_key = f"{id(image)}_{question}"
        image_input = encode_image(image, cache_key)
    else:
        image_input = image

    with devices.inference_context():
        response = model.query(
            image=image_input,
            question=question,
            stream=stream,
            settings=settings if settings else None
        )

    # Log response structure (for non-streaming)
    if not stream:
        if isinstance(response, dict):
            debug(f'VQA interrogate: handler=moondream3 response_type=dict keys={list(response.keys())}')
            if 'reasoning' in response:
                reasoning_text = response['reasoning'].get('text', '')[:100] + '...' if len(response['reasoning'].get('text', '')) > 100 else response['reasoning'].get('text', '')
                debug(f'VQA interrogate: handler=moondream3 reasoning="{reasoning_text}"')
            if 'answer' in response:
                debug(f'VQA interrogate: handler=moondream3 answer="{response["answer"]}"')

    return response


def caption(image: Image.Image, repo: str, length: str = 'normal', stream: bool = False,
            temperature: float = None, top_p: float = None, max_tokens: int = None):
    """
    Generate image captions at different lengths.

    Args:
        image: PIL Image
        repo: Model repository
        length: Caption length - 'short', 'normal', or 'long'
        stream: Enable streaming output (generator)
        temperature: Sampling temperature (overrides global setting)
        top_p: Nucleus sampling parameter (overrides global setting)
        max_tokens: Maximum tokens to generate (overrides global setting)

    Returns:
        Caption dict or string (or generator if stream=True)
    """
    model = load_model(repo)

    # Build settings - per-call parameters override global settings
    settings = get_settings() or {}
    if temperature is not None:
        settings['temperature'] = temperature
    if top_p is not None:
        settings['top_p'] = top_p
    if max_tokens is not None:
        settings['max_tokens'] = max_tokens

    debug(f'VQA interrogate: handler=moondream3 method=caption length={length} stream={stream} settings={settings}')

    with devices.inference_context():
        response = model.caption(
            image,
            length=length,
            stream=stream,
            settings=settings if settings else None
        )

    # Log response structure (for non-streaming)
    if not stream and isinstance(response, dict):
        debug(f'VQA interrogate: handler=moondream3 response_type=dict keys={list(response.keys())}')

    return response


def point(image: Image.Image, object_name: str, repo: str):
    """
    Identify coordinates of all instances of a specific object in the image.

    Args:
        image: PIL Image
        object_name: Name of object to locate
        repo: Model repository

    Returns:
        List of (x, y) tuples with coordinates normalized to 0-1 range, or None if not found
        Example: [(0.733, 0.442), (0.5, 0.6)] for 2 instances
    """
    model = load_model(repo)

    debug(f'VQA interrogate: handler=moondream3 method=point object_name="{object_name}"')

    with devices.inference_context():
        result = model.point(image, object_name)

    # Debug: Log the actual result to understand the format
    debug(f'VQA interrogate: handler=moondream3 point_raw_result="{result}" type={type(result)}')
    if isinstance(result, dict):
        debug(f'VQA interrogate: handler=moondream3 point_raw_result_keys={list(result.keys())}')

    # Parse and validate coordinates
    # Handle dict format: {'points': [{'x': 0.733, 'y': 0.442}, {'x': 0.5, 'y': 0.6}, ...]}
    if isinstance(result, dict) and 'points' in result:
        points_list = result['points']
        if points_list and len(points_list) > 0:
            coordinates = []
            for point_data in points_list:  # Iterate ALL points
                if 'x' in point_data and 'y' in point_data:
                    x = max(0.0, min(1.0, float(point_data['x'])))
                    y = max(0.0, min(1.0, float(point_data['y'])))
                    coordinates.append((x, y))
            if coordinates:
                debug(f'VQA interrogate: handler=moondream3 point_result={len(coordinates)} points found')
                return coordinates
    # Fallback: try simple list/tuple format [x, y] (for compatibility)
    elif isinstance(result, (list, tuple)) and len(result) == 2:
        x, y = result
        x = max(0.0, min(1.0, float(x)))
        y = max(0.0, min(1.0, float(y)))
        debug('VQA interrogate: handler=moondream3 point_result=1 point found')
        return [(x, y)]  # Return as list for consistency

    debug('VQA interrogate: handler=moondream3 point_result=None (not found)')
    return None


def detect(image: Image.Image, object_name: str, repo: str, max_objects: int = 10):
    """
    Detect specific objects and return bounding boxes.

    Args:
        image: PIL Image
        object_name: Name of object to detect (e.g., "car", "person")
        repo: Model repository
        max_objects: Maximum number of objects to detect (passed to model via settings)

    Returns:
        List of dicts with format:
        [
            {
                'label': 'object_name',
                'bbox': [x1, y1, x2, y2],  # Normalized 0-1 coordinates
                'confidence': 1.0
            },
            ...
        ]
    """
    model = load_model(repo)

    # Build settings dict with max_objects
    settings = {}
    if max_objects:
        settings['max_objects'] = max_objects

    debug(f'VQA interrogate: handler=moondream3 method=detect object_name="{object_name}" settings={settings}')

    with devices.inference_context():
        # Official API: model.detect(image, query, settings=None)
        # Returns: {"objects": [{"x_min": ..., "y_min": ..., "x_max": ..., "y_max": ...}]}
        result = model.detect(image, object_name, settings=settings if settings else None)

    debug(f'VQA interrogate: handler=moondream3 detect_raw_result="{result}" type={type(result)}')

    # Parse and validate bounding boxes
    detections = []

    # Handle official format: {"objects": [...]}
    if isinstance(result, dict) and 'objects' in result:
        objects_list = result['objects']
        for obj in objects_list:
            try:
                # Official format uses x_min, y_min, x_max, y_max
                if 'x_min' in obj and 'y_min' in obj and 'x_max' in obj and 'y_max' in obj:
                    # Convert to [x1, y1, x2, y2] format and validate
                    x1 = max(0.0, min(1.0, float(obj['x_min'])))
                    y1 = max(0.0, min(1.0, float(obj['y_min'])))
                    x2 = max(0.0, min(1.0, float(obj['x_max'])))
                    y2 = max(0.0, min(1.0, float(obj['y_max'])))

                    detections.append({
                        'label': object_name,  # Use query as label
                        'bbox': [x1, y1, x2, y2],
                        'confidence': 1.0  # Not provided in official API response
                    })
            except Exception as e:
                from modules import errors
                errors.display(e, 'Moondream3 detect parsing')
                continue

    debug(f'VQA interrogate: handler=moondream3 detect_result={len(detections)} objects')
    return detections


def predict(question: str, image: Image.Image, repo: str, model_name: str = None,
            mode: str = None, stream: bool = False, use_cache: bool = False, **kwargs):
    """
    Main entry point for Moondream 3 inference.
    Dispatches to appropriate method based on question/mode.

    Args:
        question: Question or task description
        image: PIL Image
        repo: Model repository
        model_name: Model display name (for logging)
        mode: Force specific mode ('query', 'caption', 'caption_short', 'caption_long', 'point', 'detect')
        stream: Enable streaming output (for query/caption)
        use_cache: Use cached image encoding (for query)
        **kwargs: Additional parameters (max_objects for detect, etc.)

    Returns:
        Response string or tuple (text, annotated_image) for detect/point modes
        (or generator if stream=True for query/caption modes)
    """
    debug(f'VQA interrogate: handler=moondream3 model_name="{model_name}" repo="{repo}" question="{question}" image_size={image.size if image else None} mode={mode} stream={stream}')

    # Clean question
    question = question.replace('<', '').replace('>', '').replace('_', ' ') if question else ''

    # Auto-detect mode from question if not specified
    if mode is None:
        question_lower = question.lower()

        # Caption detection
        if question in ['CAPTION', 'caption'] or 'caption' in question_lower:
            if 'more detailed' in question_lower or 'very long' in question_lower:
                mode = 'caption_long'
            elif 'detailed' in question_lower or 'long' in question_lower:
                mode = 'caption_normal'
            elif 'short' in question_lower or 'brief' in question_lower:
                mode = 'caption_short'
            else:
                # Default caption mode (matches vqa.py legacy behavior)
                if question == 'CAPTION':
                    mode = 'caption_short'
                elif question == 'DETAILED CAPTION':
                    mode = 'caption_normal'
                elif question == 'MORE DETAILED CAPTION':
                    mode = 'caption_long'
                else:
                    mode = 'caption_normal'

        # Point detection
        elif 'where is' in question_lower or 'locate' in question_lower or 'find' in question_lower or 'point' in question_lower:
            mode = 'point'

        # Object detection
        elif 'detect' in question_lower or 'bounding box' in question_lower or 'bbox' in question_lower:
            mode = 'detect'

        # Default to query
        else:
            mode = 'query'

    debug(f'VQA interrogate: handler=moondream3 mode_selected={mode}')

    # Dispatch to appropriate method
    try:
        if mode == 'caption_short':
            response = caption(image, repo, length='short', stream=stream)
        elif mode == 'caption_long':
            response = caption(image, repo, length='long', stream=stream)
        elif mode in ['caption', 'caption_normal']:
            response = caption(image, repo, length='normal', stream=stream)
        elif mode == 'point':
            # Extract object name from question - case insensitive, preserve object names
            object_name = question
            # Remove trigger phrases (case-insensitive)
            for phrase in ['point at', 'where is', 'locate', 'find']:
                object_name = re.sub(rf'\b{phrase}\b', '', object_name, flags=re.IGNORECASE)
            # Remove punctuation and extra whitespace
            object_name = re.sub(r'[?.!,]', '', object_name).strip()
            # Remove leading "the" only
            object_name = re.sub(r'^\s*the\s+', '', object_name, flags=re.IGNORECASE)
            debug(f'VQA interrogate: handler=moondream3 point_extracted_object="{object_name}"')
            result = point(image, object_name, repo)
            if result:
                # Handle multiple instances - return text and points for drawing
                if len(result) == 1:
                    text = f"Found at coordinates: ({result[0][0]:.3f}, {result[0][1]:.3f})"
                else:
                    # Multiple instances found - format with count
                    lines = [f"Found {len(result)} instances:"]
                    for i, (x, y) in enumerate(result, 1):
                        lines.append(f"  {i}. ({x:.3f}, {y:.3f})")
                    text = '\n'.join(lines)
                return (text, {'points': result})  # Return text and points data
            return ("Object not found", None)
        elif mode == 'detect':
            # Extract object name from question - case insensitive
            object_name = question
            # Remove trigger phrases (case-insensitive)
            for phrase in ['detect', 'find all', 'bounding box', 'bbox', 'find']:
                object_name = re.sub(rf'\b{phrase}\b', '', object_name, flags=re.IGNORECASE)
            # Remove punctuation and extra whitespace
            object_name = re.sub(r'[?.!,]', '', object_name).strip()
            # Remove leading "the" only
            object_name = re.sub(r'^\s*the\s+', '', object_name, flags=re.IGNORECASE)
            # Remove "and" and get first object (model detects one type at a time)
            if ' and ' in object_name.lower():
                object_name = re.split(r'\s+and\s+', object_name, flags=re.IGNORECASE)[0].strip()
            debug(f'VQA interrogate: handler=moondream3 detect_extracted_object="{object_name}"')

            results = detect(image, object_name, repo, max_objects=kwargs.get('max_objects', 10))
            # Format as string for display and return detections for drawing
            if results:
                lines = [f"{det['label']}: [{det['bbox'][0]:.3f}, {det['bbox'][1]:.3f}, {det['bbox'][2]:.3f}, {det['bbox'][3]:.3f}] (confidence: {det['confidence']:.2f})"
                        for det in results]
                text = '\n'.join(lines)
                return (text, {'detections': results})  # Return text and detection data
            return ("No objects detected", None)
        else:  # mode == 'query'
            if len(question) < 2:
                question = "Describe this image."
            response = query(image, question, repo, stream=stream, use_cache=use_cache)

        debug(f'VQA interrogate: handler=moondream3 response_before_clean="{response}"')
        return response

    except Exception as e:
        from modules import errors
        errors.display(e, 'Moondream3')
        return f"Error: {str(e)}"


def clear_cache():
    """Clear image encoding cache."""
    global image_cache  # pylint: disable=global-statement
    cache_size = len(image_cache)
    image_cache.clear()
    debug(f'VQA interrogate: handler=moondream3 cleared image cache cache_size_was={cache_size}')
    shared.log.debug(f'Moondream3: Cleared image cache ({cache_size} entries)')
