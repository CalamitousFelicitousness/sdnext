import os
from dataclasses import dataclass
from PIL import Image
from modules.logger import log
from .options import Options
from .helpers import b64, is_cloud_model


debug_enabled = os.environ.get('SD_LLM_DEBUG', None) is not None
debug_log = log.trace if debug_enabled else lambda *args, **kwargs: None


@dataclass(frozen=True)
class Modality:
    """Which option fields hold the instructions for one kind of output."""
    text: str # no image supplied
    image: str # image supplied together with a prompt
    image_only: str # image supplied without a prompt
    details: str # detail guidance appended to whichever of the three applies


IMAGE = Modality('t2i_prompt', 'i2i_prompt', 'i2i_noprompt', 'details_prompt')
MODALITIES = {
    'video': Modality('t2v_prompt', 'i2v_prompt', 'i2v_noprompt', 'details_prompt'),
    # music has no image-conditioned variants, so an image arriving with a music request still gets
    # the music instruction rather than an image one
    'audio': Modality('t2a_prompt', 't2a_prompt', 't2a_prompt', 'details_audio'),
}


def modality_of(module: str | None) -> Modality:
    """Prompt set for a module. Anything unrecognized enhances for images, which is the older default."""
    return MODALITIES.get(module or '', IMAGE)


def build_system(options: Options, modality: Modality, nsfw: bool, has_prompt: bool, has_image: bool) -> str:
    """Instruction for the request shape, then the nsfw, detail and format blocks."""
    if not has_image:
        field = modality.text
    else:
        field = modality.image if has_prompt else modality.image_only
    system = getattr(options, field)
    system += options.nsfw_ok if nsfw else options.nsfw_no
    system += getattr(options, modality.details)
    system += options.details_format
    return system


def get_text_template(system, prompt, options, nsfw, has_system, has_prompt, has_processor, modality, _image) -> list[dict]:
    if not has_system:
        system = build_system(options, modality, nsfw, has_prompt=has_prompt, has_image=False)
        debug_log(f'Prompt enhance: system="{system}"')
    if not has_prompt:
        prompt = 'be creative!'
    if not has_processor:
        chat_template = [
            { "role": "system", "content": system },
            { "role": "user",   "content": prompt },
        ]
    else:
        chat_template = [
            { "role": "system", "content": [
                {"type": "text", "text": system }
            ] },
            { "role": "user",   "content": [
                {"type": "text", "text": prompt},
            ] },
        ]
    return chat_template


def get_image_template(system, prompt, options, nsfw, has_system, has_prompt, _has_processor, modality, image) -> list[dict]:
    if not has_system:
        system = build_system(options, modality, nsfw, has_prompt=has_prompt, has_image=True)
        debug_log(f'Prompt enhance: system="{system}"')
    if has_prompt:
        chat_template = [
            { "role": "system", "content": [
                {"type": "text", "text": system }
            ] },
            { "role": "user",   "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": b64(image)}
            ] },
        ]
    else:
        chat_template = [
            { "role": "system", "content": [
                {"type": "text", "text": system }
            ] },
            { "role": "user",   "content": [
                {"type": "image", "image": b64(image)}
            ] },
        ]
    return chat_template


def set_template(
        system: str | None,
        prompt: str | None,
        image: Image.Image | None,
        options: Options,
        model: str,
        nsfw: bool = True,
        has_processor: bool = False,
        module: str | None = None,
) -> list[dict] | str:
    chat_template = []
    has_system = system is not None and len(system) > 4
    has_prompt = prompt is not None and len(prompt) > 4
    has_image = image is not None and isinstance(image, Image.Image)
    modality = modality_of(module)

    debug_log(f'Prompt enhance template: module={module} system={has_system} prompt={has_prompt} image={has_image} modality={modality.text} model="{model}" nsfw={nsfw} processor={has_processor}')

    if has_image:
        if is_cloud_model(model):
            pass
        elif options.processor is None:
            log.error('Prompt enhance: image not supported by model')
            return prompt if prompt is not None else '' # Return original text part if image cannot be processed

    if has_image:
        chat_template = get_image_template(system, prompt, options, nsfw, has_system, has_prompt, has_processor, modality, image)
    else:
        chat_template = get_text_template(system, prompt, options, nsfw, has_system, has_prompt, has_processor, modality, image)

    return chat_template
