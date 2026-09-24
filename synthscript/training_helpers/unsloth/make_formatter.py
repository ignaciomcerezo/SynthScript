from collections.abc import Callable
from typing import Literal

from PIL import Image


def prepare_unsloth_formatter(instruction_text: str, homogenizer: Callable[[str], str]):
    def formatter(sample: dict[Literal["image", "text"], Image.Image | str]):
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": instruction_text,
                        },
                        {
                            "type": "image",
                            "image": Image.fromarray(
                                sample["image"]  # ty: ignore[invalid-argument-type]
                            ),
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": homogenizer(
                                sample["text"]  # ty: ignore[invalid-argument-type]
                            ),
                        }
                    ],
                },
            ]
        }

    return formatter
