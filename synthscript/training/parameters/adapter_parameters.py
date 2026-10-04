from dataclasses import dataclass, fields
from typing import Any

from unsloth import FastVisionModel  # ty: ignore[unresolved-import]


@dataclass(kw_only=True, slots=True, frozen=True)
class AdapterParameters:
    """
    Parámetros para el adaptador de LoRA
    """

    model: Any
    finetune_vision_layers: bool = False  # False if not finetuning vision layers
    finetune_language_layers: bool = True  # False if not finetuning language layers
    finetune_attention_modules: bool = True  # False if not finetuning attention layers
    finetune_mlp_modules: bool = True  # False if not finetuning MLP layers
    r: int = 8  # The larger, the higher the accuracy, but might overfit
    lora_alpha: int = 8  # Recommended alpha == r at least
    lora_dropout: float = 0.05
    bias: Any = "none"
    random_state: int = 3407
    use_rslora: bool = False  # We support rank stabilized LoRA
    loftq_config: Any = None  # And LoftQ

    def to_dict(self) -> dict:
        """
        Convierte a un diccionario serializable para poder subirlo con el dataset.
        """
        result = {}
        for field in fields(self):
            value = getattr(self, field.name)

            # serializamos el modelo, que será una instancia de model.
            if field.name == "model" and not isinstance(value, str):
                if hasattr(value, "config") and hasattr(value.config, "_name_or_path"):
                    result["model"] = value.config._name_or_path
                elif hasattr(value, "__class__"):
                    result["model"] = value.__class__.__name__
                else:
                    result["model"] = str(value)
            else:
                result[field.name] = value
        return result

    def get_adapted_model(self):
        return FastVisionModel.get_peft_model(
            model=self.model,
            finetune_vision_layers=self.finetune_vision_layers,
            finetune_language_layers=self.finetune_language_layers,
            finetune_attention_modules=self.finetune_attention_modules,
            finetune_mlp_modules=self.finetune_mlp_modules,
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            random_state=self.random_state,
            use_rslora=self.use_rslora,
            loftq_config=self.loftq_config,
        )
