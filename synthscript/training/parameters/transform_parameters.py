# from synthscript.training_helpers.transform_train_test import transform_train
# from dataclasses import dataclass
# from typing import Any, Literal


# @dataclass(kw_only=True, slots=True, frozen=True)
# class TransformParameters:
#     """
#     Argumentos usados por las transformaciones aplicadas al dataset como dataloader.
#     """

#     # transform_train args

#     instruction_text: str = (
#         r"Extract all text from this image in the original French. Do not translate. Format all math, equations, and variables in standard LaTeX enclosed in '$' (e.g., $G$, $\pi_1$). Output ONLY the transcribed text without any conversational filler or introductions."
#     )

#     global_resize_scale: float = 0.5
#     max_dim: int = 1024

#     context_probability: float = 0
#     max_context_chars: int = 50  # 50

#     min_rot: int | float = 2  # 2
#     max_rot: int | float = 3  # 3

#     shift_prop: float = 0.01
#     max_escala: float = 0.02
#     maxdist: float | int = 2
#     distorsion_scale: float = 0.05

#     # CONTEXT CONFIG

#     min_context_chars: int = 20
#     max_context_chars: int = 150

#     # TRAIN DATASET - configured_transform_train args

#     augment_train: bool = True
#     straighten_train: bool = False
#     use_complex_rotation_interval_train: bool = False

#     # EVAL DATASET - configured_transform_train args

#     augment_eval: bool = False
#     straighten_eval: bool = False
#     use_complex_rotation_interval_eval: bool = False

#     def _get_att_from_names(self, att_names: list[str]) -> dict[str, Any]:
#         return {att_name: getattr(self, att_name) for att_name in att_names}

#     @property
#     def common_transform_parameters(self) -> dict[str, Any]:
#         transform_global_parameter_names: list[str] = [
#             "instruction_text",
#             "global_resize_scale",
#             "max_dim",
#             "context_probability",
#             "max_context_chars",
#             "min_rot",
#             "max_rot",
#             "shift_prop",
#             "max_escala",
#             "maxdist",
#             "min_context_chars",
#             "max_context_chars",
#             "distorsion_scale",
#         ]
#         return self._get_att_from_names(transform_global_parameter_names)

#     @property
#     def eval_transform_parameters(self) -> dict[str, Any]:
#         eval_transform_parameter_names: list[str] = [
#             "augment_eval",
#             "straighten_eval",
#             "use_complex_rotation_interval_eval",
#         ]
#         return self._get_att_from_names(eval_transform_parameter_names)

#     @property
#     def train_transform_parameters(self) -> dict[str, Any]:
#         train_transform_parameter_names: list[str] = [
#             "augment_train",
#             "straighten_train",
#             "use_complex_rotation_interval_train",
#         ]
#         return self._get_att_from_names(train_transform_parameter_names)

#     def get_configured_train_transform(self):
#         transform_train_configured = lambda batch: transform_train(
#             batch,
#             augment=self.augment_train,
#             straighten=self.straighten_train,
#             use_complex_rotation_interval=self.use_complex_rotation_interval_train,
#             maxdist=self.maxdist,
#             global_resize_scale=self.global_resize_scale,
#             shift_prop=self.shift_prop,
#             max_dim=self.max_dim,
#             context_probability=self.context_probability,
#             max_escala=self.max_escala,
#             instruction_text=self.instruction_text,
#             min_rot=self.min_rot,
#             max_rot=self.max_rot,
#             distorsion_scale=self.distorsion_scale,
#             context_mode="probabilistic",
#             max_context=self.max_context_chars,
#             min_context=self.min_context_chars,
#             randomize_context_length=True,
#         )
#         return transform_train_configured

#     def get_configured_eval_transform_without_context(self):

#         transform_eval_configured_without_context = lambda batch: transform_train(
#             batch,
#             augment=self.augment_eval,
#             straighten=self.straighten_eval,
#             use_complex_rotation_interval=self.use_complex_rotation_interval_eval,
#             maxdist=self.maxdist,
#             global_resize_scale=self.global_resize_scale,
#             shift_prop=self.shift_prop,
#             max_dim=self.max_dim,
#             context_probability=self.context_probability,
#             max_escala=self.max_escala,
#             instruction_text=self.instruction_text,
#             min_rot=self.min_rot,
#             max_rot=self.max_rot,
#             distorsion_scale=self.distorsion_scale,
#             context_mode="never",
#             max_context=0,  # irrelevante (en virtud de context_mode = "never")
#             min_context=0,  # ídem
#             randomize_context_length=False,
#         )
#         return transform_eval_configured_without_context

#     def get_configured_eval_transform_with_context(self):

#         transform_eval_configured = lambda batch: transform_train(
#             batch,
#             augment=self.augment_eval,
#             straighten=self.straighten_eval,
#             use_complex_rotation_interval=self.use_complex_rotation_interval_eval,
#             maxdist=self.maxdist,
#             global_resize_scale=self.global_resize_scale,
#             shift_prop=self.shift_prop,
#             max_dim=self.max_dim,
#             context_probability=self.context_probability,
#             max_escala=self.max_escala,
#             instruction_text=self.instruction_text,
#             min_rot=self.min_rot,
#             max_rot=self.max_rot,
#             distorsion_scale=self.distorsion_scale,
#             context_mode="always",
#             max_context=self.min_context_chars,
#             min_context=0,  # irrelevante (en virtud de randomize_context_length = False)
#             randomize_context_length=False,
#         )
#         return transform_eval_configured
