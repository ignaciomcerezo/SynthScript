import gc

import torch
from transformers import TrainerCallback  # ty: ignore[unresolved-import]


class AdaptiveCleanupCallback(TrainerCallback):
    """
    Callback para resolver (parcialmente) el problema de fragmentación de memoria.
    Vacía la caché de la tarjeta gráfica cuando sobrepasa el límite danger_limit_gb
    """

    def __init__(self, danger_limit_gb=39.0):
        self.danger_limit_bytes: float = danger_limit_gb * 1024**3

    def on_step_end(self, args, state, control, **kwargs):
        reserved_memory = torch.cuda.memory_reserved()

        if reserved_memory >= self.danger_limit_bytes:
            print(
                f"\n[Warning] VRAM usage hit {reserved_memory / 1024**3:.2f}GB at step {state.global_step}. Flushing cache to prevent OOM..."
            )

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


class ProactiveCleanupCallback(TrainerCallback):
    def __init__(self, danger_limit_gb=42.0, safety_buffer_gb=8.0, alpha=0.3):
        """
        Callback para resolver (parcialmente) el problema de fragmentación de memoria.
        Emplea estructuras más inteligentes que AdaptiveCleanupCallback, pues tiene en cuenta el currículum.

        danger_limit_gb: Máximo absoluto de VRAM después del limpiado.
        safety_buffer_gb: Mínimo colchón de VRAM vacía estática requerida para hacer una pasada.
        alpha: Factor de suavizado para el EWMA (0 < alpha <= 1). Valores altos reaccionan más rápido.
        """
        self.danger_limit_bytes = danger_limit_gb * 1024**3
        self.safety_buffer_bytes = safety_buffer_gb * 1024**3
        self.alpha = alpha

        self.ewma_peak = None
        self.allocated_at_start = 0

    def on_step_begin(self, args, state, control, **kwargs):
        allocated = torch.cuda.memory_allocated()
        total = torch.cuda.memory_reserved()

        available = self.danger_limit_bytes - allocated

        # calculamos el colchón de seguridad dinámico basado en el EWMA de picos anteriores
        dynamic_buffer = self.safety_buffer_bytes
        if self.ewma_peak is not None:
            dynamic_buffer = max(self.safety_buffer_bytes, self.ewma_peak)

        # si el espacio disponible no es capaz de absorber la demanda estimada, vaciamos caché
        if available < dynamic_buffer or total >= self.danger_limit_bytes:
            print(
                f"\n 🔮 [WARNING] Proactive VRAM flush (Step {state.global_step}):\n"
                f"   Allocated VRAM: {allocated / 1024**3:.2f}GB | Reserved Pool: {total / 1024**3:.2f}GB\n"
                f"   Available Headroom: {available / 1024**3:.2f}GB | Estimated Spike Demand: {dynamic_buffer / 1024**3:.2f}GB\n"
            )
            gc.collect()
            torch.cuda.empty_cache()

            allocated = torch.cuda.memory_allocated()

        self.allocated_at_start = allocated
        torch.cuda.reset_peak_memory_stats()

    def on_step_end(self, args, state, control, **kwargs):
        peak_allocated = torch.cuda.max_memory_allocated()
        step_peaking = peak_allocated - self.allocated_at_start

        if self.ewma_peak is None:
            self.ewma_peak = step_peaking
        else:
            self.ewma_peak = (self.alpha * step_peaking) + (
                (1 - self.alpha) * self.ewma_peak
            )

        reserved_memory = torch.cuda.memory_reserved()
        if reserved_memory >= self.danger_limit_bytes:
            gc.collect()
            torch.cuda.empty_cache()

    def on_epoch_end(self, args, state, control, **kwargs):
        """
        Al cambiar de época limpia todo rastro y destruye el EWMA acumulado.
        Esto permite que la primera iteración de la nueva etapa del curriculum calcule
        su baseline sin ensuciarse con las anteriores.
        """
        print(
            "\n [Epoch End] Flushing VRAM caches and resetting EWMA metrics for curriculum shift."
        )
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        self.ewma_peak = None
