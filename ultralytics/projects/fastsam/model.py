# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

from ultralytics.engine.engine_project import Engine_Project
from .predict import FastSAMPredictor
from .val import FastSAMValidator


class FastSAM(Engine_Project):
    """
    FastSAM model interface.

    Example:
        ```python
        from ultralytics import FastSAM

        model = FastSAM('last.pt')
        results = model.predict('ultralytics/assets/bus.jpg')
        ```
    """

    def __init__(self, model="FastSAM-x.pt"):
        """Call the __init__ method of the parent class (YOLO) with the updated default model."""
        if str(model) == "FastSAM.pt":
            model = "FastSAM-x.pt"
        assert Path(model).suffix not in (".yaml", ".yml"), "FastSAM models only support pre-trained models."
        super().__init__(model=model, task_name="segment")

    @property
    def task_map(self):
        """Returns a dictionary mapping segment task_name to corresponding predictor and validator classes."""
        return {"segment": {"predictor": FastSAMPredictor, "validator": FastSAMValidator}}
