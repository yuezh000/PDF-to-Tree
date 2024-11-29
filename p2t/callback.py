from fastNLP import Callback
from utils import save_json
import os


class SaveEvalResultCallback(Callback):
    
    def __init__(self, workdir):
        self.workdir = workdir
        super().__init__()

    def on_evaluate_end(self, trainer, results):
        eval_result = {
            "key_metric": "f1-score",
            "metrics": [
                {"name": "f1-score", "value": results['f#f']},
                {"name": "precision", "value": results['pre#f']},
                {"name": "recall", "value": results['rec#f']},
                {"name": "details", "value": results}
            ]
        }
        metrics_dir = os.path.join(self.workdir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        # Save result
        save_json(os.path.join(metrics_dir, f"eval_result.epoch_{trainer.cur_epoch_idx}.json"), eval_result)
