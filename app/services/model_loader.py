from pathlib import Path
import json
from app.inference.predict_cluster_pipeline import load_cluster_models
from app.services.model_service import get_active_model_run

def load_active_lgbm_models():
    run = get_active_model_run()
    if run is None:
        raise ValueError("Tidak ada model_run aktif")

    params = json.loads(run["params_json"])
    run_dir = Path(params["run_dir"])
    model_dir = run_dir / "models"
    n_clusters = run["n_clusters"]

    model_dict = {}
    for cid in range(n_clusters):
        model_path = model_dir / f"lgbm_full_cluster_{cid}.txt"
        model_dict[cid] = model_path

    models = load_cluster_models(model_dict)
    return models, params["feature_cols"], run
