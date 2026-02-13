from fastapi import APIRouter, Request

router = APIRouter()

@router.get("/model_version")
def model_version(request: Request):
    svc = request.app.state.infer
    return {
        "model_type": svc.model_type,
        "n_features": len(getattr(svc, "features", [])),
        "features_names": getattr(svc, "features", []),
        "selected": svc.active.get("selected_strategy"),
    }
