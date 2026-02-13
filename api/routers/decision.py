from fastapi import APIRouter, Request

router = APIRouter()

@router.get("/decision/latest")
def decision_latest(request: Request):
    svc = request.app.state.infer
    return svc.predict_latest()
