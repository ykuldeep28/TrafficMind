import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

from .schemas import SituationRequest, ExperimentRequest, ApplyStrategyRequest, TrafficState, Strategy
from .agent import get_agent, GeminiServiceError
from .mcp_client import mcp_client
from .sumo_engine import sumo_available
from .traci_controller import run_traci_adaptive_simulation
from .vision import process_cctv_feed, process_custom_upload, PRESET_FEEDS
from .ml_engine import get_ml_model_info, train_ml_model

# In-memory controller state used by the interactive browser simulation.
# SUMO remains the source of measured strategy performance; this state controls
# which tested strategy the final browser replay visualizes.
controller = {
    "state": None,
    "strategy": None,
    "source": "user input",
    "sim_time": 0.0,
    "running": False,
    "agent1": "idle",
    "agent2": "idle",
    "last_evaluation": None,
}

def _phase_status(state, strategy=None, t=0.0):
    s = state
    st = strategy or {}
    ns = int(st.get("ns_green", s.ns_green))
    ew = int(st.get("ew_green", s.ew_green))
    yellow = int(s.yellow)
    cycle = ns + yellow + ew + yellow
    offset = ns + yellow if s.start_phase == "EW" else 0
    p = (float(t) + offset) % cycle
    if p < ns:
        return {"phase": "NS", "kind": "green", "remaining": round(ns-p, 2), "ns": "GREEN", "south": "GREEN", "east": "RED", "west": "RED"}
    if p < ns + yellow:
        return {"phase": "NS", "kind": "yellow", "remaining": round(ns+yellow-p, 2), "ns": "YELLOW", "south": "YELLOW", "east": "RED", "west": "RED"}
    if p < ns + yellow + ew:
        return {"phase": "EW", "kind": "green", "remaining": round(ns+yellow+ew-p, 2), "ns": "RED", "south": "RED", "east": "GREEN", "west": "GREEN"}
    return {"phase": "EW", "kind": "yellow", "remaining": round(cycle-p, 2), "ns": "RED", "south": "RED", "east": "YELLOW", "west": "YELLOW"}

app = FastAPI(title="TrafficMind")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def startup():
    await mcp_client.start()

@app.on_event("shutdown")
async def shutdown():
    await mcp_client.close()

@app.get("/")
async def index():
    return FileResponse(
        ROOT / "static" / "index.html",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
        "sumo_available": sumo_available(),
        "mcp": "connected" if mcp_client._session else "not_started",
        "interactive_state": bool(controller["state"]),
        "ml_model": get_ml_model_info(),
    }

@app.get("/api/ml/info")
async def ml_info():
    return get_ml_model_info()

@app.post("/api/tick")
async def tick():
    if controller["state"] is not None and controller["running"]:
        controller["sim_time"] = (controller["sim_time"] + 0.25) % controller["state"].duration
    if controller["state"] is None:
        return {"status": "ok", "configured": False}
    ph = _phase_status(controller["state"], controller["strategy"], controller["sim_time"])
    return {"status": "ok", "configured": True, "sim_time": round(controller["sim_time"], 2), "signal": ph, "strategy": controller["strategy"], "source": controller["source"]}

@app.get("/api/traffic/state")
async def traffic_state():
    if controller["state"] is None:
        return {"configured": False}
    s = controller["state"]
    return {
        "configured": True,
        "north": s.north, "south": s.south, "east": s.east, "west": s.west,
        "closed_road": getattr(s, "closed_road", "none"),
        "emergency_approach": getattr(s, "emergency_approach", "none"),
        "duration": s.duration, "sim_time": round(controller["sim_time"], 2),
        "strategy": controller["strategy"], "source": controller["source"],
    }

@app.get("/api/vision/presets")
async def vision_presets():
    return {
        "presets": [
            {
                "id": k,
                "title": v["title"],
                "description": v["description"],
                "counts": v["counts"],
                "emergency_approach": v["emergency_approach"],
                "closed_road": v["closed_road"]
            }
            for k, v in PRESET_FEEDS.items()
        ]
    }

@app.post("/api/vision/detect")
async def vision_detect(preset: Optional[str] = "east_heavy"):
    return process_cctv_feed(preset or "east_heavy")

@app.post("/api/vision/upload")
async def vision_upload(file: UploadFile = File(...)):
    content = await file.read()
    return process_custom_upload(file.filename or "upload.jpg", content)

@app.post("/api/traci/simulate")
async def traci_simulate(req: SituationRequest):
    try:
        res = run_traci_adaptive_simulation(req.state, label="traci_adaptive")
        
        # Fixed baseline for direct comparison
        baseline_strategy = Strategy(
            name="Fixed Baseline",
            description="Fixed signal program without TraCI adaptation",
            ns_green=req.state.ns_green,
            ew_green=req.state.ew_green,
        )
        base = await mcp_client.call("run_sumo", {
            "state": req.state.model_dump(),
            "strategy": baseline_strategy.model_dump(),
            "label": "traci_baseline"
        })
        
        traci_metrics = res.get("metrics", {})
        waiting_reduction = round(float(base.get("avg_waiting_time", 0)) - float(traci_metrics.get("avg_waiting_time", 0)), 2)
        throughput_gain = round(float(traci_metrics.get("throughput", 0)) - float(base.get("throughput", 0)), 2)
        base_wait = max(0.1, float(base.get("avg_waiting_time", 0)))
        pct_reduction = round((waiting_reduction / base_wait) * 100, 1) if base_wait > 0 else 0.0

        return {
            **res,
            "baseline_metrics": base,
            "comparison": {
                "waiting_time_saved_sec": waiting_reduction,
                "throughput_gain_veh_h": throughput_gain,
                "pct_waiting_reduction": pct_reduction,
            }
        }
    except Exception as exc:
        raise _http_error(exc, "TraCI simulation failed")

@app.get("/api/signal/status")
async def signal_status():
    if controller["state"] is None:
        return {"configured": False}
    return {"configured": True, **_phase_status(controller["state"], controller["strategy"], controller["sim_time"]), "strategy": controller["strategy"]}

@app.post("/api/simulation/start")
async def simulation_start():
    controller["running"] = True
    return {"status": "running"}

@app.post("/api/simulation/pause")
async def simulation_pause():
    controller["running"] = False
    return {"status": "paused"}

@app.post("/api/simulation/reset")
async def simulation_reset():
    controller["sim_time"] = 0.0
    controller["running"] = False
    return {"status": "reset"}

@app.get("/api/agents/status")
async def agents_status():
    return {
        "agent1": controller["agent1"],
        "agent2": controller["agent2"],
        "mcp": "connected" if mcp_client._session else "not_started",
        "sumo": "ready" if sumo_available() else "unavailable",
        "selected_strategy": (controller["strategy"] or {}).get("name"),
    }

@app.get("/api/files")
async def files():
    return {"files":[p.name for p in (ROOT/"data").glob("*") if p.is_file()]}

@app.post("/api/ml/train")
async def trigger_train_ml():
    try:
        res = train_ml_model()
        return res
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

def _http_error(exc: Exception, default_message: str):
    if isinstance(exc, GeminiServiceError):
        return HTTPException(
            status_code=exc.status,
            detail={"code": exc.code, "message": str(exc), "retryable": exc.code in {"GEMINI_UNAVAILABLE", "GEMINI_RATE_LIMIT"}},
        )
    return HTTPException(status_code=500, detail={"code": "SERVER_ERROR", "message": str(exc) or default_message})


@app.post("/api/analyze")
async def analyze(req: SituationRequest):
    try:
        controller["state"] = req.state
        controller["strategy"] = {"name": "Current input", "ns_green": req.state.ns_green, "ew_green": req.state.ew_green}
        controller["source"] = "user input"
        controller["sim_time"] = 0.0
        controller["running"] = False
        controller["agent1"] = "analyzing"
        controller["agent2"] = "generating strategies"
        agent=get_agent()
        analysis=await agent.analyze_situation(req.state,req.description)
        strategies=await agent.generate_strategies(req.state,analysis,req.description)
        controller["agent1"] = "complete"
        controller["agent2"] = "complete"
        return {"agent1":analysis,"strategies":[s.model_dump() for s in strategies]}
    except Exception as exc:
        raise _http_error(exc, "Analysis failed")

@app.post("/api/experiment")
async def experiment(req: ExperimentRequest):
    try:
        controller["state"] = req.state
        controller["agent1"] = "analyzing"
        controller["agent2"] = "testing strategies in SUMO"
        if req.strategies:
            agent=get_agent()
            analysis=await agent.analyze_situation(req.state,req.description)
            baseline_strategy={"name":"Baseline","description":"Current signal timing without intervention.","ns_green":req.state.ns_green,"ew_green":req.state.ew_green}
            base=await mcp_client.call("run_sumo",{"state":req.state.model_dump(),"strategy":baseline_strategy,"label":"baseline"})
            candidates=[]
            for i,s in enumerate(req.strategies,1):
                metrics=await mcp_client.call("run_sumo",{"state":req.state.model_dump(),"strategy":s.model_dump(),"label":f"strategy_{i}"})
                candidates.append({"name":s.name,"description":s.description,"strategy":s.model_dump(),"metrics":metrics})
            evaluation=agent.rank_strategies(base,candidates)
            explanation=await agent.explain(req.state,analysis,base,candidates,evaluation)
            controller["agent1"] = "complete"
            controller["agent2"] = "complete — SUMO evaluation finished"
            controller["last_evaluation"] = evaluation
            selected = next((c["strategy"] for c in candidates if c["name"] == evaluation.get("best_name")), baseline_strategy)
            return {"agent1":analysis,"agent2":{"strategies":candidates},"baseline":{"name":"Baseline","description":baseline_strategy["description"],"strategy":baseline_strategy,"metrics":base},"strategies":candidates,"evaluation":evaluation,"explanation":explanation,"selected_strategy":selected}
        return await get_agent().experiment(req.state,req.description)
    except Exception as exc:
        raise _http_error(exc, "Experiment failed")

@app.post("/api/apply-strategy")
async def apply_strategy(req: ApplyStrategyRequest):
    try:
        if controller["state"] is None:
            raise HTTPException(400, "No traffic state has been configured yet.")
        result = await get_agent().evaluate_and_apply(controller["state"], req.strategy)
        controller["strategy"] = req.strategy.model_dump()
        controller["source"] = "selected SUMO strategy"
        controller["sim_time"] = 0.0
        controller["running"] = True
        controller["agent2"] = "selected strategy applied — final replay running"
        return {**result, "strategy": controller["strategy"], "phase": _phase_status(controller["state"], controller["strategy"], 0.0)}
    except HTTPException:
        raise
    except Exception as exc:
        raise _http_error(exc, "Could not apply strategy")
