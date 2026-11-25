from mcp.server.fastmcp import FastMCP
from app.schemas import TrafficState, Strategy
from app.sumo_engine import run_simulation

mcp = FastMCP("TrafficMind")

@mcp.tool()
def create_scenario(state: dict, strategy: dict) -> dict:
    return {"status": "scenario_ready", "state": state, "strategy": strategy}

@mcp.tool()
def run_sumo(state: dict, strategy: dict, label: str = "scenario") -> dict:
    return run_simulation(TrafficState(**state), Strategy(**strategy), label)

@mcp.tool()
def get_simulation_metrics(result: dict) -> dict:
    return {k: result.get(k, 0) for k in (
        "completed", "throughput", "avg_travel_time",
        "avg_waiting_time", "avg_speed", "avg_queue"
    )}

@mcp.tool()
def compare_metrics(baseline: dict, candidate: dict) -> dict:
    """Return candidate deltas versus baseline. Lower is better for waiting/travel/queue;
    higher is better for throughput and completed vehicles."""
    fields = ("avg_waiting_time", "avg_travel_time", "avg_queue", "throughput", "completed")
    return {
        field: round(float(candidate.get(field, 0)) - float(baseline.get(field, 0)), 2)
        for field in fields
    }

@mcp.tool()
def apply_signal_strategy(strategy: dict, reason: str = "") -> dict:
    """Represent the selected signal-control action for the interactive simulator."""
    s = Strategy(**strategy)
    return {
        "status": "signal_strategy_applied",
        "message": "Interactive traffic-light controller updated.",
        "ns_green": s.ns_green,
        "ew_green": s.ew_green,
        "reason": reason,
    }

@mcp.tool()
def run_traci_live_control(state: dict, strategy: dict = None, label: str = "traci_adaptive") -> dict:
    """Execute Phase 7 Real-Time Closed-Loop TraCI Adaptive Signal Control in SUMO."""
    from app.traci_controller import run_traci_adaptive_simulation
    st = TrafficState(**state)
    strat = Strategy(**strategy) if strategy else None
    return run_traci_adaptive_simulation(st, strat, label)

if __name__ == "__main__":
    mcp.run()
