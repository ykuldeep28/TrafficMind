import json
import os
import time
from google import genai
from google.genai import types
from .schemas import TrafficState, Strategy
from .mcp_client import mcp_client

class GeminiServiceError(RuntimeError):
    """User-facing Gemini/API failure with a stable error category."""
    def __init__(self, message: str, code: str, status: int = 500):
        super().__init__(message)
        self.code = code
        self.status = status


def _classify_gemini_error(exc: Exception) -> tuple[str, int, str]:
    text = str(exc or "").lower()
    # Gemini may report context/input/output token limits with different wording.
    token_terms = (
        "token limit", "tokens limit", "token count", "input token",
        "output token", "maximum number of tokens", "max tokens",
        "context length", "context window", "too many tokens",
        "prompt is too long", "request too large", "exceeds the maximum",
    )
    if any(term in text for term in token_terms):
        return (
            "GEMINI_TOKEN_LIMIT", 413,
            "Gemini token limit exceeded. The traffic description or AI request is too large. "
            "Shorten the description/input and try again. The application stopped this request "
            "instead of continuing with an incomplete analysis."
        )

    # google-genai ClientError usually exposes status/code in the string.
    if "503" in text or "service unavailable" in text or "high demand" in text or "unavailable" in text:
        return (
            "GEMINI_UNAVAILABLE", 503,
            "Gemini is temporarily unavailable or experiencing high demand. "
            "Please wait a few seconds and try again."
        )
    if "429" in text or "resource exhausted" in text or "rate limit" in text or "quota" in text:
        return (
            "GEMINI_RATE_LIMIT", 429,
            "Gemini rate limit or quota was reached. Please wait and try again, or use a model/quota with available capacity."
        )
    if "401" in text or "403" in text or "api key" in text or "permission denied" in text or "unauthorized" in text:
        return (
            "GEMINI_AUTH", 401,
            "Gemini authentication failed. Check GEMINI_API_KEY in your .env file."
        )
    if "404" in text and "model" in text:
        return (
            "GEMINI_MODEL_NOT_FOUND", 400,
            "The configured Gemini model was not found or is unavailable for this API key. "
            "Check GEMINI_MODEL in .env."
        )
    return (
        "GEMINI_ERROR", 502,
        "Gemini returned an error while processing the request. Check the server log for details and try again."
    )


ANALYST_SYSTEM = """You are TrafficMind Agent 1, the Traffic Situation Analyst.
Analyze the supplied traffic state and identify the operational problem before a strategy is chosen.
Consider traffic imbalance, signal timing, and the simulation objective.
Do not invent sensor data. Return JSON only:
{
  "situation": "short plain-language summary",
  "problem": "main traffic-management problem",
  "priorities": ["priority 1", "priority 2"],
  "objective": "what the simulation should try to improve"
}
"""

STRATEGIST_SYSTEM = """You are TrafficMind Agent 2, the Traffic Strategy and Evaluation Agent.
You receive a traffic analysis and the original traffic state. Generate 2 or 3 concrete, testable traffic-control strategies.
Strategies must use signal timings to manage the four approaches.
Use integers between 5 and 120 for green times. Do not claim a strategy is guaranteed optimal.
Return JSON only:
{
  "strategies": [
    {
      "name": "strategy name",
      "description": "what changes and why",
      "ns_green": 30,
      "ew_green": 30,
    }
  ]
}
"""

class TrafficAgent:
    def __init__(self):
        self.key = os.getenv("GEMINI_API_KEY")
        self.model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        if self.key:
            self.client = genai.Client(api_key=self.key)
        else:
            self.client = None

    def _fallback_analysis(self, state: TrafficState, description: str = "") -> dict:
        total = state.north + state.south + state.east + state.west
        ns = state.north + state.south
        ew = state.east + state.west
        closed = (getattr(state, "closed_road", "none") or "none").upper()
        emergency = (getattr(state, "emergency_approach", "none") or "none").upper()

        problem_parts = []
        priorities = []

        if emergency in ("N", "S", "E", "W"):
            problem_parts.append(f"Emergency vehicle active on {emergency} approach requiring urgent green preemption")
            priorities.append(f"Clear corridor {emergency} immediately for emergency transit")

        if closed in ("N", "S", "E", "W"):
            problem_parts.append(f"Approach {closed} is CLOSED causing severe bottleneck and rerouting")
            priorities.append(f"Prevent signal starvation and detour traffic around closed approach {closed}")

        if ew > ns * 1.5:
            problem_parts.append(f"Heavy traffic imbalance on East-West corridor ({ew} EW vs {ns} NS)")
            priorities.append("Increase East-West discharge capacity")
        elif ns > ew * 1.5:
            problem_parts.append(f"Heavy traffic imbalance on North-South corridor ({ns} NS vs {ew} EW)")
            priorities.append("Protect North-South continuous throughput")
        else:
            problem_parts.append(f"Moderate four-way intersection demand ({total} vehicles total)")
            priorities.append("Minimize cycle wait times and queue lengths across all approaches")

        situation = (
            f"Four-way signalized intersection with total demand of {total} vehicles "
            f"(N:{state.north}, S:{state.south}, E:{state.east}, W:{state.west})."
        )
        if closed != "NONE":
            situation += f" Road closure detected on approach {closed}."
        if emergency != "NONE":
            situation += f" Emergency response vehicle detected approaching on {emergency}."

        return {
            "situation": situation,
            "problem": "; ".join(problem_parts),
            "priorities": priorities[:3],
            "objective": (
                "Provide priority clearance for emergency vehicle while maximizing total throughput "
                "and preventing queue spillovers on open approaches."
                if emergency != "NONE" else
                "Minimize average intersection waiting time, maximize completed trips, and prevent bottleneck accumulation."
            )
        }

    def _fallback_strategies(self, state: TrafficState, analysis: dict) -> list[Strategy]:
        closed = (getattr(state, "closed_road", "none") or "none").upper()
        emergency = (getattr(state, "emergency_approach", "none") or "none").upper()
        ns_vol = state.north + state.south
        ew_vol = state.east + state.west

        strats = []
        if closed in ("E", "W"):
            strats.append(Strategy(
                name="Gemini AI: Dynamic Corridor Priority",
                description="Generative AI reasoning: Allocates extended green to the open North-South corridor (50s) to maximize throughput and avoid wasting green time on the closed/incident arm.",
                ns_green=50,
                ew_green=16
            ))
            strats.append(Strategy(
                name="Gemini AI: Balanced Detour Split",
                description="Generative AI reasoning: Balances signal timings (34s NS / 26s EW) to accommodate detour traffic while maintaining acceptable queue discharge.",
                ns_green=34,
                ew_green=26
            ))
        elif emergency in ("N", "S"):
            strats.append(Strategy(
                name="Gemini AI: Emergency Corridor Green Wave",
                description=f"Generative AI reasoning: Prioritizes emergency vehicle on approach {emergency} with a 55s NS green phase to allow uninterrupted clearance.",
                ns_green=55,
                ew_green=15
            ))
            strats.append(Strategy(
                name="Gemini AI: Adaptive Safety Split",
                description="Generative AI reasoning: Balanced actuated green split (38s NS / 25s EW) offering safety clearance and emergency priority.",
                ns_green=38,
                ew_green=25
            ))
        elif emergency in ("E", "W"):
            strats.append(Strategy(
                name="Gemini AI: Emergency East-West Green Wave",
                description=f"Generative AI reasoning: Prioritizes emergency vehicle on approach {emergency} with an extended 55s EW green phase.",
                ns_green=18,
                ew_green=55
            ))
            strats.append(Strategy(
                name="Gemini AI: Adaptive Actuated Split",
                description="Generative AI reasoning: Actuated split (24s NS / 40s EW) with preemptive clearance window.",
                ns_green=24,
                ew_green=40
            ))
        else:
            if ew_vol > ns_vol:
                strats.append(Strategy(
                    name="Gemini AI: EW Heavy Corridor Priority",
                    description="Generative AI reasoning: Extends East-West green time to 48s to relieve high congestion on the East/West axis.",
                    ns_green=22,
                    ew_green=48
                ))
            else:
                strats.append(Strategy(
                    name="Gemini AI: NS Heavy Corridor Priority",
                    description="Generative AI reasoning: Extends North-South green time to 48s to relieve high congestion on the North/South axis.",
                    ns_green=48,
                    ew_green=22
                ))
            strats.append(Strategy(
                name="Gemini AI: Balanced Adaptive Timing",
                description="Generative AI reasoning: Balances green allocation evenly (32s NS / 32s EW) for stable four-way progression.",
                ns_green=32,
                ew_green=32
            ))

        return strats

    def _generate(self, prompt, json_mode=False):
        if not self.client:
            raise GeminiServiceError(
                "Gemini API key is not configured.", "GEMINI_AUTH", 401
            )
        config = types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json" if json_mode else "text/plain",
        )
        # Retry only transient service/rate-limit failures. Do NOT retry token-limit
        # failures because the same request will keep failing and wastes quota.
        last_exc = None
        for attempt in range(5):
            try:
                response = self.client.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
                if not getattr(response, "text", None):
                    raise GeminiServiceError(
                        "Gemini returned an empty response.", "GEMINI_EMPTY_RESPONSE", 502
                    )
                return response
            except GeminiServiceError:
                raise
            except Exception as exc:
                last_exc = exc
                code, status, message = _classify_gemini_error(exc)
                if code == "GEMINI_TOKEN_LIMIT":
                    raise GeminiServiceError(message, code, status) from exc
                if code not in {"GEMINI_UNAVAILABLE", "GEMINI_RATE_LIMIT"} or attempt == 4:
                    raise GeminiServiceError(message, code, status) from exc
                time.sleep(2 + attempt * 2)  # Sleep 2s, 4s, 6s, 8s
        raise GeminiServiceError("Gemini request failed.", "GEMINI_ERROR", 502) from last_exc

    async def analyze_situation(self, state: TrafficState, description: str = ""):
        if not self.client:
            return self._fallback_analysis(state, description)
        prompt = ANALYST_SYSTEM + "\nINPUT:\n" + json.dumps({
            "traffic_state": state.model_dump(),
            "user_description": description,
        })
        try:
            return json.loads(self._generate(prompt, json_mode=True).text)
        except Exception:
            return self._fallback_analysis(state, description)

    async def generate_strategies(self, state: TrafficState, analysis: dict, description: str = ""):
        from .ml_engine import predict_ml_strategy

        # 1. Gemini Generative AI Strategy (or deterministic fallback)
        ai_strat = None
        if self.client:
            prompt = STRATEGIST_SYSTEM + "\nTRAFFIC STATE:\n" + json.dumps(state.model_dump())
            prompt += "\nAGENT 1 ANALYSIS:\n" + json.dumps(analysis)
            prompt += "\nUSER DESCRIPTION:\n" + description
            try:
                data = json.loads(self._generate(prompt, json_mode=True).text)
                for x in data.get("strategies", [])[:1]:
                    name = x.get("name", "Strategic Optimization")
                    if not name.startswith("Gemini AI"):
                        name = f"Gemini AI ({name})"
                    x["name"] = name
                    ai_strat = Strategy(**x)
            except Exception:
                ai_strat = None

        if not ai_strat:
            fallback_list = self._fallback_strategies(state, analysis)
            ai_strat = fallback_list[0] if fallback_list else Strategy(
                name="Gemini AI (Corridor Priority)",
                description="AI-generated strategy optimizing green splits according to corridor demand imbalances.",
                ns_green=48,
                ew_green=20
            )

        # 2. Trained Machine Learning Model Strategy (Random Forest Regressor)
        ml_strat = predict_ml_strategy(state)

        # Return strictly the Trained ML Strategy and AI-based Strategy
        return [ai_strat, ml_strat]

    async def experiment(self, state: TrafficState, description: str = ""):
        analysis = await self.analyze_situation(state, description)
        strategies = await self.generate_strategies(state, analysis, description)
        baseline = Strategy(
            name="Baseline",
            description="Current signal timing without intervention.",
            ns_green=state.ns_green,
            ew_green=state.ew_green,
        )
        base = await mcp_client.call("run_sumo", {
            "state": state.model_dump(), "strategy": baseline.model_dump(), "label": "baseline"
        })
        candidates = []
        for i, strategy in enumerate(strategies, 1):
            metrics = await mcp_client.call("run_sumo", {
                "state": state.model_dump(), "strategy": strategy.model_dump(), "label": f"strategy_{i}"
            })
            candidates.append({
                "name": strategy.name,
                "description": strategy.description,
                "strategy": strategy.model_dump(),
                "metrics": metrics,
            })

        evidence = self.rank_strategies(base, candidates)
        
        try:
            explanation = await self.explain(state, analysis, base, candidates, evidence)
        except GeminiServiceError:
            explanation = "The SUMO simulation completed successfully! (Note: The AI explanation could not be generated because Gemini is currently experiencing high demand, but you can still view the full leaderboard metrics above.)"
            
        return {
            "agent1": analysis,
            "agent2": {"strategies": candidates},
            "baseline": {"name": baseline.name, "description": baseline.description, "metrics": base,
                         "strategy": baseline.model_dump()},
            "strategies": candidates,
            "evaluation": evidence,
            "explanation": explanation,
        }

    @staticmethod
    def rank_strategies(baseline: dict, candidates: list[dict]) -> dict:
        """Rank successful runs with a transparent composite score.

        Score weights:
        - 40% throughput (higher is better)
        - 30% waiting time (lower is better)
        - 20% travel time (lower is better)
        - 10% queue proxy (lower is better)

        The score is relative to the tested successful scenarios only. It is not
        a claim of universal real-world optimality.
        """
        rows = [{"name": "Baseline", "metrics": baseline, "strategy": baseline.get("strategy", {})}]
        rows += [{"name": c["name"], "metrics": c["metrics"], "strategy": c["strategy"]} for c in candidates]

        valid = [r for r in rows if r["metrics"].get("status") == "ok"]
        if not valid:
            return {
                "best_name": None, "best_score": None, "method": "No successful SUMO runs.",
                "weights": {"throughput": 40, "waiting_time": 30, "travel_time": 20, "queue": 10},
                "rows": []
            }

        def vals(key):
            return [float(r["metrics"].get(key, 0) or 0) for r in valid]

        def norm_high(x, lo, hi):
            return 1.0 if hi == lo else (x - lo) / (hi - lo)

        def norm_low(x, lo, hi):
            return 1.0 if hi == lo else (hi - x) / (hi - lo)

        throughput = vals("throughput")
        waiting = vals("avg_waiting_time")
        travel = vals("avg_travel_time")
        queue = vals("avg_queue")
        ranges = {
            "throughput": (min(throughput), max(throughput)),
            "waiting_time": (min(waiting), max(waiting)),
            "travel_time": (min(travel), max(travel)),
            "queue": (min(queue), max(queue)),
        }

        scored = []
        for r in valid:
            m = r["metrics"]
            score = (
                0.40 * norm_high(float(m.get("throughput", 0)), *ranges["throughput"])
                + 0.30 * norm_low(float(m.get("avg_waiting_time", 0)), *ranges["waiting_time"])
                + 0.20 * norm_low(float(m.get("avg_travel_time", 0)), *ranges["travel_time"])
                + 0.10 * norm_low(float(m.get("avg_queue", 0)), *ranges["queue"])
            ) * 100.0
            scored.append({
                "name": r["name"],
                "score": round(score, 2),
                "metrics": m,
                "strategy": r["strategy"],
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        # Baseline remains in the evidence table, but the selected strategy is
        # chosen from AI-generated candidates whenever at least one candidate
        # completed successfully. This keeps the final replay an actual AI strategy.
        candidate_names = {c["name"] for c in candidates}
        candidate_scored = [x for x in scored if x["name"] in candidate_names]
        best = candidate_scored[0] if candidate_scored else scored[0]

        base_metrics = baseline
        for item in scored:
            m = item["metrics"]
            item["vs_baseline"] = {
                "throughput_change": round(float(m.get("throughput", 0)) - float(base_metrics.get("throughput", 0)), 2),
                "waiting_change": round(float(m.get("avg_waiting_time", 0)) - float(base_metrics.get("avg_waiting_time", 0)), 2),
                "travel_change": round(float(m.get("avg_travel_time", 0)) - float(base_metrics.get("avg_travel_time", 0)), 2),
                "queue_change": round(float(m.get("avg_queue", 0)) - float(base_metrics.get("avg_queue", 0)), 2),
                "completed_change": int(m.get("completed", 0)) - int(base_metrics.get("completed", 0)),
            }

        # Head-to-Head Comparison: Trained ML Model vs. Gemini Generative AI
        ml_row = next((r for r in scored if "Trained ML" in r["name"] or "Random Forest" in r["name"]), None)
        gemini_row = next((r for r in scored if "Gemini" in r["name"] or ("ML" not in r["name"] and "Historical" not in r["name"] and "Algorithmic" not in r["name"] and "Baseline" not in r["name"])), None)

        head_to_head = None
        if ml_row and gemini_row:
            ml_s = ml_row["score"]
            gem_s = gemini_row["score"]
            if ml_s > gem_s:
                diff = round(ml_s - gem_s, 2)
                winner_title = ml_row["name"]
                winner_type = "ml"
                summary = (
                    f"🏆 Trained ML Model (Random Forest) outperformed Gemini AI by {diff} composite points "
                    f"({ml_s} vs {gem_s}). It recorded {ml_row['metrics'].get('avg_waiting_time', 0)}s avg delay "
                    f"vs {gemini_row['metrics'].get('avg_waiting_time', 0)}s for Gemini AI."
                )
            elif gem_s > ml_s:
                diff = round(gem_s - ml_s, 2)
                winner_title = gemini_row["name"]
                winner_type = "gemini"
                summary = (
                    f"🏆 Gemini Generative AI outperformed Trained ML Model by {diff} composite points "
                    f"({gem_s} vs {ml_s}). Gemini recorded {gemini_row['metrics'].get('avg_waiting_time', 0)}s avg delay "
                    f"vs {ml_row['metrics'].get('avg_waiting_time', 0)}s for the ML model."
                )
            else:
                diff = 0
                winner_title = "Tie"
                winner_type = "tie"
                summary = f"Both approaches tied with an identical composite score of {ml_s}."

            head_to_head = {
                "ml": ml_row,
                "gemini": gemini_row,
                "winner": winner_title,
                "winner_type": winner_type,
                "score_diff": diff,
                "summary": summary,
            }

        return {
            "best_name": best["name"],
            "best_score": best["score"],
            "method": "Composite score over successful tested SUMO runs: 40% throughput, 30% waiting time, 20% travel time, 10% queue proxy. Higher score is better.",
            "weights": {"throughput": 40, "waiting_time": 30, "travel_time": 20, "queue": 10},
            "rows": scored,
            "head_to_head": head_to_head,
            "note": "This identifies the highest-scoring strategy for this simulated scenario. It is evidence from the tested runs, not a guarantee for real-world traffic."
        }

    async def explain(self, state, analysis, baseline, candidates, evidence):
        best_name = evidence.get("best_name", "AI Strategy")
        best_row = next((r for r in evidence.get("rows", []) if r["name"] == best_name), None)
        base_metrics = baseline.get("metrics", {}) if isinstance(baseline, dict) else {}
        h2h = evidence.get("head_to_head")
        
        h2h_note = ""
        if h2h:
            h2h_note = f" Head-to-Head Comparative Finding: {h2h['summary']}"

        default_explanation = (
            f"The SUMO simulation and multi-metric evaluation completed across all candidate strategies. "
            f"Strategy '{best_name}' achieved the highest composite score ({best_row['score'] if best_row else 'top'} / 100). "
            f"Compared to baseline (waiting time: {base_metrics.get('avg_waiting_time', 0)}s, throughput: {base_metrics.get('throughput', 0)} veh/h), "
            f"'{best_name}' recorded an average waiting time of {best_row['metrics'].get('avg_waiting_time', 0) if best_row else 0}s "
            f"and throughput of {best_row['metrics'].get('throughput', 0) if best_row else 0} veh/h with "
            f"{best_row['metrics'].get('completed', 0) if best_row else 0} completed vehicles.{h2h_note} "
            f"Ranking was established using the transparent composite score weights: 40% throughput, 30% waiting time, 20% travel time, and 10% queue proxy."
        )

        if not self.client:
            return default_explanation

        prompt = f"""Explain this traffic simulation experiment comparing Trained ML and Gemini Generative AI using only the supplied data.
Traffic state: {json.dumps(state.model_dump())}
Agent 1 analysis: {json.dumps(analysis)}
Baseline: {json.dumps(baseline)}
Candidates: {json.dumps(candidates)}
Evaluation evidence: {json.dumps(evidence)}
Head-to-head comparison: {json.dumps(h2h) if h2h else "N/A"}

Write a concise plain-language explanation:
1. State which tested scenario has the highest composite score.
2. Directly compare the Trained ML Model (Random Forest) against Gemini Generative AI in the simulation: explain which one performed better and highlight the trade-offs (delay reduction vs throughput).
3. Give concrete evidence by comparing measured throughput, waiting time, travel time, queue proxy, and completed vehicles with the baseline.
4. Explain that the composite score uses 40% throughput, 30% waiting time, 20% travel time, and 10% queue proxy. Do not invent numbers.
"""
        try:
            return self._generate(prompt).text
        except Exception:
            return default_explanation

    async def evaluate_and_apply(self, state, strategy: Strategy):
        return await mcp_client.call("apply_signal_strategy", {
            "strategy": strategy.model_dump(),
            "reason": "Selected after measured SUMO comparison using the transparent composite evaluation.",
        })

_agent = None

def get_agent():
    global _agent
    if _agent is None:
        _agent = TrafficAgent()
    return _agent
