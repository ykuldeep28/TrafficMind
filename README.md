# TrafficMind — Multi-Agent Interactive AI Traffic Simulator

TrafficMind is a college-project prototype for **AI-agent-based traffic-management strategy evaluation**. It combines:

- **Agent 1 — Traffic Situation Analyst:** interprets traffic volumes and identifies the traffic-management problem.
- **Agent 2 — Traffic Strategy Agent:** uses Agent 1's analysis to generate concrete, testable signal-control strategies.
- **MCP:** exposes scenario, SUMO simulation, metrics and signal-control tools to the application.
- **SUMO:** actually simulates the proposed traffic strategies and produces measured traffic metrics.
- **Interactive UI:** displays an animated four-way intersection with vehicles, traffic lights, current signal timing and simulation state.
- **Strategy application:** after experiments, a tested strategy can be applied to the interactive traffic-light controller so the displayed lights change to the selected plan.

## Architecture

```text
User traffic input
      |
      v
Agent 1: Traffic Situation Analyst
      |
      v
Agent 2: Traffic Strategy Agent
      |
      v
MCP Client <----> MCP Server
                     |
              +------+------+
              |             |
           SUMO tool    signal tool
              |
              v
       SUMO simulation
              |
              v
         Metrics
              |
              v
Agent 2 explanation + UI comparison
```

## What the interactive simulation means

The browser contains a lightweight visual preview of the same four-arm scenario. It shows simulated vehicles and traffic lights and changes the displayed signal timing when a strategy is applied. The **measured performance numbers come from SUMO**, not from the browser animation.

This is a simulation and decision-support prototype. It does not control real traffic infrastructure.

## Requirements

- macOS/Linux/Windows
- Python 3.11+ (the tested setup can use Python 3.11)
- Eclipse SUMO 1.27.1 or another compatible SUMO release
- Gemini API key
- MCP Python SDK **1.x** (`mcp>=1.0,<2`)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.6-flash
SUMO_HOME=/Library/Frameworks/EclipseSUMO.framework/Versions/1.27.1/EclipseSUMO
```

Check SUMO:

```bash
sumo --version
netconvert --version
```

Start:

```bash
python run.py
```

Open **http://localhost:8000**.

## Suggested demo

Use:

- North = 20
- South = 20
- East = 80
- West = 20
- Duration = 180 seconds
- NS green = 34 seconds
- EW green = 30 seconds
- Yellow = 3 seconds

Description:

> East has heavy traffic. Analyze the situation and test signal strategies that reduce queues and improve throughput.

Then:

1. Start the interactive preview.
2. Click **Analyze & Generate Strategies**.
3. Read Agent 1's situation analysis.
4. Review Agent 2's candidate strategies.
5. Click **Run baseline + AI strategies in SUMO**.
6. Compare waiting time, travel time, queue proxy, throughput and completed vehicles.
7. Click **Apply signal** on a tested strategy. The interactive controller changes its displayed NS/EW signal timing.
8. Start/restart the preview to watch the updated signal plan.

## Important metric note

`avg_queue` is a queue/congestion proxy derived from waiting time in this prototype; it is not a direct count of physical vehicles standing in a queue.

A SUMO run of 180 seconds means **180 seconds of simulated traffic**, not 180 seconds of live traffic collection.

## Future extensions

- Real camera input with YOLO vehicle counts
- Live TraCI control of a running SUMO process
- OSM road-network import
- Emergency-vehicle priority
- Historical traffic-data/RAG layer
- More detailed lane-level queue metrics


## How the system identifies the highest-scoring strategy

TrafficMind does not call a strategy "best" just because Gemini says so.

After Baseline and candidate strategies finish in SUMO, the backend calculates a transparent composite score over the **successful runs in that experiment**:

- **40% throughput** — higher is better
- **30% average waiting time** — lower is better
- **20% average travel time** — lower is better
- **10% queue proxy** — lower is better

The UI shows:
1. the highest-scoring tested strategy,
2. its composite score,
3. raw SUMO metrics,
4. changes versus the baseline,
5. completed-vehicle change,
6. the exact scoring method,
7. an **Apply highest-scoring tested strategy** button.

This is useful evidence because the strategy is evaluated using measured SUMO output rather than only an LLM opinion. The result is valid for the simulated scenario and tested assumptions; it is not a guarantee that the same strategy will always be optimal in real traffic.

## Interactive signal application

After evaluation, the user can apply the selected tested signal plan to the interactive browser simulation. The displayed traffic lights and signal timing update to that plan.

The browser animation is a visualization/controller preview. SUMO remains the source of the numerical evaluation. A future production deployment would connect the same agent/MCP control layer to a live TraCI simulation or approved traffic-control API rather than a real traffic signal directly.


### Interactive simulation preview
The browser preview was upgraded to make the traffic state visually understandable: it shows the exact entered approach counts, separate lanes, vehicle spacing, stop lines, crosswalks, three-lamp traffic signals, a live phase/countdown banner, and vehicles that stop before a red/yellow phase and move through the junction only for their green phase. The **Start phase** control can begin with either NS green or EW green.

This browser animation is a lightweight visualization synchronized to the selected signal plan; it is **not** a live TraCI view of the SUMO process. The measured strategy comparison still runs in SUMO. The start phase is also passed into the SUMO signal program so the experiment and preview can use the same initial phase.

## Gemini error handling

TrafficMind now handles common Gemini failures without hiding the reason from the user:

- **Token/context limit exceeded** → the UI reports `GEMINI_TOKEN_LIMIT` and explains that the request is too large; it does not waste retries on the same oversized request.
- **503 / temporary high demand** → the backend retries transiently up to three times with short backoff, then reports that Gemini is temporarily unavailable.
- **429 / rate limit or quota** → the backend retries transiently, then reports the rate-limit/quota condition.
- **401/403 / API key or permission problem** → the UI reports a Gemini authentication error and points to `GEMINI_API_KEY`.
- **Unknown Gemini error** → the UI shows a clear Gemini request error and the server log contains the underlying exception.

The `/api/analyze`, `/api/experiment`, and strategy-application endpoints return structured error details so the browser can show the actual category instead of a generic `[object Object]` or unexplained HTTP 500.
