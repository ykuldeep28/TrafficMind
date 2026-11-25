import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any
from .schemas import TrafficState, Strategy

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

def _sumo_bin(name="sumo"):
    found = shutil.which(name)
    if found:
        return found
    home = os.getenv("SUMO_HOME")
    if home:
        candidate = Path(home) / "bin" / name
        if candidate.exists():
            return str(candidate)
    return None

def sumo_available():
    return _sumo_bin() is not None

def _write_network(work: Path):
    nodes = work / "nodes.nod.xml"
    edges = work / "edges.edg.xml"
    net = work / "intersection.net.xml"
    nodes.write_text("""<nodes>
  <node id="C" x="0" y="0" type="traffic_light"/>
  <node id="N" x="0" y="500"/>
  <node id="S" x="0" y="-500"/>
  <node id="E" x="500" y="0"/>
  <node id="W" x="-500" y="0"/>
</nodes>""")
    edges.write_text("""<edges>
  <edge id="N2C" from="N" to="C" numLanes="1" speed="13.9"/>
  <edge id="C2N" from="C" to="N" numLanes="1" speed="13.9"/>
  <edge id="S2C" from="S" to="C" numLanes="1" speed="13.9"/>
  <edge id="C2S" from="C" to="S" numLanes="1" speed="13.9"/>
  <edge id="E2C" from="E" to="C" numLanes="1" speed="13.9"/>
  <edge id="C2E" from="C" to="E" numLanes="1" speed="13.9"/>
  <edge id="W2C" from="W" to="C" numLanes="1" speed="13.9"/>
  <edge id="C2W" from="C" to="W" numLanes="1" speed="13.9"/>
</edges>""")
    netconvert = _sumo_bin("netconvert")
    if not netconvert:
        raise RuntimeError("netconvert was not found. Check SUMO_HOME/PATH.")
    subprocess.run([
        netconvert, "--node-files", str(nodes), "--edge-files", str(edges),
        "--output-file", str(net), "--tls.guess", "true", "--no-turnarounds", "true"
    ], check=True, capture_output=True, text=True)
    return net

def _get_tls_link_count(net: Path, tls_id="C"):
    root = ET.parse(net).getroot()
    indices = []
    for c in root.findall("connection"):
        if c.attrib.get("tl") == tls_id and "linkIndex" in c.attrib:
            try:
                indices.append(int(c.attrib["linkIndex"]))
            except ValueError:
                pass
    if indices:
        return max(indices) + 1
    raise RuntimeError(f"Could not determine controlled-link count for traffic light '{tls_id}'.")

def _write_signal(work: Path, net: Path, ns_green: int, ew_green: int, yellow: int, start_phase: str = "NS"):
    add = work / "signal.add.xml"
    ns_green = max(5, int(ns_green)); ew_green = max(5, int(ew_green)); yellow = max(1, int(yellow))
    n = _get_tls_link_count(net)
    if n < 4:
        raise RuntimeError(f"Traffic light C has {n} controlled links; expected at least 4.")
    extra = "r" * (n - 4)
    states = ["GGrr" + extra, "yyrr" + extra, "rrGG" + extra, "rryy" + extra]
    offset = ns_green + yellow if str(start_phase).upper() == "EW" else 0
    add.write_text(f"""<additional>
  <tlLogic id="C" type="static" programID="trafficmind" offset="{offset}">
    <phase duration="{ns_green}" state="{states[0]}"/>
    <phase duration="{yellow}" state="{states[1]}"/>
    <phase duration="{ew_green}" state="{states[2]}"/>
    <phase duration="{yellow}" state="{states[3]}"/>
  </tlLogic>
</additional>""")
    return add

def _write_routes(work: Path, state: TrafficState, duration: int):
    route_file = work / "traffic.rou.xml"
    closed = (getattr(state, "closed_road", "none") or "none").upper()
    emergency = (getattr(state, "emergency_approach", "none") or "none").upper()

    counts = {"N": state.north, "S": state.south, "E": state.east, "W": state.west}
    all_arms = ["N", "S", "E", "W"]

    def get_valid_routes(origin):
        exits = [arm for arm in all_arms if arm != origin and (closed == "none" or arm != closed)]
        if not exits:
            exits = [arm for arm in all_arms if arm != origin]
        return [f"{origin}2C C2{dest}" for dest in exits]

    route_lines = []
    vehicle_lines = []
    rid = 0
    vid = 0

    if emergency in all_arms:
        em_routes = get_valid_routes(emergency)
        if em_routes:
            em_route = em_routes[0]
            route_lines.append(f'  <route id="r_emergency" edges="{em_route}"/>')
            vehicle_lines.append(f'  <vehicle id="v_emergency" type="emergency" route="r_emergency" depart="3"/>')
            vid += 1

    for origin, count in counts.items():
        if count <= 0:
            continue
        valid_routes = get_valid_routes(origin)
        is_closed_arm = (closed == origin)
        effective_count = count if not is_closed_arm else max(1, count // 3)

        for i in range(effective_count):
            route = valid_routes[i % len(valid_routes)]
            route_lines.append(f'  <route id="r{rid}" edges="{route}"/>')
            vtype = "bottlenecked_car" if is_closed_arm else "car"
            depart = min(int((i + 1) * (duration * 0.8 / max(1, effective_count))), max(0, duration - 2))
            vehicle_lines.append(f'  <vehicle id="v{vid}" type="{vtype}" route="r{rid}" depart="{depart}"/>')
            rid += 1
            vid += 1

    route_file.write_text("""<routes>
  <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="13.9"/>
  <vType id="bottlenecked_car" accel="1.0" decel="4.5" sigma="0.8" length="5" minGap="3.5" maxSpeed="4.5"/>
  <vType id="emergency" accel="4.0" decel="6.5" sigma="0.1" length="6.5" minGap="1.5" maxSpeed="22.2" guiShape="emergency" vClass="emergency"/>
""" + "\n".join(route_lines) + "\n" + "\n".join(vehicle_lines) + "\n</routes>")
    return route_file, vid

def _write_config(work, net, routes, signal, duration):
    cfg=work/"run.sumocfg"
    cfg.write_text(f"""<configuration>
  <input><net-file value="{net.name}"/><route-files value="{routes.name}"/><additional-files value="{signal.name}"/></input>
  <time><begin value="0"/><end value="{duration}"/><step-length value="1"/></time>
  <output><tripinfo-output value="tripinfo.xml"/></output>
  <processing><time-to-teleport value="-1"/></processing>
</configuration>""")
    return cfg

def _metrics(tripinfo: Path, inserted: int, duration: int) -> Dict[str, Any]:
    completed=0; travel=[]; waiting=[]; speeds=[]
    emergency_wait = None; emergency_travel = None; emergency_cleared = False
    if tripinfo.exists():
        root=ET.parse(tripinfo).getroot()
        for item in root.findall("tripinfo"):
            completed += 1
            dur=float(item.attrib.get("duration",0)); wait=float(item.attrib.get("waitingTime",0)); distance=float(item.attrib.get("routeLength",0))
            vtype=item.attrib.get("vType",""); vid=item.attrib.get("id","")
            travel.append(dur); waiting.append(wait)
            if dur>0: speeds.append(distance/dur)
            if vtype=="emergency" or vid=="v_emergency":
                emergency_cleared=True
                emergency_wait=round(wait, 2)
                emergency_travel=round(dur, 2)
    return {
        "vehicles_inserted": inserted, "completed": completed,
        "throughput": round(completed/max(1,duration)*3600,2),
        "avg_travel_time": round(sum(travel)/len(travel),2) if travel else 0.0,
        "avg_waiting_time": round(sum(waiting)/len(waiting),2) if waiting else 0.0,
        "avg_speed": round(sum(speeds)/len(speeds),2) if speeds else 0.0,
        "avg_queue": round(sum(waiting)/max(1,duration),2) if waiting else 0.0,
        "simulation_seconds": duration,
        "emergency_cleared": emergency_cleared,
        "emergency_waiting_time": emergency_wait,
        "emergency_travel_time": emergency_travel,
        "status":"ok"
    }

def run_simulation(state: TrafficState, strategy: Strategy, label: str):
    if not sumo_available():
        return {"status":"SUMO_NOT_FOUND","vehicles_inserted":0,"completed":0,"throughput":0.0,"avg_travel_time":0.0,"avg_waiting_time":0.0,"avg_speed":0.0,"avg_queue":0.0,"simulation_seconds":state.duration}
    work=Path(tempfile.mkdtemp(prefix="trafficmind_",dir=str(DATA_DIR)))
    try:
        net=_write_network(work)
        signal=_write_signal(work,net,strategy.ns_green,strategy.ew_green,state.yellow,state.start_phase)
        routes,inserted=_write_routes(work,state,state.duration)
        cfg=_write_config(work,net,routes,signal,state.duration)
        result=subprocess.run([_sumo_bin("sumo"),"-c",str(cfg),"--no-step-log","true","--quit-on-end","true"],capture_output=True,text=True,timeout=max(120,state.duration*3))
        if result.returncode != 0:
            return {"status":"SUMO_ERROR","error":result.stderr[-2000:],"vehicles_inserted":inserted,"completed":0,"throughput":0.0,"avg_travel_time":0.0,"avg_waiting_time":0.0,"avg_speed":0.0,"avg_queue":0.0,"simulation_seconds":state.duration}
        metrics=_metrics(work/"tripinfo.xml",inserted,state.duration)
        if (work/"tripinfo.xml").exists(): shutil.copy2(work/"tripinfo.xml",DATA_DIR/f"{label}_tripinfo.xml")
        return metrics
    except subprocess.TimeoutExpired:
        return {"status":"SUMO_TIMEOUT","vehicles_inserted":0,"completed":0,"throughput":0.0,"avg_travel_time":0.0,"avg_waiting_time":0.0,"avg_speed":0.0,"avg_queue":0.0,"simulation_seconds":state.duration}
    except Exception as exc:
        return {"status":"SUMO_ERROR","error":str(exc),"vehicles_inserted":0,"completed":0,"throughput":0.0,"avg_travel_time":0.0,"avg_waiting_time":0.0,"avg_speed":0.0,"avg_queue":0.0,"simulation_seconds":state.duration}
    finally:
        shutil.rmtree(work,ignore_errors=True)
