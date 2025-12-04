# project_tools/schedule_completeness_tool.py
import json
from datetime import datetime, timedelta

class ScheduleCompletenessTool:
    """
    name: schedule_completeness
    description: |
      Summarize coverage as pass/fail with an hour-by-hour checklist.
      Input MUST be JSON with:
      {
        "coverage_result": {"coverage_rate": float, "gaps": [...]},
        "horizon_start": ISO8601,
        "horizon_end": ISO8601,
        "slot_minutes": int (optional, default 60)
      }
      Returns JSON: {"passed": bool, "covered_hours": int, "total_hours": int, "checklist": [...]}
    """
    name = "schedule_completeness"
    description = __doc__

    def use(self, tool_input: str) -> str:
        try:
            data = json.loads(tool_input)
            coverage_result = data["coverage_result"]
            horizon_start = data["horizon_start"]; horizon_end = data["horizon_end"]
            slot_minutes = int(data.get("slot_minutes", 60))

            def parse_iso(ts): return datetime.fromisoformat(ts)

            # Build slots
            t = parse_iso(horizon_start); end = parse_iso(horizon_end)
            delta = timedelta(minutes=slot_minutes)
            slots = []
            while t < end:
                slots.append((t, t + delta))
                t += delta

            # Index gaps per slot
            gaps_by_slot = {i: [] for i in range(len(slots))}
            for g in coverage_result.get("gaps", []):
                gs, ge = parse_iso(g["slot_start"]), parse_iso(g["slot_end"])
                for i, (s, e) in enumerate(slots):
                    if gs == s and ge == e:
                        station = g.get("station_id", "unknown")
                        role = g.get("missing", {}).get("role", "?")
                        gaps_by_slot[i].append(f"{station}:{role}")

            total = len(slots)
            covered = sum(1 for i in range(total) if not gaps_by_slot[i])
            passed = (covered == total)

            checklist = []
            for i, (s, e) in enumerate(slots):
                missing_roles = ", ".join(gaps_by_slot[i]) if gaps_by_slot[i] else ""
                status = "OK" if not gaps_by_slot[i] else f"GAP ({missing_roles})"
                checklist.append({"slot_start": s.isoformat(), "slot_end": e.isoformat(), "status": status})

            return json.dumps({"passed": passed, "covered_hours": covered, "total_hours": total, "checklist": checklist})
        except Exception as e:
            return f"Error: {e}"
