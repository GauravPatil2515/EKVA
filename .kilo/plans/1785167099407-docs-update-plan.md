# Doc Update Plan

## Goal
Create/update documentation markdown files in `docs/` to reflect the reframed research direction based on prior-work analysis and external review.

## Files to Create/Update

### Create: `docs/RESEARCH.md`
- Reframed research framing (empirical characterization, not novel method)
- Four research questions (RQ1–RQ4) with detailed descriptions
- Prior work landscape table (PiKV, CAKE, Ada-KV, MEDA, InfoKV, TriRoute, MoE-nD, etc.)
- Detailed prior work analysis (PiKV, InfoKV, MoE-nD, TriRoute)
- What to cut/de-prioritize table
- Realistic target and honest framing section
- Immediate next steps

### Update: `PLAN.md` (root)
- Add framing header explaining reframing
- Add prior work context section
- Reorganize weeks around RQ1–RQ4
- De-prioritize full benchmark sweep (2 tasks only)
- Add RQ1 as Week 1–2 priority (CPU only)
- Add RQ2 as Weeks 2–3
- Keep Weeks 4–11 structure but tag RQ4 items
- Update fallback branches

### Already exists: `docs/review.md`
- External review with reframing recommendations (no changes needed)

## Verification
- Confirm `docs/RESEARCH.md` exists and has content
- Confirm `PLAN.md` has the reframed framing header
- Confirm no source code files were modified
