# ---------- MODULE 1 (Scenario + Preset Values) ----------
# Drop this near the top of your app.py (after imports).
# This defines your entire scenario in one place + provides a clean UI renderer.

import pandas as pd
import streamlit as st

PERIODS = ["B1", "B2", "B3", "B4", "B5", "B6"]

def init_module1_params():
    """
    Module 1: Preset values (Final locked scenario)
    Returns a dict of parameters used by all later modules.
    """
    return {
        # Horizon
        "periods": PERIODS,

        # Initial state
        "W0": 0,
        "Inv0": 10000,

        # Base demand (Aug–Oct, 6 bi-weeks)
        "base_demand": [18000, 22000, 28000, 32000, 45000, 52000],

        # Demand uncertainty (for Module 4 stress testing)
        "mc_runs_default": 200,
        "demand_shock_default": 0.10,  # ±10%

        # Productivity & OT
        "p": 1200,        # units/worker/bi-week
        "ot_max": 0.20,   # 20%
        "new_hire_productivity_factor": 0.50,  # 50% in hire bi-week (then 100% next)

        # Kiln capacity
        "kiln_design": 30000,
        "alpha": 0.85,
        "rent_kiln_add": 10000,
        "rent_kiln_cost": 40000,  # ₹ per bi-week

        # Warehouse capacity (base + rental toggle)
        "wh_base": 25000,
        "rent_wh_add": 30000,
        "rent_wh_cost": 30000,    # ₹ per bi-week

        # Inventory overflow handling (forced liquidation)
        "internal_unit_cost": 6.0,
        "salvage": 3.0,  # liquidation salvage ₹/unit

        # Workforce costs
        "wage": 9000,       # ₹ per worker per bi-week
        "hire_base": 12000, # ₹ per worker
        "hire_mult": [1.00, 1.25, 1.50, 1.75, 2.00, 2.00],  # linear; B5/B6=2×B1
        # Firing cost is derived: 0.5 * wage

        # Other costs
        "ot_premium_per_unit": 1.0,      # ₹ per unit * OT% (linear)
        "subcontract_unit_cost": 9.0,    # ₹ per unit
        "holding_cost": 1.5,            # ₹ per unit per bi-week
        "lost_sales_penalty": 20.0,     # ₹ per unit
    }

def ensure_module1_in_session():
    """Call once at startup."""
    if "params" not in st.session_state:
        st.session_state.params = init_module1_params()

def render_module1_preset_page():
    """
    Module 1 UI: Shows preset values cleanly.
    If you want to keep everything locked, do NOT provide edit widgets.
    """
    p = st.session_state.params

    st.title("📌 Module 1 — Preset Values (Scenario Setup)")
    st.caption("These presets define the scenario and are used by all later modules.")

    # Quick highlights
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Horizon", f"{len(p['periods'])} bi-weeks")
    c2.metric("Initial workforce (W0)", f"{p['W0']}")
    c3.metric("Opening inventory (Inv0)", f"{p['Inv0']:,}")
    c4.metric("Productivity", f"{p['p']:,}")
    c4.caption("units/worker/bi-week")

    st.divider()

    # Demand table
    st.subheader("Base Demand Forecast")
    df_demand = pd.DataFrame({"Biweek": p["periods"], "Base_Demand": p["base_demand"]})
    st.dataframe(df_demand, use_container_width=True, hide_index=True)

    st.divider()

    # Capacity & rental levers
    st.subheader("Capacity & Rental Levers")
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("#### Kiln (Bottleneck)")
        st.write({
            "Kiln design cap / bi-week": f"{p['kiln_design']:,}",
            "Effective factor (alpha)": p["alpha"],
            "Machine rental adds": f"+{p['rent_kiln_add']:,} design cap",
            "Machine rental cost": f"₹{p['rent_kiln_cost']:,} / bi-week",
            "OT rule": "OT treated as extra shift; boosts effective kiln capacity",
        })

    with right:
        st.markdown("#### Warehouse (Storage)")
        st.write({
            "Base warehouse cap": f"{p['wh_base']:,}",
            "Warehouse rental adds": f"+{p['rent_wh_add']:,} cap",
            "Warehouse rental cost": f"₹{p['rent_wh_cost']:,} / bi-week",
            "Overflow handling": "Forced liquidation (inventory capped)",
            "Liquidation loss/unit": f"₹{max(p['internal_unit_cost'] - p['salvage'], 0):.0f}",
        })

    st.divider()

    # Workforce rules + costs
    st.subheader("Workforce Rules & Costs")
    st.markdown(
        f"""
- New-hire ramp: **{int(p['new_hire_productivity_factor']*100)}% productivity in hire bi-week**, then **100%** next bi-week  
- OT maximum: **{int(p['ot_max']*100)}%**
- Wage: **₹{p['wage']:,} / worker / bi-week**
- Hiring base cost: **₹{p['hire_base']:,} / worker**
- Firing cost: **0.5 × wage = ₹{int(0.5*p['wage']):,} / worker**
"""
    )

    st.markdown("#### Hiring multipliers (seasonal, linear; B5/B6 = 2× B1)")
    df_mult = pd.DataFrame({"Biweek": p["periods"], "Hire_Multiplier": p["hire_mult"]})
    st.dataframe(df_mult, use_container_width=True, hide_index=True)

    st.divider()

    # Other costs
    st.subheader("Other Cost Parameters (₹)")
    st.write({
        "Internal variable cost (₹/unit)": p["internal_unit_cost"],
        "OT premium (₹/unit × OT%)": p["ot_premium_per_unit"],
        "Subcontract (₹/unit)": p["subcontract_unit_cost"],
        "Holding (₹/unit/bi-week)": p["holding_cost"],
        "Lost sales penalty (₹/unit)": p["lost_sales_penalty"],
        "Salvage value (₹/unit)": p["salvage"],
    })

    st.divider()

    st.subheader("Stress Testing Defaults (Module 4)")
    st.write({
        "Monte Carlo runs (default)": p["mc_runs_default"],
        "Demand shock (default)": f"±{int(p['demand_shock_default']*100)}%",
    })

# ---------- HOW TO USE ----------
# 1) Call ensure_module1_in_session() once near the top of your app (after set_page_config).
# 2) In your router, map a page to render_module1_preset_page().
#
# Example:
# ensure_module1_in_session()
#
# if st.session_state.page == "assumptions":
#     render_module1_preset_page()

# ---------- MODULE 2 (Sequential Decision + Locking Engine) ----------
# Plug this into your app.py.
# Requires: PEROIDS + params already in st.session_state["params"] from Module 1.
# This module lets the user LOCK B1 -> then B2 -> ... and shows carryover state for next bi-week.

import streamlit as st
import pandas as pd

def ensure_module2_in_session():
    """
    Initializes Module 2 session objects:
      - current_index: which bi-week is active (0..5)
      - sim_state: carryover state (workforce, inventory)
      - locked_records: list of locked bi-week outputs
      - draft_decision: current bi-week decision inputs
    """
    p = st.session_state.params
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0

    if "sim_state" not in st.session_state:
        st.session_state.sim_state = {"W_prev": int(p["W0"]), "Inv_prev": float(p["Inv0"])}

    if "locked_records" not in st.session_state:
        st.session_state.locked_records = []

    if "draft_decision" not in st.session_state:
        st.session_state.draft_decision = {
            "Target_Workforce": 0,
            "OT_pct": 0.0,
            "Subcontract_units": 0,
            "Rent_Machine": 0,
            "Rent_Warehouse": 0,
        }

def reset_module2():
    """Hard reset Module 2 (for restarting the game)."""
    p = st.session_state.params
    st.session_state.current_index = 0
    st.session_state.sim_state = {"W_prev": int(p["W0"]), "Inv_prev": float(p["Inv0"])}
    st.session_state.locked_records = []
    st.session_state.draft_decision = {
        "Target_Workforce": 0,
        "OT_pct": 0.0,
        "Subcontract_units": 0,
        "Rent_Machine": 0,
        "Rent_Warehouse": 0,
    }
    # (Optional) wipe downstream modules if present
    if "mc_results" in st.session_state:
        st.session_state.mc_results = None

def module2_step_engine(period_i: int, decision: dict, state: dict, params: dict):
    """
    One-biweek engine:
      - Derived hire/fire from target workforce
      - New hires: 50% productivity in hire bi-week
      - Internal production = min(labor cap, kiln effective cap)
      - Inventory flow + forced liquidation
    Returns: (next_state, record)
    NOTE: No Monte Carlo here. Demand used = base forecast for that bi-week.
    """

    # ---- sanitize inputs ----
    target = int(max(decision.get("Target_Workforce", 0), 0))
    ot = float(decision.get("OT_pct", 0.0))
    ot = max(0.0, min(ot, params["ot_max"]))
    sub = int(max(decision.get("Subcontract_units", 0), 0))
    rent_machine = 1 if decision.get("Rent_Machine", 0) else 0
    rent_wh = 1 if decision.get("Rent_Warehouse", 0) else 0

    W_prev = int(state["W_prev"])
    Inv_prev = float(state["Inv_prev"])

    # ---- workforce ----
    hire = max(target - W_prev, 0)
    fire = max(W_prev - target, 0)
    W_end = W_prev + hire - fire  # equals target

    # effective workforce (new hires 50% in same bi-week)
    existing = W_prev - fire
    W_eff = existing + params["new_hire_productivity_factor"] * hire  # 0.5*hire

    # ---- capacity ----
    labor_cap = W_eff * params["p"] * (1.0 + ot)

    kiln_design = params["kiln_design"] + rent_machine * params["rent_kiln_add"]
    kiln_eff = kiln_design * params["alpha"] * (1.0 + ot)  # OT as extra shift boosts kiln effective

    internal_prod = min(labor_cap, kiln_eff)

    # ---- warehouse ----
    wh_cap = params["wh_base"] + rent_wh * params["rent_wh_add"]

    # ---- demand & inventory flow (base demand) ----
    demand = float(params["base_demand"][period_i])
    available = Inv_prev + internal_prod + sub
    served = min(available, demand)
    lost = demand - served
    inv_end = available - served

    # ---- forced liquidation on overflow ----
    overflow = max(inv_end - wh_cap, 0.0)
    liq_loss_per_unit = max(params["internal_unit_cost"] - params["salvage"], 0.0)
    liquidation_cost = overflow * liq_loss_per_unit
    if overflow > 0:
        inv_end = wh_cap

    # ratios
    util = (internal_prod / kiln_design) if kiln_design > 0 else 0.0
    eff = (internal_prod / kiln_eff) if kiln_eff > 0 else 0.0

    record = {
        "Biweek": params["periods"][period_i],

        # inputs
        "Target_Workforce": target,
        "OT_pct": round(ot, 3),
        "Subcontract_units": sub,
        "Rent_Machine": rent_machine,
        "Rent_Warehouse": rent_wh,

        # start state
        "Start_Workforce": W_prev,
        "Start_Inventory": int(round(Inv_prev)),

        # derived workforce
        "Hire": hire,
        "Fire": fire,
        "Workforce_End": W_end,
        "Workforce_Effective": round(W_eff, 2),

        # capacity + output
        "LaborCap": int(round(labor_cap)),
        "KilnDesignCap": int(round(kiln_design)),
        "KilnEffectiveCap": int(round(kiln_eff)),
        "InternalProd": int(round(internal_prod)),

        # demand & inventory
        "Demand_Base": int(round(demand)),
        "Served": int(round(served)),
        "LostSales": int(round(lost)),
        "WarehouseCap": int(round(wh_cap)),
        "Inv_End": int(round(inv_end)),
        "Overflow_Liquidated": int(round(overflow)),
        "LiquidationCost": int(round(liquidation_cost)),

        # ratios
        "Utilization": round(util, 3),
        "Efficiency": round(eff, 3),
    }

    next_state = {"W_prev": W_end, "Inv_prev": float(inv_end)}
    return next_state, record

def render_module2_sequential_planner():
    """
    Module 2 main UI:
      - Decide current bi-week only
      - Preview results
      - Lock bi-week to proceed
      - Shows carryover state for next bi-week after lock
      - Maintains a history table
    """
    ensure_module2_in_session()
    p = st.session_state.params

    st.title("🧩 Module 2 — Sequential Decision Planner")
    st.caption("Decide one bi-week at a time. Lock B1 → unlock B2 → ... Carryover workforce & inventory update after each lock.")

    # Reset button
    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("🔄 Reset Module 2"):
            reset_module2()
            st.rerun()
    with c2:
        st.info("Mini computation occurs at every lock to generate next bi-week starting conditions.", icon="🧠")

    idx = st.session_state.current_index

    # History table
    if st.session_state.locked_records:
        st.subheader("Locked History (so far)")
        hist = pd.DataFrame(st.session_state.locked_records)
        cols = [
            "Biweek", "Start_Workforce", "Start_Inventory",
            "Target_Workforce", "OT_pct", "Subcontract_units", "Rent_Machine", "Rent_Warehouse",
            "InternalProd", "Served", "LostSales", "Inv_End",
            "Utilization", "Efficiency",
        ]
        st.dataframe(hist[cols], use_container_width=True, hide_index=True)

    # Finished all periods
    if idx >= len(p["periods"]):
        st.success("Module 2 complete ✅ All bi-weeks are locked.", icon="✅")
        st.caption("Next: show end-of-horizon summary, then run stress testing (Module 4) after B6 lock.")
        return

    # Current bi-week UI
    current = p["periods"][idx]
    base_dem = p["base_demand"][idx]
    state = st.session_state.sim_state

    st.divider()
    st.subheader(f"Current bi-week to decide: **{current}**")
    st.markdown(
        f"""
**Starting conditions**
- Starting workforce: **{state['W_prev']}**
- Starting inventory: **{int(round(state['Inv_prev'])):,} units**
- Base demand: **{int(base_dem):,} units**
"""
    )

    # Decision form (current bi-week only)
    draft = st.session_state.draft_decision

    with st.form(key="m2_form", clear_on_submit=False):
        a, b, c = st.columns(3, gap="large")

        with a:
            target = st.number_input("Target workforce (headcount)", min_value=0, value=int(draft["Target_Workforce"]), step=1)
            ot = st.slider("OT% (extra shift)", min_value=0.0, max_value=float(p["ot_max"]),
                           value=float(draft["OT_pct"]), step=0.01)

        with b:
            sub = st.number_input("Subcontract units", min_value=0, value=int(draft["Subcontract_units"]), step=500)
            rent_machine = st.checkbox(
                f"Rent machine (₹{p['rent_kiln_cost']:,}/bi-week, +{p['rent_kiln_add']:,} kiln cap)",
                value=bool(draft["Rent_Machine"])
            )

        with c:
            rent_wh = st.checkbox(
                f"Rent warehouse (₹{p['rent_wh_cost']:,}/bi-week, +{p['rent_wh_add']:,} storage cap)",
                value=bool(draft["Rent_Warehouse"])
            )
            st.caption("New hires are 50% productive in this bi-week.")

        preview_btn = st.form_submit_button("👀 Preview (no lock)")
        lock_btn = st.form_submit_button("🔒 Lock and go next")

    # persist draft
    st.session_state.draft_decision = {
        "Target_Workforce": int(target),
        "OT_pct": float(ot),
        "Subcontract_units": int(sub),
        "Rent_Machine": 1 if rent_machine else 0,
        "Rent_Warehouse": 1 if rent_wh else 0,
    }

    # preview or lock action
    if preview_btn or lock_btn:
        next_state, rec = module2_step_engine(idx, st.session_state.draft_decision, st.session_state.sim_state, p)

        st.markdown("### Computed outcomes (this bi-week)")
        x1, x2, x3, x4 = st.columns(4)
        x1.metric("Hire", rec["Hire"])
        x2.metric("Fire", rec["Fire"])
        x3.metric("Internal Production", f"{rec['InternalProd']:,}")
        x4.metric("Service", f"{(rec['Served']/rec['Demand_Base']):.1%}")

        y1, y2, y3, y4 = st.columns(4)
        y1.metric("Lost sales", f"{rec['LostSales']:,}")
        y2.metric("Inv end", f"{rec['Inv_End']:,}")
        y3.metric("Liquidated", f"{rec['Overflow_Liquidated']:,}")
        y4.metric("Utilization", f"{rec['Utilization']:.3f}")

        # Next bi-week starting conditions
        if idx + 1 < len(p["periods"]):
            st.info(
                f"If locked, next bi-week (**{p['periods'][idx+1]}**) starts with "
                f"Workforce **{next_state['W_prev']}** and Inventory **{int(round(next_state['Inv_prev'])):,}**.",
                icon="➡️"
            )

        # Lock
        if lock_btn:
            st.session_state.locked_records.append(rec)
            st.session_state.sim_state = next_state
            st.session_state.current_index += 1

            # auto-seed next draft target = carryover workforce
            st.session_state.draft_decision = {
                "Target_Workforce": int(next_state["W_prev"]),
                "OT_pct": 0.0,
                "Subcontract_units": 0,
                "Rent_Machine": 0,
                "Rent_Warehouse": 0,
            }
            st.rerun()

# ---------- HOW TO USE ----------
# 1) Make sure Module 1 initialized:
#    ensure_module1_in_session()
# 2) Call ensure_module2_in_session() once when entering module 2 page.
# 3) In router:
#    render_module2_sequential_planner()

# ---------- MODULE 3 (Deterministic Cost Engine) ----------
# Works on top of Module 2 locked records.
# Adds full bi-week cost breakdown + end-of-horizon summary (base demand).
#
# Assumes:
#   - Module 1 stored in st.session_state.params
#   - Module 2 produced st.session_state.locked_records (list of dicts), each dict
#     contains the inputs + ops outputs for that bi-week.
#
# You can call render_module3_summary_and_costs() on the "after B6 locked" screen,
# OR anytime you want to show running totals after each lock.

import streamlit as st
import pandas as pd
import numpy as np

def ensure_module3_in_session():
    """Initialize storage for Module 3 outputs."""
    if "m3_df" not in st.session_state:
        st.session_state.m3_df = None

def module3_compute_costs_from_locked(locked_records: list[dict], params: dict) -> pd.DataFrame:

    if not locked_records:
        return pd.DataFrame()

    rows = []

    liq_loss_per_unit = max(float(params["internal_unit_cost"]) - float(params["salvage"]), 0.0)
    wage = float(params["wage"])
    fire_base = 0.5 * wage

    for t, rec in enumerate(locked_records):

        biweek = rec.get("Biweek", f"B{t+1}")

        target = int(rec.get("Target_Workforce", 0))
        ot = float(rec.get("OT_pct", 0.0))
        sub = float(rec.get("Subcontract_units", 0))
        rent_machine = int(rec.get("Rent_Machine", 0))
        rent_wh = int(rec.get("Rent_Warehouse", 0))

        W_end = float(rec.get("Workforce_End", target))
        hire = float(rec.get("Hire", 0))
        fire = float(rec.get("Fire", 0))
        prod = float(rec.get("InternalProd", 0))
        served = float(rec.get("Served", 0))
        lost = float(rec.get("LostSales", 0))
        inv_end = float(rec.get("Inv_End", 0))
        overflow = float(rec.get("Overflow_Liquidated", 0))

        util = float(rec.get("Utilization", 0.0))
        eff = float(rec.get("Efficiency", 0.0))

        wage_cost = W_end * wage
        hire_cost = hire * float(params["hire_base"]) * float(params["hire_mult"][t])
        fire_cost = fire * fire_base

        internal_var_cost = prod * float(params["internal_unit_cost"])
        ot_premium_cost = prod * ot * float(params["ot_premium_per_unit"])

        subcontract_cost = sub * float(params["subcontract_unit_cost"])
        machine_rent_cost = rent_machine * float(params["rent_kiln_cost"])
        wh_rent_cost = rent_wh * float(params["rent_wh_cost"])

        holding_cost = inv_end * float(params["holding_cost"])
        lost_sales_cost = lost * float(params["lost_sales_penalty"])
        liquidation_cost = overflow * liq_loss_per_unit

        total_cost = (
            wage_cost + hire_cost + fire_cost +
            internal_var_cost + ot_premium_cost +
            subcontract_cost + machine_rent_cost + wh_rent_cost +
            holding_cost + lost_sales_cost + liquidation_cost
        )

        row = {
            "Biweek": biweek,
            "Hire": int(round(hire)),
            "Fire": int(round(fire)),
            "Workforce_End": int(round(W_end)),
            "InternalProd": int(round(prod)),
            "Served": int(round(served)),
            "LostSales": int(round(lost)),
            "Inv_End": int(round(inv_end)),
            "Overflow_Liquidated": int(round(overflow)),
            "Utilization": round(util, 3),
            "Efficiency": round(eff, 3),
            "WageCost": int(round(wage_cost)),
            "HireCost": int(round(hire_cost)),
            "FireCost": int(round(fire_cost)),
            "InternalVarCost": int(round(internal_var_cost)),
            "OTPremiumCost": int(round(ot_premium_cost)),
            "SubcontractCost": int(round(subcontract_cost)),
            "MachineRentCost": int(round(machine_rent_cost)),
            "WarehouseRentCost": int(round(wh_rent_cost)),
            "HoldingCost": int(round(holding_cost)),
            "LostSalesCost": int(round(lost_sales_cost)),
            "LiquidationCost": int(round(liquidation_cost)),
            "TotalCost_Biweek": int(round(total_cost)),
        }

        rows.append(row)

    return pd.DataFrame(rows)




def module3_kpis(df: pd.DataFrame) -> dict:
    """Simple summary KPIs from the Module 3 dataframe."""
    if df is None or df.empty:
        return {}

    total_cost = int(df["TotalCost_Biweek"].sum())
    total_served = int(df["Served"].sum())
    total_lost = int(df["LostSales"].sum())
    total_demand = total_served + total_lost
    service = (total_served / total_demand) if total_demand > 0 else 0.0
    ending_inv = int(df["Inv_End"].iloc[-1])

    # cost contributors (top-line)
    cost_cols = [
        "WageCost","HireCost","FireCost","InternalVarCost","OTPremiumCost",
        "SubcontractCost","MachineRentCost","WarehouseRentCost",
        "HoldingCost","LostSalesCost","LiquidationCost"
    ]
    contrib = {c: int(df[c].sum()) for c in cost_cols}

    return {
        "TotalCost": total_cost,
        "ServiceLevel": service,
        "TotalLostSales": total_lost,
        "EndingInventory": ending_inv,
        "CostContrib": contrib,
    }

def render_module3_summary_and_costs():
    """
    UI for Module 3:
      - Shows deterministic cost table + KPIs
      - Intended to show after B6 lock (full horizon)
    """
    ensure_module3_in_session()
    p = st.session_state.params
    locked = st.session_state.get("locked_records", [])

    st.subheader("📌 Module 3 — Deterministic Cost Engine (Base Demand)")

    if not locked:
        st.warning("No locked decisions yet. Complete Module 2 first.")
        return

    df = module3_compute_costs_from_locked(locked, p)
    st.session_state.m3_df = df

    # KPIs
    k = module3_kpis(df)
    if not k:
        st.warning("Could not compute KPIs.")
        return

    a, b, c, d = st.columns(4)
    a.metric("Total Cost", f"₹{k['TotalCost']:,}")
    b.metric("Service Level", f"{k['ServiceLevel']:.1%}")
    c.metric("Total Lost Sales", f"{k['TotalLostSales']:,}")
    d.metric("Ending Inventory", f"{k['EndingInventory']:,}")

    st.divider()

    st.markdown("### Cost Breakdown by Bi-week")
    show_cols = [
        "Biweek",
        "WageCost","HireCost","FireCost",
        "InternalVarCost","OTPremiumCost",
        "SubcontractCost","MachineRentCost","WarehouseRentCost",
        "HoldingCost","LostSalesCost","LiquidationCost",
        "TotalCost_Biweek",
    ]
    st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Total Cost Contributors (Cumulative)")
    contrib = k["CostContrib"]
    contrib_df = pd.DataFrame([{"CostComponent": key, "Amount": val} for key, val in contrib.items()])
    contrib_df = contrib_df.sort_values("Amount", ascending=False)
    st.dataframe(contrib_df, use_container_width=True, hide_index=True)

# ---------- HOW TO USE ----------
# After B6 lock (or anytime), call:
#   render_module3_summary_and_costs()
#
# Example: at end of Module 2:
# if st.session_state.current_index >= 6:
#     render_module3_summary_and_costs()


# ---------- MODULE 4 (Monte Carlo Stress Testing) ----------
# Evaluates robustness of the LOCKED plan under demand uncertainty.
# Requires:
#   - st.session_state.locked_records (from Module 2)
#   - st.session_state.params (from Module 1)

import numpy as np
import pandas as pd
import streamlit as st


def module4_run_monte_carlo(locked_records, params, R=200, shock=0.10, seed=42):
    """
    Monte Carlo Stress Test

    - locked_records: sequential decisions from Module 2 (B1..B6)
    - R: number of simulation runs
    - shock: demand fluctuation range (± shock)
    - seed: reproducibility

    Returns:
        summary_dict
        cost_array
        service_array
    """

    if not locked_records or len(locked_records) < len(params["periods"]):
        return None, None, None

    rng = np.random.default_rng(seed)

    total_costs = []
    service_levels = []

    for r in range(R):

        # Reset starting state each run
        W_prev = params["W0"]
        Inv_prev = float(params["Inv0"])

        run_cost = 0.0
        run_demand = 0.0
        run_served = 0.0

        for t in range(len(params["periods"])):

            rec = locked_records[t]

            # ---- Randomize demand ----
            base = params["base_demand"][t]
            multiplier = rng.uniform(1 - shock, 1 + shock)
            demand = base * multiplier

            # ---- Extract locked decision ----
            target = rec["Target_Workforce"]
            ot = rec["OT_pct"]
            sub = rec["Subcontract_units"]
            rent_machine = rec["Rent_Machine"]
            rent_wh = rec["Rent_Warehouse"]

            # ---- Workforce ----
            hire = max(target - W_prev, 0)
            fire = max(W_prev - target, 0)
            W_end = W_prev + hire - fire

            existing = W_prev - fire
            W_eff = existing + params["new_hire_productivity_factor"] * hire

            # ---- Capacity ----
            labor_cap = W_eff * params["p"] * (1 + ot)
            kiln_design = params["kiln_design"] + rent_machine * params["rent_kiln_add"]
            kiln_eff = kiln_design * params["alpha"] * (1 + ot)

            internal_prod = min(labor_cap, kiln_eff)

            # ---- Warehouse ----
            wh_cap = params["wh_base"] + rent_wh * params["rent_wh_add"]

            # ---- Inventory flow ----
            available = Inv_prev + internal_prod + sub
            served = min(available, demand)
            lost = demand - served
            inv_end = available - served

            # ---- Forced liquidation ----
            overflow = max(inv_end - wh_cap, 0)
            liq_loss_unit = max(params["internal_unit_cost"] - params["salvage"], 0)
            liquidation_cost = overflow * liq_loss_unit
            if overflow > 0:
                inv_end = wh_cap

            # ---- Costs ----
            wage_cost = W_end * params["wage"]
            hire_cost = hire * params["hire_base"] * params["hire_mult"][t]
            fire_cost = fire * (0.5 * params["wage"])
            internal_cost = internal_prod * params["internal_unit_cost"]
            ot_cost = internal_prod * ot * params["ot_premium_per_unit"]
            subcontract_cost = sub * params["subcontract_unit_cost"]
            machine_rent_cost = rent_machine * params["rent_kiln_cost"]
            wh_rent_cost = rent_wh * params["rent_wh_cost"]
            holding_cost = inv_end * params["holding_cost"]
            lost_cost = lost * params["lost_sales_penalty"]

            total_cost = (
                wage_cost + hire_cost + fire_cost +
                internal_cost + ot_cost +
                subcontract_cost + machine_rent_cost + wh_rent_cost +
                holding_cost + lost_cost + liquidation_cost
            )

            # ---- Accumulate ----
            run_cost += total_cost
            run_demand += demand
            run_served += served

            # Carryover
            W_prev = W_end
            Inv_prev = inv_end

        total_costs.append(run_cost)
        service_levels.append(run_served / run_demand if run_demand > 0 else 0)

    total_costs = np.array(total_costs)
    service_levels = np.array(service_levels)

    summary = {
        "Runs": R,
        "AvgCost": float(total_costs.mean()),
        "P90Cost": float(np.quantile(total_costs, 0.90)),
        "StdCost": float(total_costs.std()),
        "AvgService": float(service_levels.mean()),
        "P10Service": float(np.quantile(service_levels, 0.10)),
        "StdService": float(service_levels.std())
    }

    return summary, total_costs, service_levels

def render_module4_ui():

    st.subheader("🧪 Module 4 — Stress Testing (Monte Carlo)")

    if "locked_records" not in st.session_state or len(st.session_state.locked_records) < 6:
        st.warning("Complete all 6 bi-weeks before running stress test.")
        return

    col1, col2 = st.columns(2)

    with col1:
        R = st.number_input("Monte Carlo Runs", 50, 2000, 200, 50)

    with col2:
        shock = st.slider("Demand Shock (±%)", 0.0, 0.30, 0.10, 0.01)

    if st.button("Run Stress Test", use_container_width=True):

        summary, costs, services = module4_run_monte_carlo(
            st.session_state.locked_records,
            st.session_state.params,
            R=int(R),
            shock=float(shock)
        )

        if summary is None:
            st.error("Simulation failed.")
            return

        st.session_state.mc_results = {
            "summary": summary,
            "costs": costs,
            "services": services
        }

        st.success("Stress Test Complete ✅")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Avg Cost", f"₹{int(summary['AvgCost']):,}")
        k2.metric("P90 Cost", f"₹{int(summary['P90Cost']):,}")
        k3.metric("Avg Service", f"{summary['AvgService']:.1%}")
        k4.metric("P10 Service", f"{summary['P10Service']:.1%}")

        st.divider()

        st.subheader("Distribution of Total Cost")
        st.bar_chart(pd.Series(costs))

        st.subheader("Distribution of Service Level")
        st.bar_chart(pd.Series(services))

# ---------- MODULE 5 (Consultant Score + Verdict) ----------
# Uses:
#   - Module 3 deterministic df (st.session_state.m3_df) OR recompute from locked_records
#   - Module 4 summary dict: {AvgCost, P90Cost, AvgService, P10Service, ...}
#
# Outputs:
#   - Total score (0-100)
#   - Subscores: Cost, Service, Workforce, Capacity
#   - Verdict label
#
# NOTE: This module assumes you already ran Module 4 and have mc_summary available.

import streamlit as st
import pandas as pd
import numpy as np

def clamp(x: float, low: float, high: float) -> float:
    return max(low, min(x, high))

def module5_compute_consultant_score(m3_df: pd.DataFrame, mc_summary: dict,
                                    weights=None,
                                    service_good=0.95, service_bad=0.85,
                                    churn_good=0.25, churn_bad=0.80):
    """
    Computes consultant score based on:
      - Risk-adjusted cost: P90Cost (Module 4) vs base deterministic cost (Module 3)
      - Tail service reliability: P10Service (Module 4)
      - Workforce stability: churn rate using hires+fires derived from Module 2/3
      - Capacity balance: stress count based on utilization/efficiency in deterministic plan

    Returns:
      dict with total score, subscores, verdict, and intermediate metrics.
    """

    if weights is None:
        weights = {"cost": 0.30, "service": 0.35, "workforce": 0.20, "capacity": 0.15}

    if m3_df is None or m3_df.empty or mc_summary is None:
        return None

    # ---- Inputs from Module 3 ----
    base_cost = float(m3_df["TotalCost_Biweek"].sum())

    # If utilization/efficiency not present in m3_df, create placeholders
    util = m3_df["Utilization"].values if "Utilization" in m3_df.columns else np.zeros(len(m3_df))
    eff = m3_df["Efficiency"].values if "Efficiency" in m3_df.columns else np.zeros(len(m3_df))

    # Workforce exposure + churn
    # We prefer Hire/Fire columns if present; otherwise approximate churn via Target_Workforce diffs
    if "Hire" in m3_df.columns and "Fire" in m3_df.columns:
        churn = float((m3_df["Hire"] + m3_df["Fire"]).sum())
    else:
        # Approximate using workforce changes (sum abs diffs)
        if "Workforce_End" in m3_df.columns:
            W = m3_df["Workforce_End"].values.astype(float)
            W_prev = np.concatenate([[0.0], W[:-1]])
            churn = float(np.abs(W - W_prev).sum())
        else:
            churn = 0.0

    if "Workforce_End" in m3_df.columns:
        exposure = float(m3_df["Workforce_End"].sum())
    else:
        exposure = max(churn, 1.0)

    churn_rate = churn / (exposure + 1e-9)

    # Stress weeks (capacity balance)
    stress = int(np.sum((util > 0.95) | (eff > 1.00)))
    stress_frac = stress / max(len(m3_df), 1)

    # ---- Inputs from Module 4 ----
    p90_cost = float(mc_summary["P90Cost"])
    p10_service = float(mc_summary["P10Service"])

    # -----------------------
    # Subscores (0-100)
    # -----------------------

    # COST score uses P90Cost vs a dynamic band around deterministic base cost
    C_good = 0.95 * base_cost
    C_bad = 1.25 * base_cost
    S_cost = 100.0 * clamp((C_bad - p90_cost) / (C_bad - C_good + 1e-9), 0.0, 1.0)

    # SERVICE score uses P10 service vs thresholds
    S_service = 100.0 * clamp((p10_service - service_bad) / (service_good - service_bad + 1e-9), 0.0, 1.0)

    # WORKFORCE score based on churn rate
    S_workforce = 100.0 * clamp((churn_bad - churn_rate) / (churn_bad - churn_good + 1e-9), 0.0, 1.0)

    # CAPACITY score based on stress fraction
    S_capacity = 100.0 * (1.0 - clamp(stress_frac, 0.0, 1.0))

    # -----------------------
    # Weighted total
    # -----------------------
    total = (
        weights["cost"] * S_cost +
        weights["service"] * S_service +
        weights["workforce"] * S_workforce +
        weights["capacity"] * S_capacity
    )

    # Verdict
    if total >= 85:
        verdict = "Board-Ready ✅"
    elif total >= 70:
        verdict = "Strong but Risky ⚠️"
    elif total >= 50:
        verdict = "Needs Rework 🔁"
    else:
        verdict = "Rejected ❌"

    return {
        "TotalScore": float(round(total, 2)),
        "Verdict": verdict,
        "Subscores": {
            "Cost": float(round(S_cost, 2)),
            "Service": float(round(S_service, 2)),
            "Workforce": float(round(S_workforce, 2)),
            "Capacity": float(round(S_capacity, 2)),
        },
        "Metrics": {
            "BaseCost": base_cost,
            "P90Cost": p90_cost,
            "P10Service": p10_service,
            "Churn": churn,
            "Exposure": exposure,
            "ChurnRate": churn_rate,
            "StressWeeks": stress,
            "StressFraction": stress_frac,
        },
        "Weights": weights,
        "Thresholds": {
            "ServiceGood": service_good,
            "ServiceBad": service_bad,
            "ChurnGood": churn_good,
            "ChurnBad": churn_bad,
            "CostGood": C_good,
            "CostBad": C_bad,
        }
    }

def render_module5_scorecard(m3_df: pd.DataFrame, mc_summary: dict):
    """
    UI renderer for Module 5.
    Call this AFTER Module 4 stress test is completed.
    """
    st.subheader("🏛️ Module 5 — Consultant Scorecard")

    if mc_summary is None:
        st.warning("Run Module 4 (Stress Test) first to generate P90 cost and P10 service.")
        return
    if m3_df is None or m3_df.empty:
        st.warning("Module 3 deterministic cost table not found.")
        return

    # Optional weight controls
    with st.expander("Adjust weights (optional)", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        w_cost = c1.slider("Weight: Cost", 0.0, 1.0, 0.30, 0.05)
        w_service = c2.slider("Weight: Service", 0.0, 1.0, 0.35, 0.05)
        w_work = c3.slider("Weight: Workforce", 0.0, 1.0, 0.20, 0.05)
        w_cap = c4.slider("Weight: Capacity", 0.0, 1.0, 0.15, 0.05)

        total_w = w_cost + w_service + w_work + w_cap
        if total_w == 0:
            st.error("Weights cannot all be zero.")
            return
        # normalize
        weights = {
            "cost": w_cost / total_w,
            "service": w_service / total_w,
            "workforce": w_work / total_w,
            "capacity": w_cap / total_w
        }
        st.caption(f"Normalized weights: {weights}")
    # If expander not opened, use defaults
    if "weights" not in locals():
        weights = {"cost": 0.30, "service": 0.35, "workforce": 0.20, "capacity": 0.15}

    result = module5_compute_consultant_score(m3_df, mc_summary, weights=weights)
    if result is None:
        st.error("Could not compute score.")
        return

    # Top line
    st.success(f"**Consultant Score: {result['TotalScore']:.2f} / 100**  —  {result['Verdict']}")

    # Subscores
    s = result["Subscores"]
    a, b, c, d = st.columns(4)
    a.metric("Cost (P90-based)", f"{s['Cost']:.1f}")
    b.metric("Service (P10-based)", f"{s['Service']:.1f}")
    c.metric("Workforce Stability", f"{s['Workforce']:.1f}")
    d.metric("Capacity Balance", f"{s['Capacity']:.1f}")

    st.divider()
    st.markdown("### Key Metrics Used")
    m = result["Metrics"]
    st.write({
        "Base deterministic cost (Module 3)": f"₹{int(m['BaseCost']):,}",
        "P90 cost (Module 4)": f"₹{int(m['P90Cost']):,}",
        "P10 service (Module 4)": f"{m['P10Service']:.1%}",
        "Churn (Σ hire + fire)": int(m["Churn"]),
        "Exposure (Σ workforce end)": int(m["Exposure"]),
        "Churn rate": f"{m['ChurnRate']:.3f}",
        "Stress weeks": f"{m['StressWeeks']}/6",
    })

    st.caption("Clamp is used internally to keep each subscore within 0–100.")

# ---------- HOW TO USE ----------
# After Module 4:
#   mc_summary = st.session_state.mc_results["summary"]
# After Module 3:
#   m3_df = st.session_state.m3_df
# Then:
#   render_module5_scorecard(m3_df, mc_summary)
# ==============================
# MAIN APP ROUTER
# ==============================

st.set_page_config(page_title="Diya Ops Simulator", layout="wide")

ensure_module1_in_session()

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to:",
    [
        "Home",
        "How to Play",
        "Module 1 – Presets",
        "Module 2 – Sequential Planner",
        "Module 3 – Deterministic Summary",
        "Module 4 – Stress Test",
        "Module 5 – Consultant Score"
    ]
)

if page == "Home":

    st.title("🪔 Diya Manufacturing Operations Simulator")

    st.markdown("""
    ## 📖 The Story

    You are the Operations Head of Diya Manufacturing.  
    Demand fluctuates. Attrition is rising. Costs are increasing.  

    Your challenge:  
    Balance hiring, productivity, and churn while maintaining profitability.

    Every decision impacts workforce stability and financial performance.

    ---

    ## 🎯 Objective

    Manage operational decisions across multiple modules to:

    - Stabilize workforce levels  
    - Control hiring and salary costs  
    - Reduce volatility  
    - Maintain sustainable operating margins  

    ---

    ## 🕹 How To Play

    1. Start with Module 1 – Presets to load baseline assumptions  
    2. Use Module 2 – Sequential Planner to adjust hiring decisions  
    3. Analyze operational output in Module 3 – Deterministic  
    4. Test uncertainty in Module 4 – Stress Test  
    5. Review overall performance in Module 5 – Consultant Score

    ---

    ## 🏆 Winning Strategy

    Success requires:

    - Smart hiring timing  
    - Balanced cost structure  
    - Controlled attrition impact  
    - Sustainable operational growth  

    Think like a consultant. Decide like a COO.
    """)
elif page == "How to Play":

    st.title("📘 How To Play")

    st.markdown("""
    ## Overview

    You are the Operations Head of Diya Manufacturing.
    Your responsibility is to balance workforce growth,
    cost stability, and attrition risk.

    The simulator models long-term operational consequences
    of short-term hiring decisions.

    ---

    ## Step-by-Step Guide

    ### 1️⃣ Module 1 – Presets
    Set baseline parameters:
    - Initial workforce
    - Attrition rate
    - Salary assumptions
    - Demand level

    These assumptions drive the entire model.

    ### 2️⃣ Module 2 – Sequential Planner
    Decide how many employees to hire each period.

    Over-hiring → Cost explosion  
    Under-hiring → Capacity shortage  

    Smart timing is critical.

    ### 3️⃣ Module 3 – Deterministic Summary
    View:
    - Workforce evolution
    - Total cost trajectory
    - Attrition impact

    This shows the direct impact of your decisions.

    ### 4️⃣ Module 4 – Stress Test
    Simulate:
    - Demand shocks
    - Attrition spikes
    - Cost pressure

    Test resilience of your strategy.

    ### 5️⃣ Module 5 – Consultant Score
    Your strategy is evaluated on:
    - Stability
    - Cost efficiency
    - Risk control
    - Margin sustainability

    ---

    ## Winning Strategy

    The best strategies:
    - Avoid extreme hiring spikes
    - Absorb attrition smoothly
    - Maintain predictable cost growth
    - Balance scaling with stability

    Think long-term.
    Avoid reactive decisions.
    """)

elif page == "Module 1 – Presets":
    render_module1_preset_page()

elif page == "Module 2 – Sequential Planner":
    render_module2_sequential_planner()

elif page == "Module 3 – Deterministic Summary":
    if "locked_records" in st.session_state and len(st.session_state.locked_records) > 0:
        df = module3_compute_costs_from_locked(
            st.session_state.locked_records,
            st.session_state.params
        )
        st.session_state.m3_df = df
        render_module3_summary_and_costs()
    else:
        st.warning("Complete Module 2 first.")

elif page == "Module 4 – Stress Test":
    render_module4_ui()

elif page == "Module 5 – Consultant Score":
    if "m3_df" in st.session_state and "mc_results" in st.session_state:
        render_module5_scorecard(
            st.session_state.m3_df,
            st.session_state.mc_results["summary"]
        )
    else:
        st.warning("Run Module 3 and Module 4 first.")

