import gradio as gr
import requests
import json
import pandas as pd
import time
from datetime import datetime

# URL of the backend FastAPI server (same container, internal)
API_URL = "http://127.0.0.1:8000"

def init_state():
    return {
        "case_id": "C1",
        "role": "defense",
        "phase": "setup",
        "step": 0,
        "max_steps": 30,
        "history": [],
        "verdict": None,
        "jury": {"analytical": 0.5, "empathetic": 0.5, "skeptical": 0.5},
        "available_actions": ["pass"],
        "my_evidence": [],
        "opponent_last": None,
        "error": None
    }

def format_evidence_html(evidence_list):
    if not evidence_list:
        return "<div style='color: #666; font-style: italic;'>No evidence available yet.</div>"
    
    html = "<div style='display: flex; flex-direction: column; gap: 8px;'>"
    for ev in evidence_list:
        status_color = "#28a745" if ev.get("presented") else "#6c757d"
        status_text = "PRESENTED" if ev.get("presented") else "HIDDEN"
        
        strength_val = ev.get("strength", 0)
        strength_color = "#dc3545" if strength_val > 0.8 else "#ffc107" if strength_val > 0.5 else "#17a2b8"
        
        html += f"""
        <div style='border: 1px solid #ddd; border-radius: 6px; padding: 10px; background-color: #f8f9fa;'>
            <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                <strong style='color: #003366;'>{ev.get('id')}</strong>
                <span style='background-color: {status_color}; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.8em; font-weight: bold;'>{status_text}</span>
            </div>
            <div style='font-size: 0.9em; margin-bottom: 8px;'>{ev.get('description', '')}</div>
            <div style='display: flex; gap: 10px; font-size: 0.8em; color: #555;'>
                <span><span style='color: {strength_color};'>■</span> Strength: {strength_val:.2f}</span>
                <span><span style='color: #e83e8c;'>■</span> Emotion: {ev.get('emotional_impact', 0):.2f}</span>
                {"" if ev.get('admissible', True) else "<span style='color: red; font-weight: bold;'>INADMISSIBLE</span>"}
            </div>
        </div>
        """
    html += "</div>"
    return html

def format_history_html(history):
    if not history:
        return "<div style='color: #666; font-style: italic;'>Trial has not started.</div>"
    
    html = "<div style='display: flex; flex-direction: column; gap: 10px; height: 400px; overflow-y: auto; padding-right: 10px;'>"
    
    # Reverse to show newest at top, or keep chronological
    for entry in history:
        role = entry.get("role", "unknown")
        action = entry.get("action_type", "pass")
        
        # Styling based on role
        if role == "prosecutor":
            border_color = "#dc3545" # Red
            bg_color = "#fdf3f4"
            icon = "🔴"
        elif role == "defense":
            border_color = "#007bff" # Blue
            bg_color = "#f0f7ff"
            icon = "🔵"
        elif role == "judge":
            border_color = "#343a40" # Dark gray
            bg_color = "#f8f9fa"
            icon = "⚖️"
        else:
            border_color = "#6c757d"
            bg_color = "#f8f9fa"
            icon = "⚪"
            
        # Format action details
        details = ""
        if action == "opening_statement" or action == "closing_argument":
            details = f"<i>\"{entry.get('argument_text', '')}\"</i><br><small>Framing: {entry.get('framing', 'factual')}</small>"
        elif action == "present_evidence":
            details = f"Presented Evidence: <b>{entry.get('evidence_id', '')}</b><br><small>Framing: {entry.get('framing', 'factual')}</small>"
        elif action == "object":
            details = f"Objected to <b>{entry.get('target', '')}</b> ({entry.get('objection_type', '')})"
        elif action in ["sustain", "overrule"]:
            details = f"Objection <b>{entry.get('objection_ruling', '').upper()}</b>"
            
        html += f"""
        <div style='border-left: 4px solid {border_color}; padding: 10px; background-color: {bg_color}; border-radius: 0 6px 6px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.05);'>
            <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                <strong>{icon} {role.title()}</strong>
                <span style='color: #888; font-size: 0.8em;'>Step {entry.get('step', '?')}</span>
            </div>
            <div style='margin-bottom: 5px;'><span style='background-color: #e9ecef; padding: 2px 6px; border-radius: 4px; font-size: 0.8em;'>{action.replace('_', ' ').title()}</span></div>
            <div style='font-size: 0.9em; line-height: 1.4;'>{details}</div>
        </div>
        """
    html += "</div>"
    return html

def get_jury_plot(jury_data):
    # This is a simple HTML/CSS bar chart since matplotlib in Gradio can be slow/fiddly
    if not jury_data:
        jury_data = {"analytical": 0.5, "empathetic": 0.5, "skeptical": 0.5}
        
    html = "<div style='display: flex; justify-content: space-around; align-items: flex-end; height: 150px; padding: 10px; background: #1a1a1a; border-radius: 8px;'>"
    
    colors = {
        "analytical": "#3498DB", # Blue
        "empathetic": "#E91E63", # Pink
        "skeptical": "#FF9800"   # Orange
    }
    
    for j_type, val in jury_data.items():
        # Scale value (0-1) to percentage, inverted so 0 = bottom, 1 = top
        # Actually, in Adversa, >0.5 is guilty, <0.5 is not guilty. Let's make <0.5 (defense win) GREEN and >0.5 RED
        # Height is just the value
        height_pct = val * 100
        
        color = colors.get(j_type, "#ccc")
        
        # Determine current leaning
        leaning = "GUILTY" if val > 0.5 else "NOT GUILTY" if val < 0.5 else "UNDECIDED"
        leaning_color = "#ff4d4d" if val > 0.5 else "#4CAF50" if val < 0.5 else "#aaa"
        
        html += f"""
        <div style='display: flex; flex-direction: column; align-items: center; width: 30%;'>
            <div style='color: white; font-size: 0.8em; margin-bottom: 5px; font-weight: bold;'>{val:.2f}</div>
            <div style='width: 100%; height: 100px; background: #333; position: relative; border-radius: 4px 4px 0 0; overflow: hidden;'>
                <div style='position: absolute; bottom: 0; width: 100%; height: {height_pct}%; background: {color}; transition: height 0.5s ease;'></div>
                <div style='position: absolute; bottom: 50%; width: 100%; height: 2px; background: rgba(255,255,255,0.5); z-index: 10;'></div>
            </div>
            <div style='color: white; font-size: 0.9em; margin-top: 8px; text-transform: capitalize;'>{j_type}</div>
            <div style='color: {leaning_color}; font-size: 0.7em; font-weight: bold; margin-top: 2px;'>{leaning}</div>
        </div>
        """
        
    html += "</div>"
    
    # Add legend
    html += """
    <div style='display: flex; justify-content: center; gap: 15px; margin-top: 10px; font-size: 0.8em;'>
        <span><span style='color: #ff4d4d;'>■</span> Guilty (>0.5)</span>
        <span><span style='color: #4CAF50;'>■</span> Not Guilty (<0.5)</span>
        <span><span style='color: white;'>—</span> Neutral (0.5)</span>
    </div>
    """
    return html

def api_reset(case_id, role, seed):
    try:
        url = f"{API_URL}/reset"
        payload = {
            "seed": int(seed),
            "options": {"case_id": case_id, "role": role}
        }
        r = requests.post(url, json=payload, timeout=10)
        
        if r.status_code == 200:
            data = r.json()
            obs = data.get("observation", {})
            
            # Fetch case details for UI
            case_info = requests.get(f"{API_URL}/cases/{case_id}").json()
            
            state = init_state()
            state["case_id"] = case_id
            state["role"] = role
            state["phase"] = obs.get("phase", "opening")
            state["step"] = obs.get("step", 0)
            state["max_steps"] = obs.get("max_steps", 30)
            state["my_evidence"] = obs.get("my_evidence", [])
            state["jury"] = obs.get("jury_sentiment", {"analytical": 0.5, "empathetic": 0.5, "skeptical": 0.5})
            state["available_actions"] = obs.get("available_actions", ["pass"])
            state["history"] = obs.get("public_record", [])
            state["case_name"] = case_info.get("name", case_id)
            state["charges"] = case_info.get("charges", "")
            state["gt"] = case_info.get("ground_truth", "")
            
            return state, update_ui_from_state(state)
        else:
            state = init_state()
            state["error"] = f"API Error: {r.status_code} - {r.text}"
            return state, update_ui_error(state["error"])
            
    except Exception as e:
        state = init_state()
        state["error"] = f"Connection Error: {str(e)}"
        return state, update_ui_error(state["error"])

def api_step(state, action_type, evidence_id, argument_text, framing, objection_type, objection_target):
    if state.get("verdict"):
        return state, update_ui_from_state(state) # Already done
        
    try:
        # Build action dict
        action = {
            "role": state["role"],
            "action_type": action_type
        }
        
        if action_type in ["opening_statement", "closing_argument"]:
            action["argument_text"] = argument_text or "..."
            action["framing"] = framing
        elif action_type == "present_evidence":
            action["evidence_id"] = evidence_id
            action["framing"] = framing
        elif action_type == "object":
            action["objection_type"] = objection_type
            action["target"] = objection_target
            
        url = f"{API_URL}/step"
        r = requests.post(url, json={"action": action}, timeout=10)
        
        if r.status_code == 200:
            data = r.json()
            obs = data.get("observation", {})
            done = data.get("done", False)
            
            state["phase"] = obs.get("phase", state["phase"])
            state["step"] = obs.get("step", state["step"])
            state["my_evidence"] = obs.get("my_evidence", state["my_evidence"])
            state["jury"] = obs.get("jury_sentiment", state["jury"])
            state["available_actions"] = obs.get("available_actions", ["pass"])
            state["history"] = obs.get("public_record", state["history"])
            state["opponent_last"] = obs.get("last_opponent_action")
            
            if done:
                # Fetch final state to get verdict
                state_r = requests.get(f"{API_URL}/state").json()
                state["verdict"] = state_r.get("verdict")
                state["verdict_correct"] = state_r.get("verdict_correct")
                state["phase"] = "verdict"
                
            return state, update_ui_from_state(state)
        else:
            state["error"] = f"API Error: {r.status_code} - {r.text}"
            return state, update_ui_from_state(state)
            
    except Exception as e:
        state["error"] = f"Connection Error: {str(e)}"
        return state, update_ui_from_state(state)

def bot_step(state):
    """Simulate the opponent and judge taking their turns automatically."""
    if state.get("verdict") or state.get("phase") == "setup":
        return state, update_ui_from_state(state)
        
    # We'll just poll the API using heuristic fallback from inference.py for the opponent
    # To keep this UI self-contained without needing inference.py, we'll build a simple logic here
    
    current_phase = state["phase"]
    my_role = state["role"]
    
    # Check if we should wait for user
    turn_order = {
        "opening": ["prosecutor", "defense"],
        "prosecution_case": ["prosecutor", "defense"] * 5,
        "defense_case": ["defense", "prosecutor"] * 5,
        "closing": ["prosecutor", "defense", "prosecutor", "defense"],
        "verdict": ["judge", "judge"]
    }
    
    # We don't have perfect knowledge of whose turn it is without /state, 
    # but we can try actions for other roles until we hit "not_your_turn"
    
    roles_to_try = ["judge", "prosecutor", "defense"]
    roles_to_try.remove(my_role)
    
    made_move = False
    max_auto_steps = 5 # Don't loop forever
    
    for _ in range(max_auto_steps):
        moved_this_loop = False
        
        for role in roles_to_try:
            # Let's try passing for them just to advance state
            action = {"role": role, "action_type": "pass"}
            
            # If judge and pending objection, judge must rule
            # (We don't track pending objection in UI state perfectly, so just try)
            
            try:
                r = requests.post(f"{API_URL}/step", json={"action": action}, timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    # If reward is -0.5 and error is not_your_turn, this wasn't their turn
                    info = data.get("info", {})
                    if info.get("error") == "not_your_turn":
                        continue
                        
                    # It WAS their turn!
                    moved_this_loop = True
                    made_move = True
                    
                    obs = data.get("observation", {})
                    state["phase"] = obs.get("phase", state["phase"])
                    state["step"] = obs.get("step", state["step"])
                    state["my_evidence"] = obs.get("my_evidence", state["my_evidence"])
                    state["jury"] = obs.get("jury_sentiment", state["jury"])
                    state["available_actions"] = obs.get("available_actions", ["pass"])
                    state["history"] = obs.get("public_record", state["history"])
                    
                    if data.get("done", False):
                        state_r = requests.get(f"{API_URL}/state").json()
                        state["verdict"] = state_r.get("verdict")
                        state["verdict_correct"] = state_r.get("verdict_correct")
                        state["phase"] = "verdict"
                        return state, update_ui_from_state(state)
                        
            except Exception:
                pass
                
        if not moved_this_loop:
            break # No one could move, must be user's turn
            
    return state, update_ui_from_state(state)


def update_ui_error(err_msg):
    # Returns updates for all UI components in an error state
    return (
        gr.update(value=f"<div style='color:red; font-weight:bold;'>{err_msg}</div>"), # header
        gr.update(), # evidence
        gr.update(), # history
        gr.update(), # jury
        gr.update(value="pass"), # action_type
        gr.update(choices=[]), # evidence_id
        gr.update(), # arg_text
        gr.update(), # framing
        gr.update(), # obj_type
        gr.update(), # obj_target
        gr.update(interactive=False), # submit_btn
        gr.update() # verdict_msg
    )

def update_ui_from_state(state):
    # Format header
    case_name = state.get("case_name", "Unknown Case")
    charges = state.get("charges", "")
    phase = state.get("phase", "setup").replace("_", " ").title()
    step = state.get("step", 0)
    max_steps = state.get("max_steps", 30)
    
    header_html = f"""
    <div style='background: #001f3f; color: white; padding: 20px; border-radius: 8px; margin-bottom: 15px;'>
        <h2 style='margin: 0 0 10px 0; color: #FFDC00;'>Adversa Legal Simulator</h2>
        <div style='display: flex; justify-content: space-between; font-size: 1.1em;'>
            <div><strong>Case:</strong> {case_name} ({state.get('case_id')})</div>
            <div><strong>Role:</strong> {state.get('role').title()}</div>
        </div>
        <div style='font-size: 0.9em; margin-top: 5px; color: #aaa;'>{charges}</div>
        <div style='display: flex; justify-content: space-between; margin-top: 15px; background: rgba(255,255,255,0.1); padding: 10px; border-radius: 4px;'>
            <div><strong>Phase:</strong> <span style='color: #7FDBFF;'>{phase}</span></div>
            <div><strong>Step:</strong> {step}/{max_steps}</div>
        </div>
    </div>
    """
    
    if state.get("error"):
        header_html += f"<div style='background: #ffcccc; color: #cc0000; padding: 10px; border-radius: 4px; margin-bottom: 15px; border-left: 4px solid #cc0000;'><strong>Error:</strong> {state['error']}</div>"
        state["error"] = None # Clear after showing
    
    # Format evidence
    evidence_html = format_evidence_html(state.get("my_evidence", []))
    
    # Format history
    history_html = format_history_html(state.get("history", []))
    
    # Format jury
    jury_html = get_jury_plot(state.get("jury", {}))
    
    # UI Controls setup based on available actions
    avail = state.get("available_actions", ["pass"])
    
    # Setup evidence dropdown
    unpresented_ev = [e["id"] for e in state.get("my_evidence", []) if not e.get("presented")]
    
    # Setup verdict message
    verdict_html = ""
    if state.get("verdict"):
        v = state["verdict"].upper()
        v_color = "#dc3545" if v == "GUILTY" else "#28a745"
        gt = state.get("gt", "").upper()
        correct = "✅ Correct" if state.get("verdict_correct") else "❌ Incorrect"
        
        verdict_html = f"""
        <div style='background: {v_color}11; border: 2px solid {v_color}; padding: 20px; border-radius: 8px; text-align: center; margin-top: 20px;'>
            <h2 style='color: {v_color}; margin: 0 0 10px 0;'>VERDICT: {v}</h2>
            <div style='font-size: 1.2em; color: #555;'>Ground Truth: {gt} ({correct})</div>
        </div>
        """
        
    return (
        gr.update(value=header_html),
        gr.update(value=evidence_html),
        gr.update(value=history_html),
        gr.update(value=jury_html),
        gr.update(choices=avail, value=avail[0] if avail else None),
        gr.update(choices=unpresented_ev, value=unpresented_ev[0] if unpresented_ev else None, visible="present_evidence" in avail),
        gr.update(visible=any(a in avail for a in ["opening_statement", "closing_argument"])),
        gr.update(visible=True), # framing always visible
        gr.update(visible="object" in avail),
        gr.update(visible="object" in avail),
        gr.update(interactive=not bool(state.get("verdict"))),
        gr.update(value=verdict_html)
    )

def toggle_action_inputs(action_type):
    # Dynamically show/hide inputs based on action type
    show_ev = action_type == "present_evidence"
    show_arg = action_type in ["opening_statement", "closing_argument"]
    show_obj = action_type == "object"
    show_framing = action_type in ["present_evidence", "opening_statement", "closing_argument"]
    
    return (
        gr.update(visible=show_ev),
        gr.update(visible=show_arg),
        gr.update(visible=show_framing),
        gr.update(visible=show_obj),
        gr.update(visible=show_obj)
    )

# ─── GRADIO UI DEFINITION ───────────────────────────────────────────────────

css = """
.container { max-width: 1200px; margin: auto; }
.panel { background: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
"""

with gr.Blocks(title="Adversa Legal Simulator", css=css, theme=gr.themes.Default(primary_hue="blue", secondary_hue="gray")) as app:
    
    state = gr.State(init_state())
    
    with gr.Row():
        gr.HTML("<h1 style='text-align: center; width: 100%; margin-bottom: 5px; color: #001f3f;'>⚖️ Adversa Interactive Demo</h1>")
        
    with gr.Row():
        gr.HTML("<p style='text-align: center; width: 100%; margin-top: 0; color: #666;'>Experience the multi-agent courtroom environment. Play as the Defense Attorney and try to win the jury.</p>")
    
    with gr.Row():
        with gr.Column(scale=3):
            # Header info
            header = gr.HTML(value="<div style='text-align:center; padding:20px;'>Configure trial settings below to start.</div>")
            
            # Setup controls
            with gr.Accordion("Trial Setup", open=True) as setup_acc:
                with gr.Row():
                    case_dd = gr.Dropdown(choices=["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"], value="C1", label="Select Case")
                    role_dd = gr.Dropdown(choices=["defense", "prosecutor"], value="defense", label="Your Role")
                    seed_num = gr.Number(value=42, label="Random Seed (determines evidence strength)")
                start_btn = gr.Button("🚀 Start Trial", variant="primary")
            
            # Action controls
            with gr.Group(elem_classes=["panel"]):
                gr.Markdown("### Your Turn")
                action_type = gr.Dropdown(choices=["pass"], value="pass", label="Action Type")
                
                with gr.Row():
                    evidence_id = gr.Dropdown(choices=[], label="Evidence to Present", visible=False)
                    arg_text = gr.Textbox(label="Argument Text", placeholder="Enter your argument here...", lines=2, visible=False)
                
                with gr.Row():
                    framing = gr.Radio(choices=["factual", "emotional", "authority"], value="factual", label="Framing Strategy")
                    obj_type = gr.Dropdown(choices=["hearsay", "relevance", "coerced", "leading"], label="Objection Type", visible=False)
                    obj_target = gr.Textbox(label="Target Evidence ID (e.g., E3)", visible=False)
                
                with gr.Row():
                    submit_btn = gr.Button("Submit Action", variant="primary", interactive=False)
                    auto_opp_btn = gr.Button("🤖 Auto-Play Opponent Turn", interactive=False)
            
            verdict_msg = gr.HTML()
            
            # Jury status
            with gr.Group(elem_classes=["panel"]):
                gr.Markdown("### 🧠 Jury Psychology Status")
                jury_plot = gr.HTML()
                
        with gr.Column(scale=2):
            # Right column: Logs and Evidence
            with gr.Tabs():
                with gr.TabItem("📜 Trial Record"):
                    history_html = gr.HTML("<div style='color: #666; font-style: italic; padding: 20px;'>Trial has not started.</div>")
                
                with gr.TabItem("📁 My Evidence File"):
                    evidence_html = gr.HTML("<div style='color: #666; font-style: italic; padding: 20px;'>Trial has not started.</div>")

    # ─── EVENT HANDLERS ─────────────────────────────────────────────────────
    
    start_btn.click(
        api_reset,
        inputs=[case_dd, role_dd, seed_num],
        outputs=[state, header, evidence_html, history_html, jury_plot, action_type, evidence_id, arg_text, framing, obj_type, obj_target, submit_btn, verdict_msg]
    ).then(
        lambda: gr.update(interactive=True), None, auto_opp_btn
    )
    
    action_type.change(
        toggle_action_inputs,
        inputs=[action_type],
        outputs=[evidence_id, arg_text, framing, obj_type, obj_target]
    )
    
    submit_btn.click(
        api_step,
        inputs=[state, action_type, evidence_id, arg_text, framing, obj_type, obj_target],
        outputs=[state, header, evidence_html, history_html, jury_plot, action_type, evidence_id, arg_text, framing, obj_type, obj_target, submit_btn, verdict_msg]
    )
    
    auto_opp_btn.click(
        bot_step,
        inputs=[state],
        outputs=[state, header, evidence_html, history_html, jury_plot, action_type, evidence_id, arg_text, framing, obj_type, obj_target, submit_btn, verdict_msg]
    )

if __name__ == "__main__":
    print("Starting Adversa Gradio Dashboard on port 7860...")
    app.launch(server_name="0.0.0.0", server_port=7860)
