# app.py
from flask import Flask, render_template, request
import joblib
import pandas as pd
import pickle
import logging
import re
import os
from threading import Lock
import html

# Optional: local Llama integration (GGUF via llama-cpp-python)
# If you don't have local GGUF model, set llm = None (app still works but will show a warning)
try:
    from llama_cpp import Llama
except Exception:
    Llama = None

# ---------------------------
# Basic setup
# ---------------------------
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
llm_lock = Lock()

# ---------------------------
# Load ML prediction model & company mapping
# ---------------------------
# (these files should exist in the project directory)
model = joblib.load("career_model.pkl")
with open("company_mapping.pkl", "rb") as f:
    company_mapping = pickle.load(f)

# ---------------------------
# Local GGUF LLM setup (conservative defaults)
# Update LLM_MODEL_PATH to your gguf path if you have one.
# If you don't want a local model, set llm = None and the app will show a friendly message.
# ---------------------------
LLM_MODEL_PATH = "/Users/faatinashahul/Downloads/carrer_app/models/mistral-7b/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

llm = None
if Llama is not None and os.path.exists(LLM_MODEL_PATH):
    try:
        # conservative settings to reduce MPS/OOM issues on macOS
        llm = Llama(model_path=LLM_MODEL_PATH, n_ctx=1024, n_threads=4, n_gpu_layers=0)
        logging.info("Loaded local GGUF model at %s", LLM_MODEL_PATH)
    except Exception:
        logging.exception("Failed to load local GGUF LLM; continuing with llm=None")
        llm = None
else:
    logging.info("No local LLM available (Llama missing or gguf file not found).")

# ---------------------------
# Utility helpers (top-level)
# ---------------------------
def unique_preserve_order(items):
    """Return list of unique strings preserving original order (case-insensitive)."""
    seen = set()
    out = []
    for it in items:
        if not isinstance(it, str):
            it = str(it)
        k = it.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(it.strip())
    return out

def unique_preserve_order_by_name(dict_list):
    """For list of dicts with 'name' key, dedupe by lowercase name preserving order."""
    seen = set()
    out = []
    for d in dict_list:
        name = d.get("name", "") if isinstance(d, dict) else str(d)
        k = name.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append({"name": name.strip(), "desc": (d.get("desc","").strip() if isinstance(d, dict) else "")})
    return out

# ---------------------------
# LLM generation helper
# ---------------------------
def generate_text(prompt: str) -> str:
    """
    Generate text using the local LLM if available.
    Returns a string. On error or if llm is None, returns a helpful message.
    """
    if not llm:
        logging.warning("LLM not loaded; skipping LLM generation.")
        return "⚠ AI model not loaded — check file path or installation."

    with llm_lock:
        try:
            resp = llm(
                prompt,
                max_tokens=700,
                temperature=0.65,
                top_p=0.92,
                repeat_penalty=1.15
            )

            # handle various return formats from llama-cpp-python
            text = ""
            if isinstance(resp, dict) and resp.get("choices"):
                ch = resp["choices"][0]
                text = ch.get("text") or ch.get("content") or ch.get("generated_text", "")
            elif isinstance(resp, list) and len(resp) > 0:
                text = resp[0].get("generated_text", "") or str(resp[0])
            else:
                text = str(resp)

            if not (text and text.strip()):
                logging.error("LLM returned empty text. Raw resp: %s", resp)
                return "⚠ AI model returned no text."

            logging.info("LLM generated %d chars", len(text))
            return text.strip()

        except Exception as e:
            logging.exception("LLM generation failed")
            return f"⚠ AI model failed to generate output. Exception: {e}"

# ---------------------------
# Routes - simple pages
# ---------------------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/form")
def form():
    return render_template("index.html")

# ---------------------------
# /predict route (structured only output)
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read form data safely (use .get with defaults)
        def safe_int(key, default=0):
            try:
                return int(request.form.get(key, default))
            except Exception:
                return default

        data = {
            "UCS1511": safe_int("UCS1511"),
            "UCS1512": safe_int("UCS1512"),
            "UCS1601": safe_int("UCS1601"),
            "UCS1602": safe_int("UCS1602"),
            "UCS1603": safe_int("UCS1603"),
            "UCS1604": safe_int("UCS1604"),
            "UCS1701": safe_int("UCS1701"),
            "UCS1818": safe_int("UCS1818"),
            "Did you do any certification courses additionally?": safe_int("certification", 0),
        }

        # Gather multi-selects and custom inputs
        skills = request.form.getlist("skills")           # list
        interests_select = request.form.get("interests", "").strip()
        interests_custom = request.form.get("interests_custom", "").strip()
        interests = interests_custom or interests_select

        certificate = request.form.get("certificate_title", "").strip()
        certificate_custom = request.form.get("certificate_custom", "").strip()
        certificate = certificate_custom or certificate

        # combined text summary for LLM prompt
        combined_text = " | ".join(filter(None, [interests, ", ".join(skills), certificate]))
        data["combined_text"] = combined_text

        # Build dataframe and predict
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]

        # original mapping-based recommendations
        companies = list(company_mapping.get(prediction, []))
        companies = [c.title() for c in companies]
        companies = unique_preserve_order(companies)

        logging.info("Prediction: %s", prediction)
        logging.info("Mapped companies: %s", companies)

        # LLM prompt - explicitly require labeled sections for robust parsing
        top_companies_block = "\n".join("- " + c for c in companies) if companies else "- N/A"
        # Make prompt with clear newlines
        llm_prompt = (
            "You are Careerly — an expert career advisor. Produce one structured career report for me.\n\n"
            f"PREDICTED_ROLE: {prediction}\n"
            f"PROFILE: {combined_text}\n\n"
            "Output EXACTLY in the following sections (use blank lines between sections):\n\n"
            "WHY THIS ROLE:\n"
            "Write 2–4 clear sentences explaining why this role fits the student.\n\n"
            "KEY SKILLS:\n"
            "- Skill 1\n"
            "- Skill 2\n"
            "- Skill 3\n"
            "- Skill 4\n"
            "- Skill 5\n\n"
            "LEARNING ROADMAP:\n"
            "- Step 1: Recommended certification (platform + course if possible)\n"
            "- Step 2: Internship or hands-on project idea\n"
            "- Step 3: Portfolio project suggestion with measurable outcome\n"
            "- Step 4: Communities or meetups to join\n\n"
            "TOP COMPANIES:\n"
            "Provide a short, 1–2 sentence explanation for each company on the list, using this exact format per line:\n"
            "Company Name - Short explanation (one or two sentences).\n"
            f"Use these companies as a starting point:\n{top_companies_block}\n\n"
            "Do NOT add other sections. Keep skills short (max 6–8 words). Keep the tone professional and actionable.\n"
        )

        # Generate LLM output
        explanation_raw = generate_text(llm_prompt)
        explanation = explanation_raw  # keep name expected by templates if needed
        logging.info("RAW LLM OUTPUT (truncated): %s", (explanation_raw or "")[:800].replace("\n", "\\n"))

        # -------------------------
        # Parse LLM output into structured sections
        # -------------------------
        raw = explanation_raw or ""

        sections = {"why": "", "skills": [], "roadmap": [], "companies": []}

        # match labeled sections; allow a few header variants (case-insensitive)
        pattern = re.compile(
            r"(WHY THIS ROLE:|WHY_THIS_ROLE:|KEY SKILLS:|KEY_SKILLS:|KEY SKILLS:|LEARNING ROADMAP:|LEARNING_ROADMAP:|LEARNING ROADMAP:|TOP COMPANIES:|TOP_COMPANIES:|TOP COMPANIES:)\s*(.*?)\s*(?=(?:WHY THIS ROLE:|WHY_THIS_ROLE:|KEY SKILLS:|KEY_SKILLS:|LEARNING ROADMAP:|LEARNING_ROADMAP:|TOP COMPANIES:|TOP_COMPANIES:|$))",
            flags=re.IGNORECASE | re.DOTALL,
        )

        for m in pattern.finditer(raw):
            header = m.group(1).strip().upper().replace(":", "")
            body = m.group(2).strip()

            if header.startswith("WHY"):
                sections["why"] = body

            elif "KEY" in header:
                lines = [ln.strip() for ln in re.split(r"\r?\n", body) if ln.strip()]
                skills_clean = []
                for ln in lines:
                    # remove bullets / numbering
                    ln = re.sub(r'^[\-\u2022\*\d\.\)\s]+', '', ln).strip()
                    if ln:
                        skills_clean.append(ln)
                sections["skills"] = skills_clean

            elif "LEARNING" in header:
                lines = [ln.strip() for ln in re.split(r"\r?\n", body) if ln.strip()]
                roadmap_clean = []
                for ln in lines:
                    ln = re.sub(r'^[\-\u2022\*\d\.\)\s]+', '', ln).strip()
                    if ln:
                        roadmap_clean.append(ln)
                sections["roadmap"] = roadmap_clean

            elif "TOP" in header:
                lines = [ln.strip() for ln in re.split(r"\r?\n", body) if ln.strip()]
                parsed = []
                for ln in lines:
                    # expected format: "Company Name - explanation" or "Company Name: explanation"
                    if " - " in ln:
                        name, desc = ln.split(" - ", 1)
                    elif ":" in ln:
                        name, desc = ln.split(":", 1)
                    else:
                        parts = ln.split("-", 1)
                        if len(parts) == 2:
                            name, desc = parts[0], parts[1]
                        else:
                            name, desc = ln, ""
                    name = name.strip()
                    desc = desc.strip()
                    if name:
                        parsed.append({"name": name, "desc": desc})
                sections["companies"] = parsed

        # fallback: if parser found nothing, put raw into "why"
        if not any([sections["why"], sections["skills"], sections["roadmap"], sections["companies"]]) and raw:
            sections["why"] = raw

        # ---------- CLEAN & DEDUPE SECTION OUTPUT ----------
        # Clean roadmap lines: remove stray asterisks, "Step X", extra whitespace
        clean_roadmap = []
        for s in sections.get("roadmap", []):
            t = re.sub(r'[\*\u2022]+', '', s)  # remove bullets/asterisks
            t = re.sub(r'^\s*Step\s*\d+\s*[:\-]\s*', '', t, flags=re.I)  # remove "Step 1:"
            t = re.sub(r'\s{2,}', ' ', t).strip()
            t = html.unescape(t)
            if t:
                clean_roadmap.append(t)
        sections["roadmap"] = clean_roadmap

        # Clean skills: trim, remove numbering and asterisks
        clean_skills = []
        for s in sections.get("skills", []):
            t = re.sub(r'^[\-\u2022\*\d\.\)\s]+', '', s).strip()
            t = re.sub(r'\s{2,}', ' ', t)
            if t:
                clean_skills.append(t)
        sections["skills"] = clean_skills

        # Clean LLM-parsed companies and dedupe
        parsed_companies = sections.get("companies", [])
        sections["companies"] = unique_preserve_order_by_name(parsed_companies)

        # Also dedupe the mapping list you show under "Other Recommended Companies"
        companies = unique_preserve_order(companies)

        # Fallback: if top companies empty, try using mapping companies
        if not sections["companies"] and companies:
            sections["companies"] = [{"name": c, "desc": ""} for c in companies[:3]]
        # ---------- end cleaning ----------

        # log parsed lengths for debug
        logging.info(
            "Parsed sections lengths - why:%d skills:%d roadmap:%d companies:%d",
            len(sections["why"]),
            len(sections["skills"]),
            len(sections["roadmap"]),
            len(sections["companies"])
        )

        # --- parse top_companies entries which may be dict-like strings from the LLM ---
        import ast

        def parse_company_entry(entry):
            """
            Accepts an entry which might be:
            - a dict already: {'name': 'X', 'desc': 'Y'}
            - a string representation of a dict: "{'name': 'X', 'desc': 'Y'}"
            - a plain company name string: "Acme"
            Returns a dict with keys: name, desc (desc may be empty).
            """
            if not entry:
                return {"name": "", "desc": ""}

            # if it's already a dict
            if isinstance(entry, dict):
                return {"name": entry.get("name", "").strip(), "desc": entry.get("desc", "").strip()}

            # if it's a string, try to parse python dict literal safely
            if isinstance(entry, str):
                s = entry.strip()
                # common case: python dict string
                if s.startswith("{") and "name" in s.lower():
                    try:
                        # ast.literal_eval is safer than eval
                        parsed = ast.literal_eval(s)
                        if isinstance(parsed, dict):
                            return {"name": str(parsed.get("name", "")).strip(),
                                    "desc": str(parsed.get("desc", "")).strip()}
                    except Exception:
                        # fallthrough to other heuristics
                        pass

                # try JSON decode if it looks like JSON
                try:
                    import json
                    parsed = json.loads(s)
                    if isinstance(parsed, dict):
                        return {"name": str(parsed.get("name", "")).strip(),
                                "desc": str(parsed.get("desc", "")).strip()}
                except Exception:
                    pass

                # fallback: treat the string as a plain name, maybe contains a hyphenated description
                # e.g. "Amazon - global e-commerce ..."
                if " - " in s:
                    name, desc = s.split(" - ", 1)
                    return {"name": name.strip(), "desc": desc.strip()}
                # if comma separates name and description
                if "," in s and len(s.split(",")) > 1 and len(s.split(",")[0].split()) <= 4:
                    parts = s.split(",", 1)
                    return {"name": parts[0].strip(), "desc": parts[1].strip()}

                # otherwise treat whole string as name
                return {"name": s, "desc": ""}

            # anything else -> empty
            return {"name": str(entry), "desc": ""}

        # Build structured top company objects for the template
        top_company_objs = []
        for e in sections.get("companies", []):
            obj = parse_company_entry(e)
            if obj["name"]:               # only keep ones with a name
                top_company_objs.append(obj)

        # dedupe by lowercased name while preserving order
        seen = set()
        top_companies_final = []
        for c in top_company_objs:
            k = c["name"].strip().lower()
            if k and k not in seen:
                seen.add(k)
                top_companies_final.append(c)

        # Also dedupe mapping-based 'companies' (best workplaces)
        companies = unique_preserve_order(companies)   # you already defined unique_preserve_order earlier

        # Render template (top_companies is list of dicts with name + desc)
        

        return render_template(
            "result.html",
            result=prediction,
            companies=companies,               # mapping-based list (displayed in company cards)
            why=sections["why"],
            skills=sections["skills"],
            roadmap=sections["roadmap"],
            top_companies=top_companies_final   # structured list of dicts with name/desc
        )

    except Exception as e:
        logging.exception("Error in /predict")
        return f"ERROR: {e}", 500

# ---------------------------
# Run app
# ---------------------------
if __name__ == "__main__":
    # if port 5000 is busy, you can change to another port, e.g. app.run(debug=True, port=5001)
    app.run(debug=True)
