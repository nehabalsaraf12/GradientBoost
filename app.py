"""
Salary Estimator - Flask app serving gradient.pkl (sklearn GradientBoostingRegressor)

Model inputs (in this exact order, all numeric):
    Department, Experience_Years, Education_Level, Age, Gender, City

Deploy on Render:
    Build command : pip install -r requirements.txt
    Start command : gunicorn app:app
"""

import os
import pickle
import warnings

import pandas as pd
from flask import Flask, jsonify, render_template_string, request

# --------------------------------------------------------------------------
# 1. CONFIG  -  EDIT THIS BLOCK TO MATCH HOW YOU ENCODED YOUR TRAINING DATA
# --------------------------------------------------------------------------
# The model was trained on label-encoded numbers. The position of each label in
# these lists IS the number sent to the model (index 0 -> 0, index 1 -> 1 ...).
# If you used sklearn's LabelEncoder, the order is alphabetical (as below).
# If you used a different mapping, reorder / rename the labels to match it.
DEPARTMENTS = ["Finance", "HR", "IT", "Marketing", "Sales"]          # codes 0-4
EDUCATION = ["Bachelor's", "High School", "Master's", "PhD"]         # codes 0-3
GENDERS = ["Female", "Male"]                                         # codes 0-1
CITIES = ["Bangalore", "Chennai", "Delhi", "Mumbai", "Pune"]         # codes 0-4

CURRENCY = "₹"            # symbol shown before every amount
GAUGE_MIN = 20_000        # left end of the result bar
GAUGE_MAX = 130_000       # right end of the result bar

# Ranges the model has actually seen during training (used for soft warnings)
TRAINED_AGE = (22, 58)
TRAINED_EXPERIENCE = (0, 20)

FEATURE_ORDER = [
    "Department",
    "Experience_Years",
    "Education_Level",
    "Age",
    "Gender",
    "City",
]

# --------------------------------------------------------------------------
# 2. LOAD MODEL (once, at start-up)
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "gradient.pkl")

warnings.filterwarnings("ignore", category=UserWarning)
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

# --------------------------------------------------------------------------
# 3. HERO ILLUSTRATION (inline SVG - no external image needed)
# --------------------------------------------------------------------------
HERO_SVG = """
<svg class="hero-art" viewBox="0 0 520 380" role="img"
     aria-label="Illustration of rising salary bars with a glowing sun and a coin"
     xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="barA" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#7C6CFF"/><stop offset="1" stop-color="#4B3FD6"/>
    </linearGradient>
    <linearGradient id="barB" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#37E0C8"/><stop offset="1" stop-color="#139C9C"/>
    </linearGradient>
    <linearGradient id="barC" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#FF8FB1"/><stop offset="1" stop-color="#E5447F"/>
    </linearGradient>
    <radialGradient id="sun" cx="50%" cy="50%" r="50%">
      <stop offset="0" stop-color="#FFE9A8"/>
      <stop offset="0.55" stop-color="#FFC857" stop-opacity="0.55"/>
      <stop offset="1" stop-color="#FFC857" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="coin" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#FFE08A"/><stop offset="1" stop-color="#F0A92B"/>
    </linearGradient>
  </defs>

  <circle cx="400" cy="92" r="92" fill="url(#sun)"/>
  <circle cx="400" cy="92" r="34" fill="#FFE9A8"/>

  <g fill="none" stroke="#ffffff" stroke-opacity="0.16" stroke-width="1">
    <line x1="30" y1="330" x2="490" y2="330"/>
    <line x1="30" y1="270" x2="490" y2="270"/>
    <line x1="30" y1="210" x2="490" y2="210"/>
    <line x1="30" y1="150" x2="490" y2="150"/>
  </g>

  <rect x="62"  y="250" width="64" height="80"  rx="12" fill="url(#barA)"/>
  <rect x="146" y="205" width="64" height="125" rx="12" fill="url(#barB)"/>
  <rect x="230" y="160" width="64" height="170" rx="12" fill="url(#barC)"/>
  <rect x="314" y="108" width="64" height="222" rx="12" fill="url(#barA)"/>
  <rect x="398" y="52"  width="64" height="278" rx="12" fill="url(#barB)" opacity="0.92"/>

  <polyline points="94,236 178,190 262,146 346,94 430,40"
            fill="none" stroke="#ffffff" stroke-width="3" stroke-linecap="round"
            stroke-linejoin="round" stroke-dasharray="2 9" opacity="0.85"/>

  <g transform="translate(120 112)">
    <circle r="38" fill="url(#coin)"/>
    <circle r="29" fill="none" stroke="#B8761A" stroke-opacity="0.45" stroke-width="2"/>
    <text text-anchor="middle" y="11" font-family="Sora, Arial, sans-serif"
          font-weight="700" font-size="32" fill="#8A530C">₹</text>
  </g>

  <g fill="#ffffff">
    <circle cx="48"  cy="60"  r="3" opacity="0.8"/>
    <circle cx="250" cy="48"  r="2.5" opacity="0.6"/>
    <circle cx="488" cy="190" r="3" opacity="0.7"/>
    <circle cx="40"  cy="190" r="2" opacity="0.6"/>
  </g>
</svg>
"""

# --------------------------------------------------------------------------
# 4. PAGE TEMPLATE  (HTML + CSS + a little JS, all in this file)
# --------------------------------------------------------------------------
PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Salary Estimator</title>
  <meta name="description" content="Estimate an annual salary from department, experience, education, age, gender and city.">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Sora:wght@500;700;800&display=swap" rel="stylesheet">

  <style>
    /* ---------- Design tokens ---------- */
    :root {
      --ink:        #12143A;   /* deep indigo, page base      */
      --ink-2:      #1E1F5C;
      --violet:     #6C5CFF;   /* primary accent              */
      --violet-dk:  #4B3FD6;
      --teal:       #2FD9C0;   /* secondary accent            */
      --pink:       #FF6B9A;   /* warm accent                 */
      --gold:       #FFC857;
      --paper:      #F7F6FF;   /* card surface                */
      --text:       #1A1B3F;
      --muted:      #6A6C92;
      --line:       #E1DFF5;
      --danger:     #C7254E;
      --font-display: "Sora", "Segoe UI", system-ui, sans-serif;
      --font-body:    "DM Sans", "Segoe UI", system-ui, sans-serif;
    }

    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: var(--font-body);
      color: #fff;
      background:
        radial-gradient(900px 600px at 8% 12%,  rgba(108, 92, 255, 0.55), transparent 60%),
        radial-gradient(800px 560px at 95% 90%, rgba(47, 217, 192, 0.38), transparent 60%),
        radial-gradient(600px 400px at 70% 5%,  rgba(255, 107, 154, 0.25), transparent 60%),
        linear-gradient(135deg, var(--ink) 0%, var(--ink-2) 55%, #0E3A5C 100%);
      background-attachment: fixed;
    }

    /* ---------- Layout ---------- */
    .shell {
      max-width: 1180px;
      margin: 0 auto;
      padding: 40px 24px 56px;
      display: grid;
      grid-template-columns: 1.05fr 0.95fr;
      gap: 56px;
      align-items: center;
      min-height: 100vh;
    }

    .brand {
      display: flex; align-items: center; gap: 10px;
      font-family: var(--font-display);
      font-weight: 700; font-size: 1rem; letter-spacing: 0.01em;
      margin-bottom: 28px;
    }
    .brand-mark {
      width: 30px; height: 30px; border-radius: 9px;
      background: linear-gradient(135deg, var(--violet), var(--teal));
      display: grid; place-items: center; font-size: 0.95rem;
    }

    /* ---------- Hero ---------- */
    .hero h1 {
      font-family: var(--font-display);
      font-weight: 800;
      font-size: clamp(2.2rem, 4.6vw, 3.6rem);
      line-height: 1.06;
      letter-spacing: -0.02em;
      margin: 0 0 16px;
      max-width: 14ch;
    }
    .hero p.lead {
      font-size: 1.1rem; line-height: 1.6;
      color: rgba(255, 255, 255, 0.78);
      max-width: 44ch; margin: 0 0 30px;
    }
    .hero-art {
      width: 100%; max-width: 520px; height: auto; display: block;
      filter: drop-shadow(0 30px 40px rgba(5, 6, 40, 0.45));
    }
    .facts {
      display: flex; flex-wrap: wrap; gap: 10px; margin: 26px 0 0; padding: 0; list-style: none;
    }
    .facts li {
      font-size: 0.88rem; padding: 8px 14px; border-radius: 999px;
      background: rgba(255, 255, 255, 0.09);
      border: 1px solid rgba(255, 255, 255, 0.16);
      backdrop-filter: blur(6px);
    }

    /* ---------- Form card ---------- */
    .card {
      background: var(--paper);
      color: var(--text);
      border-radius: 28px;
      padding: 34px 32px 30px;
      box-shadow:
        0 40px 80px -20px rgba(4, 5, 36, 0.65),
        0 0 0 1px rgba(255, 255, 255, 0.5) inset;
    }
    .card h2 {
      font-family: var(--font-display);
      font-size: 1.45rem; margin: 0 0 4px; letter-spacing: -0.01em;
    }
    .card .sub { margin: 0 0 22px; color: var(--muted); font-size: 0.95rem; }

    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px 14px; }
    .field { display: flex; flex-direction: column; gap: 6px; }
    .field.full { grid-column: 1 / -1; }
    .field label { font-size: 0.88rem; font-weight: 700; color: var(--text); }

    .field select,
    .field input {
      width: 100%;
      font: inherit; font-size: 1rem; color: var(--text);
      padding: 12px 14px;
      background: #fff;
      border: 1.5px solid var(--line);
      border-radius: 14px;
      transition: border-color 0.15s, box-shadow 0.15s;
      appearance: none;
      -webkit-appearance: none;
    }
    .field select {
      background-image:
        linear-gradient(45deg, transparent 50%, var(--muted) 50%),
        linear-gradient(135deg, var(--muted) 50%, transparent 50%);
      background-position: calc(100% - 20px) 50%, calc(100% - 14px) 50%;
      background-size: 6px 6px, 6px 6px;
      background-repeat: no-repeat;
      padding-right: 38px;
    }
    .field select:invalid { color: var(--muted); }
    .field input:hover, .field select:hover { border-color: #C9C5EE; }
    .field input:focus-visible, .field select:focus-visible {
      outline: none;
      border-color: var(--violet);
      box-shadow: 0 0 0 4px rgba(108, 92, 255, 0.2);
    }

    .btn {
      margin-top: 22px; width: 100%;
      font-family: var(--font-display); font-weight: 700; font-size: 1.02rem;
      color: #fff; border: 0; cursor: pointer;
      padding: 15px 20px; border-radius: 16px;
      background: linear-gradient(120deg, var(--violet) 0%, var(--violet-dk) 45%, #13A4A0 120%);
      box-shadow: 0 14px 28px -10px rgba(75, 63, 214, 0.7);
      transition: transform 0.15s, box-shadow 0.15s, filter 0.15s;
    }
    .btn:hover { transform: translateY(-2px); filter: brightness(1.06); box-shadow: 0 18px 30px -10px rgba(75, 63, 214, 0.8); }
    .btn:active { transform: translateY(0); }
    .btn:focus-visible { outline: 3px solid var(--gold); outline-offset: 3px; }

    /* ---------- Messages ---------- */
    .errors {
      margin: 0 0 18px; padding: 12px 16px; border-radius: 14px;
      background: #FFE9EF; color: var(--danger); font-size: 0.92rem;
      border: 1px solid #F7C2D1;
    }
    .errors ul { margin: 0; padding-left: 18px; }

    /* ---------- Result ---------- */
    .result {
      margin: 0 0 24px; padding: 22px 22px 18px; border-radius: 20px; color: #fff;
      background:
        radial-gradient(260px 160px at 100% 0%, rgba(47, 217, 192, 0.45), transparent 70%),
        linear-gradient(135deg, var(--ink-2), var(--violet-dk));
      scroll-margin-top: 20px;
    }
    .result .label { margin: 0; font-size: 0.9rem; color: rgba(255, 255, 255, 0.75); }
    .result .value {
      margin: 4px 0 14px;
      font-family: var(--font-display); font-weight: 800;
      font-size: clamp(2.2rem, 6vw, 3rem); letter-spacing: -0.02em; line-height: 1.1;
    }
    .result .value .cur { color: var(--gold); margin-right: 4px; }

    .gauge {
      position: relative; height: 10px; border-radius: 999px;
      background: rgba(255, 255, 255, 0.18);
    }
    .gauge-fill {
      height: 100%; border-radius: inherit;
      background: linear-gradient(90deg, var(--teal), var(--gold) 60%, var(--pink));
    }
    .gauge-dot {
      position: absolute; top: 50%; width: 20px; height: 20px; border-radius: 50%;
      background: #fff; border: 4px solid var(--violet);
      transform: translate(-50%, -50%);
      box-shadow: 0 4px 10px rgba(0, 0, 0, 0.35);
    }
    .gauge-scale {
      display: flex; justify-content: space-between;
      margin-top: 8px; font-size: 0.78rem; color: rgba(255, 255, 255, 0.65);
    }
    .result .monthly { margin: 12px 0 0; font-size: 0.95rem; color: rgba(255, 255, 255, 0.85); }
    .result .notes {
      margin: 12px 0 0; padding: 10px 12px; border-radius: 12px;
      background: rgba(255, 200, 87, 0.16); color: #FFE7A8; font-size: 0.85rem;
    }

    footer.foot {
      grid-column: 1 / -1; text-align: center;
      font-size: 0.82rem; color: rgba(255, 255, 255, 0.55);
    }

    /* ---------- Responsive ---------- */
    @media (max-width: 920px) {
      .shell { grid-template-columns: 1fr; gap: 36px; min-height: 0; padding-top: 28px; }
      .hero h1 { max-width: none; }
      .hero-art { max-width: 380px; margin: 0 auto; }
      .facts { justify-content: flex-start; }
    }
    @media (max-width: 520px) {
      .grid { grid-template-columns: 1fr; }
      .card { padding: 26px 20px 24px; border-radius: 24px; }
    }
    @media (prefers-reduced-motion: reduce) {
      html { scroll-behavior: auto; }
      .btn { transition: none; }
    }
  </style>
</head>

<body>
  <main class="shell">

    <!-- Left: message + illustration -->
    <section class="hero">
      <div class="brand"><span class="brand-mark" aria-hidden="true">₹</span> Salary Estimator</div>
      <h1>What should this role pay?</h1>
      <p class="lead">
        Describe an employee's profile and get an estimated annual salary,
        predicted by a gradient boosting model trained on past salary data.
      </p>
      {{ hero_svg|safe }}
      <ul class="facts">
        <li>100 decision trees</li>
        <li>6 profile inputs</li>
        <li>Instant estimate</li>
      </ul>
    </section>

    <!-- Right: form card -->
    <section class="card" aria-labelledby="form-title">
      <h2 id="form-title">Employee profile</h2>
      <p class="sub">Fill in all six fields to see the estimate.</p>

      {% if result %}
      <div class="result" id="result" role="status" aria-live="polite">
        <p class="label">Estimated annual salary</p>
        <p class="value">
          <span class="cur">{{ currency }}</span><span id="amount" data-value="{{ result.value }}">{{ result.text }}</span>
        </p>
        <div class="gauge" aria-hidden="true">
          <div class="gauge-fill" style="width: {{ result.pct }}%"></div>
          <div class="gauge-dot" style="left: {{ result.pct }}%"></div>
        </div>
        <div class="gauge-scale" aria-hidden="true">
          <span>{{ gauge_min }}</span><span>{{ gauge_max }}</span>
        </div>
        <p class="monthly">About {{ currency }}{{ result.monthly }} per month</p>
        {% for note in result.notes %}<p class="notes">{{ note }}</p>{% endfor %}
      </div>
      {% endif %}

      {% if errors %}
      <div class="errors" role="alert">
        <ul>{% for e in errors %}<li>{{ e }}</li>{% endfor %}</ul>
      </div>
      {% endif %}

      <form method="post" action="/" novalidate>
        <div class="grid">

          <div class="field">
            <label for="department">Department</label>
            <select id="department" name="department" required>
              <option value="" disabled {% if form.department == '' %}selected{% endif %}>Choose department</option>
              {% for name in departments %}
              <option value="{{ loop.index0 }}" {% if form.department == loop.index0|string %}selected{% endif %}>{{ name }}</option>
              {% endfor %}
            </select>
          </div>

          <div class="field">
            <label for="education">Education level</label>
            <select id="education" name="education" required>
              <option value="" disabled {% if form.education == '' %}selected{% endif %}>Choose education</option>
              {% for name in education %}
              <option value="{{ loop.index0 }}" {% if form.education == loop.index0|string %}selected{% endif %}>{{ name }}</option>
              {% endfor %}
            </select>
          </div>

          <div class="field">
            <label for="experience">Experience (years)</label>
            <input id="experience" name="experience" type="number" inputmode="decimal"
                   min="0" max="45" step="0.5" placeholder="e.g. 6" value="{{ form.experience }}" required>
          </div>

          <div class="field">
            <label for="age">Age</label>
            <input id="age" name="age" type="number" inputmode="numeric"
                   min="18" max="70" step="1" placeholder="e.g. 32" value="{{ form.age }}" required>
          </div>

          <div class="field">
            <label for="gender">Gender</label>
            <select id="gender" name="gender" required>
              <option value="" disabled {% if form.gender == '' %}selected{% endif %}>Choose gender</option>
              {% for name in genders %}
              <option value="{{ loop.index0 }}" {% if form.gender == loop.index0|string %}selected{% endif %}>{{ name }}</option>
              {% endfor %}
            </select>
          </div>

          <div class="field">
            <label for="city">City</label>
            <select id="city" name="city" required>
              <option value="" disabled {% if form.city == '' %}selected{% endif %}>Choose city</option>
              {% for name in cities %}
              <option value="{{ loop.index0 }}" {% if form.city == loop.index0|string %}selected{% endif %}>{{ name }}</option>
              {% endfor %}
            </select>
          </div>

        </div>
        <button class="btn" type="submit">Estimate salary</button>
      </form>
    </section>

    <footer class="foot">Estimates come from a statistical model and are not an offer or a guarantee.</footer>
  </main>

  <script>
    // One moment of motion: count the salary up and bring the result into view.
    (function () {
      var el = document.getElementById("amount");
      if (!el) return;
      var box = document.getElementById("result");
      if (box) box.scrollIntoView({ block: "center" });

      var target = parseFloat(el.getAttribute("data-value"));
      var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      if (reduce || isNaN(target)) return;

      var finalText = el.textContent;
      var start = null, duration = 900;
      function tick(ts) {
        if (start === null) start = ts;
        var p = Math.min((ts - start) / duration, 1);
        var eased = 1 - Math.pow(1 - p, 3);
        el.textContent = Math.round(target * eased).toLocaleString("en-IN");
        if (p < 1) { requestAnimationFrame(tick); } else { el.textContent = finalText; }
      }
      requestAnimationFrame(tick);
    })();
  </script>
</body>
</html>
"""

# --------------------------------------------------------------------------
# 5. HELPERS
# --------------------------------------------------------------------------
def indian_format(n):
    """Group digits the Indian way: 1234567 -> '12,34,567'."""
    n = int(round(n))
    s = str(abs(n))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts) + "," + tail
    return ("-" if n < 0 else "") + s


def short_money(n):
    """120000 -> '$120k' for the gauge scale."""
    return f"{CURRENCY}{n // 1000}k"


def parse_and_validate(data):
    """Validate raw form / JSON data. Returns (clean_values, errors)."""
    errors = []
    clean = {}

    def pick(key, options, label):
        raw = str(data.get(key, "")).strip()
        if raw == "":
            errors.append(f"Choose a {label}.")
            return None
        try:
            idx = int(raw)
        except ValueError:
            errors.append(f"{label.capitalize()} is not valid.")
            return None
        if not 0 <= idx < len(options):
            errors.append(f"{label.capitalize()} is not valid.")
            return None
        return idx

    def number(key, label, lo, hi):
        raw = str(data.get(key, "")).strip()
        if raw == "":
            errors.append(f"Enter {label}.")
            return None
        try:
            val = float(raw)
        except ValueError:
            errors.append(f"{label.capitalize()} must be a number.")
            return None
        if not lo <= val <= hi:
            errors.append(f"{label.capitalize()} must be between {lo} and {hi}.")
            return None
        return val

    clean["Department"] = pick("department", DEPARTMENTS, "department")
    clean["Experience_Years"] = number("experience", "years of experience", 0, 45)
    clean["Education_Level"] = pick("education", EDUCATION, "education level")
    clean["Age"] = number("age", "age", 18, 70)
    clean["Gender"] = pick("gender", GENDERS, "gender")
    clean["City"] = pick("city", CITIES, "city")

    if not errors and clean["Age"] - clean["Experience_Years"] < 14:
        errors.append("Experience looks too high for this age. Check both values.")

    return clean, errors


def run_prediction(clean):
    """Run the model and build everything the result card needs."""
    frame = pd.DataFrame([[clean[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)
    value = max(float(model.predict(frame)[0]), 0.0)

    pct = (value - GAUGE_MIN) / (GAUGE_MAX - GAUGE_MIN) * 100
    pct = round(min(max(pct, 3), 97), 1)

    notes = []
    if not TRAINED_AGE[0] <= clean["Age"] <= TRAINED_AGE[1]:
        notes.append(
            f"Age {int(clean['Age'])} is outside the {TRAINED_AGE[0]}-{TRAINED_AGE[1]} "
            "range the model learned from, so treat this estimate with care."
        )
    if not TRAINED_EXPERIENCE[0] <= clean["Experience_Years"] <= TRAINED_EXPERIENCE[1]:
        notes.append(
            f"{clean['Experience_Years']:g} years of experience is outside the "
            f"{TRAINED_EXPERIENCE[0]}-{TRAINED_EXPERIENCE[1]} year range the model learned from."
        )

    return {
        "value": round(value),
        "text": indian_format(value),
        "monthly": indian_format(value / 12),
        "pct": pct,
        "notes": notes,
    }


# --------------------------------------------------------------------------
# 6. ROUTES
# --------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    form = {k: "" for k in ("department", "experience", "education", "age", "gender", "city")}
    errors, result = [], None

    if request.method == "POST":
        for key in form:
            form[key] = request.form.get(key, "").strip()
        clean, errors = parse_and_validate(form)
        if not errors:
            result = run_prediction(clean)

    return render_template_string(
        PAGE,
        hero_svg=HERO_SVG,
        form=form,
        errors=errors,
        result=result,
        departments=DEPARTMENTS,
        education=EDUCATION,
        genders=GENDERS,
        cities=CITIES,
        currency=CURRENCY,
        gauge_min=short_money(GAUGE_MIN),
        gauge_max=short_money(GAUGE_MAX),
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API. Body keys: department, experience, education, age, gender, city
    (department / education / gender / city are the numeric codes)."""
    payload = request.get_json(silent=True) or {}
    clean, errors = parse_and_validate(payload)
    if errors:
        return jsonify({"ok": False, "errors": errors}), 400
    out = run_prediction(clean)
    return jsonify({"ok": True, "predicted_salary": out["value"], "notes": out["notes"]})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)