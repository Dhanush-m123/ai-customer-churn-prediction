const YN = ["Yes", "No"];
const YNI = ["Yes", "No", "No internet service"];

// name, label, type, options/default
const FIELDS = [
  ["gender", "Gender", "select", ["Male", "Female"]],
  ["SeniorCitizen", "Senior citizen", "select", [["0", "No"], ["1", "Yes"]]],
  ["Partner", "Has partner", "select", YN],
  ["Dependents", "Has dependents", "select", YN],
  ["tenure", "Tenure (months)", "number", { min: 0, max: 120, value: 12 }],
  ["Contract", "Contract", "select", ["Month-to-month", "One year", "Two year"]],
  ["PaymentMethod", "Payment method", "select",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]],
  ["PaperlessBilling", "Paperless billing", "select", YN],
  ["MonthlyCharges", "Monthly charges ($)", "number", { min: 0, max: 1000, step: "0.01", value: 70 }],
  ["TotalCharges", "Total charges ($, optional)", "number", { min: 0, step: "0.01", value: "", optional: true }],
  ["PhoneService", "Phone service", "select", YN],
  ["MultipleLines", "Multiple lines", "select", ["No", "Yes", "No phone service"]],
  ["InternetService", "Internet service", "select", ["Fiber optic", "DSL", "No"]],
  ["OnlineSecurity", "Online security", "select", YNI],
  ["OnlineBackup", "Online backup", "select", YNI],
  ["DeviceProtection", "Device protection", "select", YNI],
  ["TechSupport", "Tech support", "select", YNI],
  ["StreamingTV", "Streaming TV", "select", YNI],
  ["StreamingMovies", "Streaming movies", "select", YNI],
];

const $ = (id) => document.getElementById(id);
const fieldsEl = $("fields");

for (const [name, label, type, opts] of FIELDS) {
  const wrap = document.createElement("label");
  wrap.textContent = label;
  let el;
  if (type === "select") {
    el = document.createElement("select");
    for (const o of opts) {
      const [value, text] = Array.isArray(o) ? o : [o, o];
      const opt = document.createElement("option");
      opt.value = value; opt.textContent = text;
      el.appendChild(opt);
    }
  } else {
    el = document.createElement("input");
    el.type = "number";
    for (const [k, v] of Object.entries(opts)) if (k !== "optional") el[k] = v;
    if (opts.optional) el.dataset.optional = "1";
  }
  el.name = name; el.id = name;
  wrap.appendChild(el);
  fieldsEl.appendChild(wrap);
}

function collect() {
  const data = {};
  for (const [name, , type] of FIELDS) {
    const el = $(name);
    if (type === "number") {
      if (el.value === "") { if (el.dataset.optional) continue; el.classList.add("bad"); throw new Error(`Enter a value for "${el.parentElement.firstChild.textContent}".`); }
      el.classList.remove("bad");
      data[name] = Number(el.value);
    } else {
      data[name] = name === "SeniorCitizen" ? Number(el.value) : el.value;
    }
  }
  return data;
}

function showError(msg) {
  const e = $("error");
  e.textContent = msg; e.hidden = !msg;
}

function showResult(r) {
  $("empty").hidden = true; $("output").hidden = false;
  const risky = r.prediction === "Churn";
  const v = $("verdict");
  v.textContent = risky ? "Likely to churn" : "Likely to stay";
  v.className = risky ? "risk" : "safe";
  const pct = (r.churn_probability * 100).toFixed(1);
  $("pct").textContent = pct + "%";
  $("bar").style.width = pct + "%";
  $("bar").className = "bar" + (risky ? " risk" : "");
  $("mark").style.left = (r.threshold * 100) + "%";
  $("detail").textContent = `Decision threshold ${(r.threshold * 100).toFixed(0)}% (the dark tick). Model: ${r.model_version}. ${r.note}`;
}

async function loadHistory() {
  try {
    const res = await fetch("/history?limit=10");
    const rows = await res.json();
    const body = $("rows");
    body.replaceChildren();
    for (const r of rows) {
      const tr = document.createElement("tr");
      const cells = [r.created_at.replace("T", " ").replace("+00:00", ""), r.contract, r.tenure + " mo",
        "$" + Number(r.monthly_charges).toFixed(2), r.prediction, (r.churn_probability * 100).toFixed(1) + "%"];
      for (const c of cells) { const td = document.createElement("td"); td.textContent = c; tr.appendChild(td); }
      body.appendChild(tr);
    }
    $("no-history").hidden = rows.length > 0;
  } catch { /* history is optional */ }
}

$("form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  showError("");
  const btn = $("submit");
  let payload;
  try { payload = collect(); } catch (e) { showError(e.message); return; }
  btn.disabled = true; btn.textContent = "Estimating…";
  try {
    const res = await fetch("/predict", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) {
      const detail = typeof data.detail === "string" ? data.detail
        : "Some values are invalid: " + data.detail.map((d) => d.loc.slice(-1)[0]).join(", ");
      throw new Error(detail);
    }
    showResult(data);
    loadHistory();
  } catch (e) {
    showError(e.message || "Could not reach the server.");
  } finally {
    btn.disabled = false; btn.textContent = "Estimate churn risk";
  }
});

loadHistory();
