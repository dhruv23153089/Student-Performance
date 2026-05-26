import { startTransition, useEffect, useState } from "react";
import axios from "axios";
import "./App.css";
import bookImage from "./assets/book.png";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

const defaultForm = {
  weekly_self_study_hours: 12,
  attendance_percentage: 86,
  class_participation: 6,
};

const localPresets = [
  {
    name: "High Achiever",
    description: "High consistency and strong classroom presence.",
    values: {
      weekly_self_study_hours: 18,
      attendance_percentage: 94,
      class_participation: 8,
    },
  },
  {
    name: "Balanced Learner",
    description: "Stable profile with room for growth.",
    values: {
      weekly_self_study_hours: 12,
      attendance_percentage: 86,
      class_participation: 6,
    },
  },
  {
    name: "Needs Support",
    description: "Helpful for planning an improvement path.",
    values: {
      weekly_self_study_hours: 6,
      attendance_percentage: 72,
      class_participation: 4,
    },
  },
];

const fieldConfig = [
  {
    name: "weekly_self_study_hours",
    label: "Weekly Self Study",
    min: 0,
    max: 40,
    step: 0.5,
    hint: "Time spent learning outside class.",
  },
  {
    name: "attendance_percentage",
    label: "Attendance",
    min: 0,
    max: 100,
    step: 1,
    hint: "Class attendance across the term.",
  },
  {
    name: "class_participation",
    label: "Class Participation",
    min: 0,
    max: 10,
    step: 0.5,
    hint: "Questions, answers, and discussion activity.",
  },
];

function formatValue(name, value) {
  if (name === "weekly_self_study_hours") return `${value.toFixed(1)} hrs`;
  if (name === "attendance_percentage") return `${Math.round(value)}%`;
  return `${value.toFixed(1)}/10`;
}

function getScoreTone(score) {
  if (score >= 85) return "tone-excellent";
  if (score >= 70) return "tone-strong";
  if (score >= 55) return "tone-developing";
  return "tone-risk";
}

function App() {
  const [form, setForm] = useState(defaultForm);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [presets, setPresets] = useState(localPresets);

  useEffect(() => {
    let isMounted = true;

    async function loadPresets() {
      try {
        const response = await axios.get(`${API_BASE}/presets`);
        if (!isMounted) return;

        startTransition(() => {
          setPresets(
            response.data.profiles.map((profile) => ({
              name: profile.name,
              description: "Quick preset for scenario testing.",
              values: {
                weekly_self_study_hours: profile.weekly_self_study_hours,
                attendance_percentage: profile.attendance_percentage,
                class_participation: profile.class_participation,
              },
            }))
          );
        });
      } catch {
        // Keep local presets when the API is unavailable.
      }
    }

    loadPresets();

    return () => {
      isMounted = false;
    };
  }, []);

  const engagementPreview = Math.round(
    ((form.weekly_self_study_hours / 40 +
      form.attendance_percentage / 100 +
      form.class_participation / 10) /
      3) *
      100
  );

  const summaryChips = [
    {
      label: "Study",
      value: formatValue("weekly_self_study_hours", form.weekly_self_study_hours),
    },
    {
      label: "Attendance",
      value: formatValue("attendance_percentage", form.attendance_percentage),
    },
    {
      label: "Participation",
      value: formatValue("class_participation", form.class_participation),
    },
    {
      label: "Engagement",
      value: `${engagementPreview}`,
    },
  ];

  function handleChange(name, value) {
    setForm((current) => ({
      ...current,
      [name]: Number(value),
    }));
  }

  function applyPreset(preset) {
    setForm(preset.values);
    setError("");
    setResult(null);
  }

  function resetForm() {
    setForm(defaultForm);
    setError("");
    setResult(null);
  }

  async function predict() {
    setLoading(true);
    setError("");

    try {
      const response = await axios.post(`${API_BASE}/predict`, form);
      startTransition(() => {
        setResult(response.data);
      });
    } catch (requestError) {
      setError(
        requestError?.response?.data?.detail?.[0]?.msg ||
          "Unable to connect to the prediction service. Make sure the backend is running."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="topbar-main">
          <div className="topbar-copy">
            <p className="eyebrow">Student Performance Intelligence</p>
            <h1>Predict score and grade with a cleaner, faster workflow.</h1>
            <p className="topbar-text">
              Adjust the student profile, run the forecast, and review the most useful
              insights without the extra visual noise.
            </p>
          </div>

          <div className="hero-image-wrap">
            <img alt="Books and study illustration" className="hero-image" src={bookImage} />
          </div>
        </div>

        <div className="summary-strip">
          {summaryChips.map((chip) => (
            <div className="summary-chip" key={chip.label}>
              <span>{chip.label}</span>
              <strong>{chip.value}</strong>
            </div>
          ))}
        </div>
      </header>

      <main className="workspace">
        <section className="panel panel-form">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Scenario Lab</p>
              <h2>Student profile</h2>
            </div>
            <button className="ghost-button" onClick={resetForm} type="button">
              Reset
            </button>
          </div>

          <div className="preset-row">
            {presets.map((preset) => (
              <button
                className="preset-card"
                key={preset.name}
                onClick={() => applyPreset(preset)}
                type="button"
              >
                <strong>{preset.name}</strong>
                <span>{preset.description}</span>
              </button>
            ))}
          </div>

          <div className="field-grid">
            {fieldConfig.map((field) => (
              <label className="field-card" key={field.name}>
                <div className="field-head">
                  <div>
                    <span>{field.label}</span>
                    <small>{field.hint}</small>
                  </div>
                  <strong>{formatValue(field.name, form[field.name])}</strong>
                </div>

                <input
                  max={field.max}
                  min={field.min}
                  onChange={(event) => handleChange(field.name, event.target.value)}
                  step={field.step}
                  type="range"
                  value={form[field.name]}
                />

                <input
                  className="number-input"
                  max={field.max}
                  min={field.min}
                  onChange={(event) => handleChange(field.name, event.target.value)}
                  step={field.step}
                  type="number"
                  value={form[field.name]}
                />
              </label>
            ))}
          </div>

          <div className="action-row">
            <button className="primary-button" disabled={loading} onClick={predict} type="button">
              {loading ? "Analyzing..." : "Predict Score & Grade"}
            </button>
            <p className="helper-text">
              Small improvements in study consistency and participation can shift the
              prediction noticeably.
            </p>
          </div>

          {error ? <div className="error-banner">{error}</div> : null}
        </section>

        <section className="panel panel-result">
          <div className="section-heading compact">
            <div>
              <p className="eyebrow">Prediction Output</p>
              <h2>Score outlook</h2>
            </div>
            <span className="mini-badge">Live</span>
          </div>

          {result ? (
            <div className="result-stack">
              <div className={`score-banner ${getScoreTone(result.predicted_score)}`}>
                <div>
                  <span>Predicted Score</span>
                  <strong>{result.predicted_score}</strong>
                </div>
                <div>
                  <span>Predicted Grade</span>
                  <strong>{result.predicted_grade}</strong>
                </div>
              </div>

              <div className="result-meta">
                <div className="metric-card">
                  <span>Performance Band</span>
                  <strong>{result.performance_band}</strong>
                </div>
                <div className="metric-card">
                  <span>Risk Level</span>
                  <strong>{result.risk_level}</strong>
                </div>
                <div className="metric-card">
                  <span>Focus Area</span>
                  <strong>{result.focus_area}</strong>
                </div>
                <div className="metric-card">
                  <span>Engagement</span>
                  <strong>{result.insights.engagement_score}</strong>
                </div>
                <div className="metric-card">
                  <span>Consistency</span>
                  <strong>{result.insights.consistency_index}</strong>
                </div>
                <div className="metric-card">
                  <span>Support Need</span>
                  <strong>{result.insights.support_need_index}</strong>
                </div>
              </div>

              <div className="text-panels">
                <div className="list-card">
                  <h3>Strengths</h3>
                  <ul>
                    {result.strengths.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>

                <div className="list-card">
                  <h3>Next Moves</h3>
                  <ul>
                    {result.recommendations.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          ) : (
            <div className="empty-state">
              <strong>No prediction yet</strong>
              <p>Choose a preset or adjust the sliders, then run the forecast.</p>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
