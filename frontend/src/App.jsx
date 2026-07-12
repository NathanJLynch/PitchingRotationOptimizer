import React, { useState, useCallback } from "react";
import { ChevronDown, Loader2, RefreshCw, Play, AlertCircle } from "lucide-react";

// ─────────────────────────────────────────────────────────────
// CONFIG
// ─────────────────────────────────────────────────────────────

const API_BASE = "http://localhost:8000";

const TEAMS = [
  { id: 1, name: "New York Yankees", abbr: "NYY" },
  { id: 2, name: "Boston Red Sox", abbr: "BOS" },
  { id: 3, name: "Toronto Blue Jays", abbr: "TOR" },
  { id: 4, name: "Baltimore Orioles", abbr: "BAL" },
  { id: 5, name: "Tampa Bay Rays", abbr: "TB" },
  { id: 6, name: "Minnesota Twins", abbr: "MIN" },
  { id: 7, name: "Detroit Tigers", abbr: "DET" },
  { id: 8, name: "Kansas City Royals", abbr: "KC" },
  { id: 9, name: "Cleveland Guardians", abbr: "CLE" },
  { id: 10, name: "Chicago White Sox", abbr: "CWS" },
  { id: 11, name: "Houston Astros", abbr: "HOU" },
  { id: 12, name: "Texas Rangers", abbr: "TEX" },
  { id: 13, name: "Los Angeles Angels", abbr: "LAA" },
  { id: 14, name: "Seattle Mariners", abbr: "SEA" },
  { id: 15, name: "Oakland Athletics", abbr: "OAK" },
  { id: 16, name: "New York Mets", abbr: "NYM" },
  { id: 17, name: "Philadelphia Phillies", abbr: "PHI" },
  { id: 18, name: "Atlanta Braves", abbr: "ATL" },
  { id: 19, name: "Washington Nationals", abbr: "WSH" },
  { id: 20, name: "Miami Marlins", abbr: "MIA" },
  { id: 21, name: "Milwaukee Brewers", abbr: "MIL" },
  { id: 22, name: "Chicago Cubs", abbr: "CHC" },
  { id: 23, name: "St. Louis Cardinals", abbr: "STL" },
  { id: 24, name: "Cincinnati Reds", abbr: "CIN" },
  { id: 25, name: "Pittsburgh Pirates", abbr: "PIT" },
  { id: 26, name: "Los Angeles Dodgers", abbr: "LAD" },
  { id: 27, name: "San Francisco Giants", abbr: "SF" },
  { id: 28, name: "San Diego Padres", abbr: "SD" },
  { id: 29, name: "Arizona Diamondbacks", abbr: "ARI" },
  { id: 30, name: "Colorado Rockies", abbr: "COL" },
];

// ─────────────────────────────────────────────────────────────
// SIGNATURE ELEMENT: stitched baseball fatigue indicator
// Fatigue 0-4 rendered as a row of 5 small ball icons —
// filled (fresh) vs. outline (gassed) — instead of a generic bar.
// ─────────────────────────────────────────────────────────────

function BallIcon({ filled }) {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" className="ball-icon">
      <circle
        cx="12" cy="12" r="10"
        fill={filled ? "#F7F4EA" : "none"}
        stroke={filled ? "#0B2545" : "#0B254540"}
        strokeWidth="1.5"
      />
      {filled && (
        <>
          <path d="M7 4.5 Q 12 12, 7 19.5" fill="none" stroke="#C8102E" strokeWidth="1" />
          <path d="M17 4.5 Q 12 12, 17 19.5" fill="none" stroke="#C8102E" strokeWidth="1" />
        </>
      )}
    </svg>
  );
}

function FatigueIndicator({ level }) {
  const labels = ["Just pitched", "2 days rest", "3 days rest", "4 days rest", "Fully fresh"];
  return (
    <div className="fatigue-wrap" title={labels[level]}>
      <div className="fatigue-balls">
        {[0, 1, 2, 3, 4].map((i) => (
          <BallIcon key={i} filled={i <= level} />
        ))}
      </div>
      <span className="fatigue-label">{labels[level]}</span>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// SUB-COMPONENTS
// ─────────────────────────────────────────────────────────────

function TeamPicker({ teams, value, onChange }) {
  return (
    <div className="team-picker">
      <select value={value ?? ""} onChange={(e) => onChange(Number(e.target.value))}>
        <option value="" disabled>Select your club</option>
        {teams.map((t) => (
          <option key={t.id} value={t.id}>{t.abbr} — {t.name}</option>
        ))}
      </select>
      <ChevronDown size={16} className="picker-chevron" />
    </div>
  );
}

function StatChip({ label, value, tone = "neutral" }) {
  return (
    <div className={`stat-chip tone-${tone}`}>
      <span className="stat-chip-value">{value}</span>
      <span className="stat-chip-label">{label}</span>
    </div>
  );
}

function AssignmentRow({ a, idx }) {
  const date = new Date(a.game_date + "T00:00:00");
  const dateStr = date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  const weekday = date.toLocaleDateString("en-US", { weekday: "short" });

  const regimeTone = {
    pennant_race: "race",
    clinched: "clinched",
    out_of_race: "neutral",
  }[a.division_regime] || "neutral";

  return (
    <div className="assignment-row" style={{ "--row-idx": idx }}>
      <div className="row-date">
        <span className="row-weekday">{weekday}</span>
        <span className="row-daynum">{dateStr}</span>
      </div>

      <div className="row-matchup">
        <span className="row-vs-label">vs</span>
        <span className="row-opponent">{a.opponent_name}</span>
      </div>

      <div className="row-pitcher">
        <span className="row-pitcher-name">{a.pitcher_name}</span>
        <FatigueIndicator level={a.fatigue_level} />
      </div>

      <div className="row-stats">
        <StatChip label="Matchup" value={a.matchup_score.toFixed(2)} tone="matchup" />
        <StatChip
          label="Division"
          value={a.division_bonus > 0 ? `+${a.division_bonus.toFixed(2)}` : "—"}
          tone={regimeTone}
        />
        <StatChip
          label="Fatigue pen."
          value={a.fatigue_penalty > 0 ? `−${a.fatigue_penalty.toFixed(2)}` : "—"}
          tone="penalty"
        />
      </div>

      <div className="row-total">
        <span className="row-total-value">{a.total_score.toFixed(2)}</span>
        <span className="row-total-label">SCORE</span>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// MAIN APP
// ─────────────────────────────────────────────────────────────

export default function RotationOptimizer() {
  const [teamId, setTeamId] = useState(null);
  const [horizonDays, setHorizonDays] = useState(14);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const runOptimizer = useCallback(async () => {
    if (!teamId) return;
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API_BASE}/games/optimizer/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          team_id: teamId,
          horizon_days: horizonDays,
          trigger: "manual",
        }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${res.status})`);
      }

      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError(e.message || "Something went wrong reaching the optimizer.");
      setResult(null);
    } finally {
      setLoading(false);
    }
  }, [teamId, horizonDays]);

  const selectedTeam = TEAMS.find((t) => t.id === teamId);

  return (
    <div className="ro-app">
      <style>{STYLES}</style>

      {/* ── Header / hero ────────────────────────────────── */}
      <header className="ro-hero">
        <div className="hero-eyebrow">ROTATION OPTIMIZER</div>
        <h1 className="hero-title">
          Who starts <span className="hero-accent">next</span>?
        </h1>
        <p className="hero-sub">
          A dynamic program weighing matchup quality, fatigue, and the
          standings.
        </p>
      </header>

      {/* ── Controls ──────────────────────────────────────── */}
      <section className="ro-controls">
        <TeamPicker teams={TEAMS} value={teamId} onChange={setTeamId} />

        <div className="horizon-control">
          <label htmlFor="horizon">Horizon</label>
          <input
            id="horizon"
            type="range"
            min={3}
            max={45}
            value={horizonDays}
            onChange={(e) => setHorizonDays(Number(e.target.value))}
          />
          <span className="horizon-value">{horizonDays}d</span>
        </div>

        <button
          className="run-btn"
          onClick={runOptimizer}
          disabled={!teamId || loading}
        >
          {loading ? (
            <>
              <Loader2 size={16} className="spin" /> Solving…
            </>
          ) : result ? (
            <>
              <RefreshCw size={16} /> Re-run
            </>
          ) : (
            <>
              <Play size={16} /> Run optimizer
            </>
          )}
        </button>
      </section>

      {/* ── Error state ───────────────────────────────────── */}
      {error && (
        <div className="ro-error">
          <AlertCircle size={18} />
          <div>
            <strong>Couldn't solve the rotation.</strong>
            <p>{error}</p>
          </div>
        </div>
      )}

      {/* ── Empty state ───────────────────────────────────── */}
      {!result && !error && !loading && (
        <div className="ro-empty">
          <div className="empty-card">
            <span className="empty-card-suit">⚾</span>
          </div>
          <p>
            {selectedTeam
              ? `Ready to build the ${selectedTeam.abbr} rotation. Hit run.`
              : "Pick a club above, then run the optimizer."}
          </p>
        </div>
      )}

      {/* ── Loading skeleton ──────────────────────────────── */}
      {loading && (
        <div className="ro-loading">
          {[0, 1, 2].map((i) => (
            <div key={i} className="skeleton-row" style={{ "--row-idx": i }} />
          ))}
        </div>
      )}

      {/* ── Results ───────────────────────────────────────── */}
      {result && !loading && (
        <section className="ro-results">
          <div className="results-summary">
            <div className="summary-block">
              <span className="summary-value">{result.assignments.length}</span>
              <span className="summary-label">games slated</span>
            </div>
            <div className="summary-divider" />
            <div className="summary-block">
              <span className="summary-value">{result.total_score.toFixed(2)}</span>
              <span className="summary-label">total score</span>
            </div>
            <div className="summary-divider" />
            <div className="summary-block">
              <span className="summary-value">
                {new Date(result.horizon_start + "T00:00:00").toLocaleDateString("en-US", { month: "short", day: "numeric" })}
                {" – "}
                {new Date(result.horizon_end + "T00:00:00").toLocaleDateString("en-US", { month: "short", day: "numeric" })}
              </span>
              <span className="summary-label">window</span>
            </div>
          </div>

          <div className="results-list">
            {result.assignments.map((a, idx) => (
              <AssignmentRow key={a.game_id} a={a} idx={idx} />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// STYLES
// ─────────────────────────────────────────────────────────────

const STYLES = `
@import url('https://fonts.googleapis.com/css2?family=Archivo+Black&family=Archivo:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

.ro-app {
  --navy: #0B2545;
  --cream: #F7F4EA;
  --red: #C8102E;
  --green: #2D6A4F;
  --gold: #D4A843;
  --ink: #1A1A1A;

  font-family: 'Archivo', sans-serif;
  background: var(--cream);
  background-image:
    radial-gradient(ellipse at top left, rgba(11,37,69,0.04), transparent 50%),
    repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(11,37,69,0.025) 40px);
  color: var(--ink);
  min-height: 100%;
  padding: 32px 24px 64px;
  border-radius: 12px;
}

.ro-app * { box-sizing: border-box; }

/* ── Hero ───────────────────────────────────────────── */

.ro-hero {
  max-width: 640px;
  margin: 0 auto 36px;
  text-align: center;
}

.hero-eyebrow {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.18em;
  color: var(--red);
  margin-bottom: 12px;
}

.hero-title {
  font-family: 'Archivo Black', sans-serif;
  font-size: clamp(32px, 6vw, 48px);
  line-height: 1.05;
  margin: 0 0 14px;
  color: var(--navy);
}

.hero-accent {
  color: var(--red);
  position: relative;
}

.hero-sub {
  font-size: 15px;
  line-height: 1.5;
  color: #4A5568;
  margin: 0;
}

/* ── Controls ───────────────────────────────────────── */

.ro-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
  align-items: center;
  justify-content: center;
  max-width: 720px;
  margin: 0 auto 28px;
  padding: 18px;
  background: white;
  border: 2px solid var(--navy);
  border-radius: 10px;
  box-shadow: 4px 4px 0 var(--navy);
}

.team-picker {
  position: relative;
  flex: 1;
  min-width: 220px;
}

.team-picker select {
  width: 100%;
  appearance: none;
  font-family: 'Archivo', sans-serif;
  font-weight: 600;
  font-size: 14px;
  padding: 11px 36px 11px 14px;
  border: 1.5px solid #D8D2C0;
  border-radius: 6px;
  background: var(--cream);
  color: var(--ink);
  cursor: pointer;
}

.team-picker select:focus-visible {
  outline: 2px solid var(--red);
  outline-offset: 1px;
}

.picker-chevron {
  position: absolute;
  right: 12px;
  top: 50%;
  transform: translateY(-50%);
  pointer-events: none;
  color: var(--navy);
}

.horizon-control {
  display: flex;
  align-items: center;
  gap: 8px;
}

.horizon-control label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.08em;
  color: var(--navy);
  text-transform: uppercase;
}

.horizon-control input[type="range"] {
  width: 110px;
  accent-color: var(--red);
}

.horizon-value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  font-weight: 600;
  color: var(--red);
  min-width: 28px;
}

.run-btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-family: 'Archivo', sans-serif;
  font-weight: 700;
  font-size: 14px;
  padding: 12px 22px;
  background: var(--red);
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  transition: transform 0.12s ease, box-shadow 0.12s ease;
  box-shadow: 3px 3px 0 var(--navy);
}

.run-btn:hover:not(:disabled) {
  transform: translate(-1px, -1px);
  box-shadow: 4px 4px 0 var(--navy);
}

.run-btn:active:not(:disabled) {
  transform: translate(1px, 1px);
  box-shadow: 2px 2px 0 var(--navy);
}

.run-btn:disabled {
  background: #C2BCA8;
  box-shadow: none;
  cursor: not-allowed;
}

.run-btn:focus-visible {
  outline: 2px solid var(--navy);
  outline-offset: 2px;
}

.spin { animation: spin 0.9s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

/* ── Error ──────────────────────────────────────────── */

.ro-error {
  max-width: 600px;
  margin: 0 auto 24px;
  display: flex;
  gap: 12px;
  padding: 16px 18px;
  background: #FDEEEE;
  border: 1.5px solid var(--red);
  border-radius: 8px;
  color: #8A0F22;
}

.ro-error strong { display: block; font-size: 14px; margin-bottom: 2px; }
.ro-error p { margin: 0; font-size: 13px; opacity: 0.85; }

/* ── Empty state ────────────────────────────────────── */

.ro-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
  padding: 56px 20px;
  text-align: center;
}

.empty-card {
  width: 72px;
  height: 72px;
  border: 2.5px dashed #C2BCA8;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 32px;
  opacity: 0.7;
}

.ro-empty p {
  font-size: 14px;
  color: #6B7280;
  max-width: 320px;
}

/* ── Loading skeleton ───────────────────────────────── */

.ro-loading {
  max-width: 880px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.skeleton-row {
  height: 76px;
  border-radius: 8px;
  background: linear-gradient(90deg, #EDE9DC 0%, #F7F4EA 50%, #EDE9DC 100%);
  background-size: 200% 100%;
  animation: shimmer 1.4s ease-in-out infinite;
  animation-delay: calc(var(--row-idx) * 0.08s);
}

@keyframes shimmer {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}

/* ── Results ────────────────────────────────────────── */

.ro-results {
  max-width: 880px;
  margin: 0 auto;
}

.results-summary {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 28px;
  margin-bottom: 24px;
  padding: 16px 24px;
  background: var(--navy);
  border-radius: 10px;
  color: var(--cream);
}

.summary-block {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
}

.summary-value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 20px;
  font-weight: 700;
}

.summary-label {
  font-size: 10px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  opacity: 0.65;
}

.summary-divider {
  width: 1px;
  height: 28px;
  background: rgba(247,244,234,0.2);
}

.results-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.assignment-row {
  display: grid;
  grid-template-columns: 64px 1fr 1.3fr auto auto;
  align-items: center;
  gap: 18px;
  padding: 14px 18px;
  background: white;
  border: 1.5px solid #E5DFCE;
  border-radius: 8px;
  animation: rowIn 0.35s ease backwards;
  animation-delay: calc(var(--row-idx) * 0.04s);
}

@keyframes rowIn {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}

.assignment-row:hover {
  border-color: var(--navy);
}

.row-date {
  display: flex;
  flex-direction: column;
  align-items: center;
  font-family: 'JetBrains Mono', monospace;
}

.row-weekday {
  font-size: 10px;
  letter-spacing: 0.06em;
  color: #9CA3AF;
  text-transform: uppercase;
}

.row-daynum {
  font-size: 15px;
  font-weight: 700;
  color: var(--navy);
}

.row-matchup {
  display: flex;
  flex-direction: column;
  gap: 1px;
}

.row-vs-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: #9CA3AF;
  letter-spacing: 0.05em;
}

.row-opponent {
  font-weight: 700;
  font-size: 14px;
  color: var(--ink);
}

.row-pitcher {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.row-pitcher-name {
  font-weight: 600;
  font-size: 14px;
}

.fatigue-wrap {
  display: flex;
  align-items: center;
  gap: 6px;
}

.fatigue-balls {
  display: flex;
  gap: 2px;
}

.fatigue-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: #9CA3AF;
}

.row-stats {
  display: flex;
  gap: 8px;
}

.stat-chip {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 5px 9px;
  border-radius: 5px;
  background: #F3F1E8;
  min-width: 58px;
}

.stat-chip-value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  font-weight: 700;
}

.stat-chip-label {
  font-size: 9px;
  letter-spacing: 0.04em;
  color: #9CA3AF;
  text-transform: uppercase;
  margin-top: 1px;
}

.tone-matchup .stat-chip-value { color: var(--navy); }
.tone-race { background: #FDEEEE; }
.tone-race .stat-chip-value { color: var(--red); }
.tone-clinched { background: #EAF4EE; }
.tone-clinched .stat-chip-value { color: var(--green); }
.tone-penalty .stat-chip-value { color: #B45309; }

.row-total {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding-left: 14px;
  border-left: 2px solid #E5DFCE;
}

.row-total-value {
  font-family: 'Archivo Black', sans-serif;
  font-size: 20px;
  color: var(--navy);
}

.row-total-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px;
  letter-spacing: 0.08em;
  color: #9CA3AF;
}

/* ── Responsive ─────────────────────────────────────── */

@media (max-width: 720px) {
  .assignment-row {
    grid-template-columns: 1fr;
    gap: 10px;
  }
  .row-date { flex-direction: row; gap: 6px; }
  .row-total { border-left: none; border-top: 1px solid #E5DFCE; padding: 10px 0 0; }
  .row-stats { flex-wrap: wrap; }
}
`;