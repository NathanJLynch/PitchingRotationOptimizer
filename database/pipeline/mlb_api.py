# database/pipeline/mlb_api.py
import requests
from datetime import date

BASE = "https://statsapi.mlb.com/api/v1"

def fetch_schedule(team_id: int, start_date: date, end_date: date) -> list:
    """Pull scheduled games for a team."""
    res = requests.get(f"{BASE}/schedule", params={
        "teamId":    team_id,
        "startDate": start_date.isoformat(),
        "endDate":   end_date.isoformat(),
        "sportId":   1,
        "hydrate":   "team,probablePitcher",
    })
    res.raise_for_status()
    games = []
    for date_entry in res.json().get("dates", []):
        for g in date_entry.get("games", []):
            games.append({
                "mlb_game_id":    g["gamePk"],
                "game_date":      date_entry["date"],
                "home_team_id":   g["teams"]["home"]["team"]["id"],
                "away_team_id":   g["teams"]["away"]["team"]["id"],
                "home_team_name": g["teams"]["home"]["team"]["name"],
                "away_team_name": g["teams"]["away"]["team"]["name"],
                "probable_home":  g["teams"]["home"].get("probablePitcher", {}).get("id"),
                "probable_away":  g["teams"]["away"].get("probablePitcher", {}).get("id"),
            })
    return games

def fetch_standings(league_id: int = 104) -> list:
    """Pull NL standings. 103=AL, 104=NL."""
    res = requests.get(f"{BASE}/standings", params={
        "leagueId": league_id,
        "season":   date.today().year,
        "hydrate":  "team",
    })
    res.raise_for_status()
    rows = []
    for record in res.json().get("records", []):
        division_name = record["division"]["nameShort"]
        for team_rec in record["teamRecords"]:
            rows.append({
                "mlb_team_id":     team_rec["team"]["id"],
                "team_name":       team_rec["team"]["name"],
                "division":        division_name,
                "wins":            team_rec["wins"],
                "losses":          team_rec["losses"],
                "games_behind":    float(team_rec["gamesBack"].replace("-", "0")),
                "win_pct":         float(team_rec["winningPercentage"]),
                "run_differential": team_rec.get("runDifferential", 0),
                "last_10":         team_rec.get("records", {})
                                           .get("splitRecords", [{}])[0]
                                           .get("wins", 0),
            })
    return rows

def fetch_pitcher_game_log(mlb_player_id: int, season: int) -> list:
    """Pull game-by-game pitching log for a pitcher."""
    res = requests.get(f"{BASE}/people/{mlb_player_id}/stats", params={
        "stats":  "gameLog",
        "group":  "pitching",
        "season": season,
    })
    res.raise_for_status()
    splits = res.json().get("stats", [{}])[0].get("splits", [])
    return [{
        "game_date":      s["date"],
        "opponent_id":    s["opponent"]["id"],
        "pitch_count":    s["stat"].get("numberOfPitches", 0),
        "innings_pitched": float(s["stat"].get("inningsPitched", 0)),
        "strikeouts":     s["stat"].get("strikeOuts", 0),
        "walks":          s["stat"].get("baseOnBalls", 0),
        "hits":           s["stat"].get("hits", 0),
        "earned_runs":    s["stat"].get("earnedRuns", 0),
    } for s in splits]