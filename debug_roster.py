import requests

res = requests.get(
    "https://statsapi.mlb.com/api/v1/teams/158/roster",
    params={
        "rosterType": "active",
        "season": 2026,
        "hydrate": "person(stats(type=season,group=pitching))"
    },
    timeout=15
)

roster = res.json().get("roster", [])
pitchers = [e for e in roster if e.get("position", {}).get("abbreviation") == "P"]
print(f"Total pitchers on roster: {len(pitchers)}")
print()

for p in pitchers[:3]:
    person = p["person"]
    stats_list = person.get("stats", [])
    print(f"Name: {person['fullName']}")
    print(f"Stats entries: {len(stats_list)}")

    if stats_list:
        first = stats_list[0]
        print(f"Stat type displayName: {first.get('type', {}).get('displayName')}")
        splits = first.get("splits", [])
        print(f"Splits count: {len(splits)}")
        if splits:
            stat = splits[0].get("stat", {})
            print(f"Stat keys: {list(stat.keys())[:10]}")
            print(f"gamesStarted: {stat.get('gamesStarted')}")
            print(f"inningsPitched: {stat.get('inningsPitched')}")
    print()