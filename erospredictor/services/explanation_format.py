"""Presentation of model diagnostics; no inference or Qt dependency."""

from html import escape


ROLES = ("Top", "Jungle", "Mid", "ADC", "Support")
MODEL_NAMES = {
    "gnn": "Gráfmodell (GNN)",
    "roleweighted": "Szerepkörrel súlyozott modell",
    "roleaware": "Szerepkört ismerő modell",
    "statistical": "Mérkőzésstatisztika",
    "fallback": "Semleges becslés (hiányzó modell)",
}


def explanations_html(records, champion_names):
    """Format the explicit second-stage result using escaped champion names."""
    def name(champion_id):
        return escape(champion_names.get(champion_id, f"Hős #{champion_id}"))

    selected = next((record for record in records if record.get("is_selected")), None)
    if selected is not None:
        visible = [selected] + [record for record in records if record["id"] != selected["id"]][:5]
    else:
        visible = records[:5]
    title = "A kijelölt hős és az első öt ajánlás" if selected is not None else "Az első öt ajánlás részletesen"
    parts = [
        f"<h2>{title}</h2>",
        "<p>A változások <b>százalékpontban</b> értendők. Az üres pozícióhoz "
        "viszonyított becslés hiányos draft szimulációja. Az egyes társak és "
        "ellenfelek hatása külön vizsgálatból származik; ezek nem adhatók össze, "
        "és nem bizonyítanak okozati kapcsolatot.</p>",
    ]
    comparisons_started = False
    for position, record in enumerate(visible, 1):
        if record is selected:
            parts.append("<hr><h2>Kijelölt hős</h2>")
            rank = record.get("recommendation_rank")
            parts.append(f"<p>Az ajánlási lista {rank}. helyezettje.</p>" if rank is not None
                         else "<p>A kijelölt hős nincs az első öt ajánlás között.</p>")
            heading = name(record['id'])
        else:
            if selected is not None and not comparisons_started:
                group = ("Az első öt ajánlás többi hőse" if selected.get("recommendation_rank") is not None
                         else "Az első öt ajánlás")
                parts.append(f"<hr><h2>{group}</h2>")
                comparisons_started = True
            rank = record.get("recommendation_rank") or (position - 1 if selected is not None else position)
            heading = f"{rank}. {name(record['id'])}"
        parts.append(
            f"<hr><h2>{heading} — {record['wr']:.2f}%</h2>"
            f"<p>Ugyanez a draft üres ajánlási pozícióval: "
            f"{record['baseline_win_prob']:.2f}%. A hős becsült többlete: "
            f"<b>{record['candidate_delta_pp']:+.2f} százalékpont</b>.</p>"
        )
        parts.append("<h3>A modellek eredménye és tényleges súlya</h3>"
                     "<table cellpadding='6'><tr><th>Modell</th>"
                     "<th>Győzelmi esély</th><th>Súly</th></tr>")
        for component in record.get("components", []):
            label = escape(MODEL_NAMES.get(component["model"], component["model"]))
            parts.append(f"<tr><td>{label}</td><td>{component['win_prob']:.2f}%</td>"
                         f"<td>{component['weight'] * 100:.1f}%</td></tr>")
        parts.append("</table>")
        unavailable = record.get("unavailable_models", {})
        if unavailable:
            labels = ", ".join(escape(MODEL_NAMES.get(key, key)) for key in unavailable)
            parts.append(f"<p><b>Nem elérhető vagy hibát jelzett:</b> {labels}. "
                         "Az eredmény az elérhető modellek újrasúlyozott becslése.</p>")
        interactions = record.get("interactions", [])
        if interactions:
            parts.append("<h3>Melyik ismert hőshöz kapcsolódik az ajánlás?</h3>"
                         "<p>A szám azt mutatja, mennyivel változik az ajánlott hős "
                         "többlete, ha az adott társat vagy ellenfelet ismerjük. "
                         "A pozitív érték kedvez az ajánlásnak.</p>"
                         "<table cellpadding='6'><tr><th>Hős</th><th>Viszony</th>"
                         "<th>Pozíció</th><th>Hatás</th></tr>")
            for item in sorted(interactions, key=lambda item: abs(item["delta_pp"]), reverse=True):
                relation = "Csapattárs" if item["side"] == "ally" else "Ellenfél"
                if item.get("is_lane_opponent"):
                    relation += " (azonos pozíció)"
                role_index = item.get("role_index", -1)
                role = ROLES[role_index] if 0 <= role_index < len(ROLES) else "?"
                parts.append(f"<tr><td>{name(item['champion_id'])}</td><td>{relation}</td>"
                             f"<td>{role}</td><td>{item['delta_pp']:+.2f} szp.</td></tr>")
            parts.append("</table>")
        else:
            parts.append("<p>Még nincs másik ismert hős az összehasonlításhoz.</p>")
        for key, title in (("lane_matchup", "Történeti lane-párosítás"),
                           ("bot_synergy", "Történeti ADC–Support együttműködés")):
            stats = record.get(key)
            parts.append(f"<h3>{title}</h3>")
            if stats:
                parts.append(f"<p>Győzelmi arány: <b>{stats['winrate']:.2f}%</b>, "
                             f"minta: <b>{stats['matches']:g} mérkőzés</b>.")
                weighted = stats.get("weighted_winrate")
                if weighted is not None:
                    parts.append(f" Rangsúllyal számítva: {weighted:.2f}%.")
                parts.append(" Ez teljes mérkőzések eredménye, nem a lane megnyerésének esélye. "
                             "Kis mintánál a statisztika bizonytalan.</p>")
            else:
                parts.append("<p>Nincs elérhető minta ehhez a párosításhoz.</p>")
    if not records:
        parts.append("<p>Nincs elemezhető ajánlás.</p>")
    return "".join(parts)
