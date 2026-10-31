"""Explain supported counter exposure without presenting it as loss probability."""

from html import escape
from PyQt6.QtWidgets import QDialog, QPushButton, QTextBrowser, QVBoxLayout


class BlindPickDialog(QDialog):
    def __init__(self, record, names, roles, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Blindpick · {names.get(record['id'], record['id'])}")
        self.resize(760, 630)
        layout = QVBoxLayout(self)
        self.details = QTextBrowser()
        layout.addWidget(self.details)
        info = record["blind"]
        html = ("<p>A jelzés a jelenlegi pozíciókiosztást követi. Az alap győzelmi arányokkal és mintazajjal korrigált "
                f"párosításokból a legalább {info['threshold_pp']:.0f} százalékpontos, szereprelevanciával súlyozott hátrányokat tekinti counternek. "
                "Csak a nem off-meta hős–pozíciók számítanak.</p>")
        if info.get("baseline"):
            base = info["baseline"]
            html += f"<p>A hős pozíciós alaparánya: {base['winrate']:.1f}% · {base['matches']} megfigyelés.</p>"
        if info["risk_percent"] is not None:
            html += f"<h3>Hátralévő, ismert counterek kitettsége: ~{info['risk_percent']:.1f}%</h3>"
        if info.get("coverage_percent") is not None:
            html += f"<p>Átlagos pickrate-lefedettség: {info['coverage_percent']:.1f}%. "
            html += f"Az ismeretlen párosításokkal együtt lehetséges tartomány: {info['risk_percent']:.1f}–{info['upper_percent']:.1f}% (nem konfidenciaintervallum).</p>"
        if info["status"] in ("unavailable", "off_meta"):
            html += "<p>Ehhez a hőshöz/pozícióhoz nincs használható új interakciós statisztika. Az új adatfájl a tanítás előtti feldolgozásban készül.</p>"
        elif info["status"] == "insufficient":
            html += "<p>⚠ Legalább egy hátralévő szerepkör pickrate-tömegének kevesebb mint felére van adat; a kis szám nem jelent biztonságos blindot.</p>"
        elif info["status"] == "resolved":
            html += "<p>Minden ellenfél pozíciója ki van töltve; további ismeretlen választásra nincs kitettség.</p>"
        if info["known_counters"]:
            html += "<h3>Már látható counterek</h3><ul>"
            for item in info["known_counters"]:
                html += (f"<li>{escape(names.get(item['id'], str(item['id'])))} · {escape(roles[item['role']])}: "
                         f"{item['weighted_delta_pp']:+.1f} pp súlyozott, zajkorrigált hatás · {item['matches']} párosítás.</li>")
            html += "</ul>"
        for item in info["role_risks"]:
            html += f"<h3>{escape(roles[item['role']])}: ~{item['risk_percent']:.1f}% · lefedettség {item['coverage_percent']:.1f}%</h3><ul>"
            for counter in item["counters"][:10]:
                html += (f"<li>{escape(names.get(counter['id'], str(counter['id'])))}: {counter['pick_rate_percent']:.1f}% maradék pickrate · "
                         f"{counter['weighted_delta_pp']:+.1f} pp súlyozott hatás · {counter['matches']} párosítás.</li>")
            html += "</ul>"
        html += ("<p>Ez gyakoriságokból számolt közelítés, nem az ellenfél döntésének vagy a vereségnek a valószínűsége. "
                 "A hátralévő pozíciók kombinálása függetlenséget közelít; a flex pickek és a játékos hőspoolja ettől eltérhet. "
                 "A már választott és kitiltott hősök kimaradnak a maradék poolból. Az ismeretlen párosításokat nem tekintjük igazoltan biztonságosnak.</p>")
        self.details.setHtml(html)
        close = QPushButton("Bezárás")
        close.clicked.connect(self.accept)
        layout.addWidget(close)
