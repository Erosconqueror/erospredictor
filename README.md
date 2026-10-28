# erospredictor
League of Legends hősválasztási segédprogram Python/PyQt6 felülettel, három neurális
modellel és mérkőzésstatisztikával.

Az új funkciók fejlesztési ága: **`overfeatures`**. A szakdolgozat védéséig a többi
branch változatlan marad.

## Telepítés és indítás

Windows alatt először `install.bat`, majd `start_app_noupdate.bat`.
Az utóbbi megtartja a helyben tanított modelleket. A `start_app.bat` a korábbi,
publikus modellletöltőt is elindítja; az nem az alábbi tanítási folyamat.
Sikeres helyi tanítás után a létrejövő manifest miatt a publikus letöltő kihagyja
a modellcserét, így a következő indítás megőrzi a helyi eredményt.

## Egy indítás: adatgyűjtés, tanítás és kalibrálás

A projekt virtuális környezete, PostgreSQL adatbázis és érvényes Riot API-kulcs szükséges.
A kapcsolódási adatokat a `configs.py` vagy az alábbi környezeti változók adják meg:
`RIOT_API_KEY`, `DB_NAME`, `DB_USER`, `DB_PASS`, `DB_HOST`, `DB_PORT`.
Az alkalmazást a modellcsere idejére zárd be.

```powershell
# Alapértelmezés: az utolsó öt megjelent patch, minden rang + MIXED.
.\update_models.bat

# Három legutóbbi patch.
.\update_models.bat --last-patches 3

# Konkrét patchablak; a 26.x elnevezés 16.x belső verzióként is megadható.
.\update_models.bat --patches 16.5 16.6 16.7

# Korábbi patchnél végződő ablak.
.\update_models.bat --end-patch 16.7 --last-patches 3

# Terv ellenőrzése adatbázis és tanítás nélkül.
.\update_models.bat --patches 16.5 16.6 16.7 --dry-run

# Meglévő adatbázisból, új scrape nélkül.
.\update_models.bat --last-patches 3 --skip-scrape
```

Az automatikus patchlista a Riot Data Dragon megjelent verzióiból készül, így
évfordulón sem egyszerű sorszámcsökkentést használ. Explicit `--patches` mellett
a `--dry-run` hálózati kapcsolat nélkül is működik.

Az adatgyűjtés véges: alapértelmezetten rangonként legfeljebb 200 játékost vizsgál,
és legfeljebb 2000 új, megfelelő patchű meccset ment. Az ismert meccseket átugorja.
A `--matches-per-tier`, `--players-per-tier`, `--pages-per-division`,
`--matches-per-player`, `--region` és `--continent` kapcsolókkal állítható.
További beállítások: `env\Scripts\python.exe erospredictor\update_pipeline.py --help`.

A kiválasztott patchekre szűrt adatokból készül a statisztika, az off-meta lista,
a rangonkénti GNN, Roleweighted és Roleaware modell, majd a Golden Dataset
ellenőrzése és az ensemble kalibrációja. A modellek külön ideiglenes mappában
készülnek; csak a teljes kért folyamat sikere után kerülnek a helyükre.
Hiba vagy elégtelen adatmennyiség esetén a korábbi modellek megmaradnak,
a már begyűjtött meccsek pedig felhasználhatók a következő indításkor.
Az eredmény és a kihagyott adatok száma a `data/update_manifest.json` fájlba kerül.
Egyszerre csak egy modellfrissítés futhat.

A Riot-kérések közös ütemezést kapnak: alapból legfeljebb 20 kérés/másodperc és
100 kérés/120 másodperc egy kulcs és route alatt. A szerver által közölt alkalmazás-
és metóduskereteket, valamint számlálókat is figyeli. A magasabb metóduslimit nem
növeli meg az alkalmazás teljes keretét. A 429-es válasznál kivárja a teljes
`Retry-After` időt, majd ugyanazt a kérést legfeljebb ötször újrapróbálja;
hosszabb várakozáskor kiírja a hátralévő időt. Így egy nagyobb adatgyűjtésnél
a várakozás normális. Más HTTP-hibákat nem ismétel korlátlanul.
A [Riot limitkezelési leírása](https://developer.riotgames.com/docs/portal#web-apis_rate-limiting)
részletezi az egymás mellett érvényes kereteket.

Az aktuális, rögzített hőskatalógus 172 hőst tartalmaz. A benne még nem szereplő
hősöket tartalmazó meccseket a tanítás jelzett módon kihagyja; ez a folyamat
nem számozza át a meglévő hősazonosítókat és nem bővíti automatikusan a katalógust.
A jelenlegi GUI az új, ismeretlen hősök beolvasásánál is figyelmeztet.

## Hősök beolvasása a saját LoL-kliensből

Nyisd meg a LoL-kliens hősválasztását, majd az Erospredictorban kattints a
**LoL-kliens beolvasása** gombra. A már rögzített választások és a tiltások
a helyi kliensből érkeznek; nincs OCR, képernyőkép-feltöltés vagy folyamatos képfeldolgozás.
Ez a gomb csak olvas, nem választ és nem tilt hőst a kliensben.

A megjelenő ellenőrző ablakban rendeld pozícióhoz azokat a hősöket, amelyeknél
a kliens nem közli a szerepkört. Az ellenfél listájának sorrendjét a program
nem tekinti lane-sorrendnek. Ha az oldal sem biztos, válaszd ki a saját csapatod
Kék/Piros oldalát is. A **Draft kitöltése** gomb az ellenőrzött drafttal felváltja
a főablak korábbi választásait és tiltásait. A még nem véglegesített hover kimarad.

Ha az automatikus keresés nem találja a klienst, a **LoL lockfile…** gombbal
válaszd ki a League of Legends telepítési mappájának `lockfile` fájlját.
Példa: `C:\Riot Games\League of Legends\lockfile` vagy egy másik meghajtón a
megfelelő telepítési mappa. A fájl a futó klienshez tartozik; nem a kliens `.exe`
fájlját vagy a Riot API-kulcsot kell megadni. Nyisd meg a klienst és a hősválasztást.
Alternatív beállítás: `EROSPREDICTOR_LCU_LOCKFILE` környezeti változó.
A helyi hitelesítési token nem jelenik meg a hibaüzenetekben.

Az `install.bat` telepítéskor automatikusan megkeresi a kliens helyét, és elmenti
az adott gépen. Bezárt kliensnél a telepítési mappa `LeagueClient.exe` fájlja
alapján is felismeri a helyet. Ha a LoL még nincs telepítve, az alapértelmezés
`C:\Riot Games\League of Legends\lockfile`; a program később újra keres, ha ez
nem létezik. Az első sikeresen olvasott lockfile és a kézzel kiválasztott hely
is megmarad a következő alkalmazásindításokra.

Az útvonal a `%LOCALAPPDATA%\Erospredictor\settings.json` helyi beállításába kerül,
így más felhasználó gépéhez nem viszi át. Csak a helyét jegyzi meg: a lockfile
változó jelszavát és portját minden beolvasáskor újra kiolvassa. A fájlválasztó
is a megjegyzett vagy alapértelmezett útvonalról indul.

A [League Client API](https://developer.riotgames.com/docs/lol#league-client-api)
Riot által nem támogatott helyi interfész, ezért kliensfrissítés után élő ellenőrzést
igényelhet. Közvetítés, képernyőkép és profi draft felismerése egy következő fejlesztés.

## Opcionális magyarázat az első öt ajánláshoz

Először kérj normál **AJÁNLÁS**-t. A **Miért? – Az első 5 részletes elemzése**
gomb külön, második lépésként számol; kikapcsolt állapotban nincs extra inferencia.
A felület a hosszabb számítások alatt is reagál.

A részletek megmutatják a modellek győzelmiesély-becslését és tényleges súlyát,
az ajánlott hős becsült többletét az üres pozícióhoz képest, valamint azt,
hogyan változik ez a többlet egy-egy ismert társ vagy ellenfél elhagyásakor.
Ez feltételes modellérzékenység, nem összeadható vagy okozati bizonyíték.
A lane-párosítás és ADC–Support szinergia történeti győzelmi aránya külön,
a mérkőzésszámmal együtt látható. A lane-statisztika teljes meccsek kimenetelét
méri, nem az ösvény megnyerésének esélyét. Hiányzó mintát nem pótol kitalált értékkel.
Draft-, rang- vagy szűrőváltás után előbb új ajánlást kell kérni.

## Egyszerű nyilas hőshatások a főablakon

A felső **Egyszerű hőshatások** kapcsoló alapból ki van kapcsolva, ilyenkor
nincs hozzáadott hatásszámítás. Bekapcsolva a hősök és szerepköreik mellett
zöld **▲**, piros **▼**, illetve semleges **→** jelzi a hatást. Az erősebb
szín nagyobb eltérést jelent, az **arany keret** pedig a fókuszhős lane-ellenfelét
emeli ki. A **● Fókusz** felirat jelöli a kiválasztott pozíciót.

Predikcióhoz válaszd ki lent a csapatot és szerepkört, töltsd ki ott a vizsgált
hőst, majd nyomd meg a **PREDIKCIÓ** gombot. Az aktuális győzelmi becslés mellett
megjelennek az adott hőshöz viszonyított hatások. Üres fókuszpozíció mellett
a normál predikció továbbra is lefut, csak a nyilas elemzéshez kér hősválasztást.

Ajánlásnál először az első jelölt a fókusz. A listában bármelyik ajánlott hős
nevére kattintva átválthatsz rá, anélkül, hogy a draftba beírná. A kijelölt ajánlás
színes hátteret kap. Ez egyetlen hőst vizsgál, nem indítja el az ötös részletes
elemzést. Teljes draftnál legfeljebb 20 gyors, kétoldali predikciót számol.

A szám **százalékpontban** mutatja, hogyan változik a fókuszhős becsült többlete
egy másik ismert hős jelenlétében. Pozitív: kedvező a kiválasztott csapat és hős
számára; negatív: kedvezőtlen. Ez modellérzékenység, nem biztos lane-győzelem vagy
összeadható magyarázat. Draft-, rang-, csapat- vagy szerepkörváltás törli a korábbi
nyilakat, így elavult eredmény nem marad a hősök mellett.

## Ellenőrzések

```powershell
Push-Location erospredictor
..\env\Scripts\python.exe -m unittest discover -v
Pop-Location
```

Az automatizált tesztek helyi fixture-ökkel és mockokkal működnek; nem indítanak
éles adatgyűjtést, adatbázisírást vagy teljes modell-újratanítást.
