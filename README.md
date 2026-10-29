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

# Az eddigi folytatási pont helyett tudatosan új futás indítása.
.\update_models.bat --last-patches 3 --fresh
```

Külön, csak adatgyűjtésre szolgáló mód:

```powershell
.\update_models.bat --scrape-only

# A terv ellenőrzése konkrét patchablakkal, hálózat és adatbázis nélkül.
.\update_models.bat --scrape-only --patches 16.19 16.20 16.21 --dry-run
```

A `--scrape-only` a jóváhagyott rangprofil számait **arányként** használja,
és nagyjából ezer meccses körökben gyűjt tovább, rögzített teljes meccslimit nélkül.
A már meglévő adatokat is figyelembe veszi, először az alulreprezentált rangokat
tölti fel. Ha egy rang elérhető játékosai elfogynak, a többinél tovább keres;
emiatt az elérhető adatoktól függően az arány eltérhet a céltól. Az összes
elérhető játékos feldolgozása vagy kézi keresési korlát elérése után befejeződik.
Játékosonként alapból a legutóbbi 50 meccset kéri le; `--matches-per-player 100`
a kérésenként megengedett nagyobb előzménylistát használja.

Ebben a módban nincs tanítás, QA, kalibráció vagy modellcsere. A friss hőslistát
csak a gyűjtéshez használja, a telepített katalógust, ikonokat és konfigurációt
nem publikálja. A normál frissítő később fel tudja használni az elmentett meccseket.
A `--skip-scrape` és `--scrape-only` egymást kizárja; az arányt a `--tier-target`
kapcsolóval lehet módosítani, a `--matches-per-tier` ehhez a módhoz nem használható.

A gyűjtés külön `.scrape_only/state.json` folytatási pontot használ, így a normál
frissítés esetleges tanítási mentése megmarad. **Ctrl+C** után ugyanazzal a
paranccsal folytatható; kapcsolatbontásnál automatikusan újrapróbálkozik. Egyszerre
a két mód közül csak az egyik futhat. A befejezett gyűjtés összesítése a
`.scrape_only/last_collection.json` fájlba kerül; a következő kézi indítás új
gyűjtési menetet indít. A `--fresh` ebben a módban csak a külön gyűjtési mentését
kezdi újra, az adatbázisban tárolt meccsek megmaradnak.

Az automatikus patchlista a Riot Data Dragon megjelent verzióiból készül, így
évfordulón sem egyszerű sorszámcsökkentést használ. Explicit `--patches` mellett
a `--dry-run` hálózati kapcsolat nélkül is működik.

Az alapértelmezett, közelítő játékoseloszlást követő profil az utolsó öt patchből
összesen **107 000 használható, egyedi meccset** céloz. A táblázat rangonkénti,
az öt patchre együtt értendő célokat tartalmaz; Gold és Silver külön-külön is
legalább 20 000 meccset kap. Ez szándékosan választott mintavételi profil.

| Rang | Meccscél |
| --- | ---: |
| Challenger | 2 000 |
| Grandmaster | 3 000 |
| Master | 5 000 |
| Diamond | 8 000 |
| Emerald | 12 000 |
| Platinum | 16 000 |
| Gold | 22 000 |
| Silver | 20 000 |
| Bronze | 14 000 |
| Iron | 5 000 |

A már adatbázisban lévő, megfelelő régiójú, rangú és patchű, érvényes meccsek
beszámítanak: a frissítő a hiányzó mennyiséget gyűjti. A célig minden elérhető
oldalon továbbkeres, alapból nincs 200 játékosos vagy kétoldalas korlát. Ha minden
játékost végignézett, de még kevés a meccs, várakozás után újra keres. A kvóták
teljesüléséig nem indít kisebb mintával tanítást. A nagyobb korábbi rangadatokból
csak a profil szerinti mennyiséget használja; a rendelkezésre álló patchek között
közel egyenletesen mintáz, majd a meglévő frissességi súlyokat alkalmazza.

A profil a `configs.py` `DATASET_TARGETS` beállításában szerepel. Egyedi cél például
`--tier-target GOLD=24000`; ismételhető több ranghoz. A külön megadott
`--matches-per-tier 2000` a régebbi, egységes újmeccs-kvótás működést választja.
A `--players-per-tier`, `--pages-per-division`, `--matches-per-player`, `--region`
és `--continent` kapcsolók is használhatók. Kézi keresési korlát vagy `--skip-scrape`
mellett a hiányos profil beavatkozást kér, és megtartja az adatokat.
További beállítások: `env\Scripts\python.exe erospredictor\update_pipeline.py --help`.

A kiválasztott patchekre szűrt adatokból készül a statisztika, az off-meta lista,
a rangonkénti GNN, Roleweighted és Roleaware modell, majd a Golden Dataset
ellenőrzése és az ensemble kalibrációja. A modellek külön ideiglenes mappában
készülnek; csak a teljes kért folyamat sikere után kerülnek a helyükre.
Hiba vagy elégtelen adatmennyiség esetén a korábbi modellek megmaradnak,
a már begyűjtött meccsek pedig felhasználhatók a következő indításkor.
Az eredmény és a kihagyott adatok száma a `data/update_manifest.json` fájlba kerül.
Egyszerre csak egy modellfrissítés futhat.

A frissítő automatikusan folytatja a félbemaradt futást. Az első, javított verziós
indítás a korábban már letöltött, a kiválasztott patchekhez és rangokhoz tartozó,
használható meccseket is beszámítja a gyűjtési kvótába. Ha például a CHALLENGER
2000 megfelelő meccse már megvan, továbblép a következő rangra.

A `.model_update/state.json` tartósan megjegyzi a patchablakot, a kész rangokat,
az aktuális oldalt, játékost és meccset. A meccs adatbázisba mentését újraindítás
után is biztonságosan lezárja, dupla beszúrás és dupla kvótaszámolás nélkül.
Az adatgyűjtés után rögzített tanítóadat-pillanatképet használ, így a már kész
modellek és a folytatott tanítás ugyanabból az adatból dolgoznak.

A tanítás minden lezárt korszak után atomikusan menti a modellt, az optimalizálót,
a legjobb súlyokat és a véletlengenerátor állapotát. Újraindítás után az utolsó
lezárt korszaktól folytatja; a félbeszakadt korszakot újrafuttatja. A teljesen kész,
QA-ellenőrzött modellek ellenőrzőösszegét is megvizsgálja, és ezeket nem tanítja újra.
A végső modellcsere megszakadása után a megőrzött forrásokból fejezi be a cserét.

Váratlan Python-folyamatleállásnál az indító tíz másodperc múlva újraindítja a
frissítést a mentett ponttal. Kapcsolati vagy adatbázis-hibánál a futó program
várakozás után magától újrapróbálkozik. **Ctrl+C** vagy az ablak bezárása leállítja;
felhasználói megszakítás után nem indítja újra magát. A következő kézi indítás
ugyanazokkal a paraméterekkel ismét folytatható. A `--fresh` szándékosan új futást
kezd, a már adatbázisba gyűjtött meccseket megtartva.

Hibás kulcs, hiányzó függőség, kézzel korlátozott elégtelen adat vagy sikertelen QA esetén
beavatkozást kér, és megtartja a folytatási pontot. Megváltoztatott futási/tanítási
paraméterekhez új futás szükséges, hogy ne keveredjenek eltérő adatok és modellek.
A gyűjtési profil, keresési korlátok vagy katalógusfrissítés megváltoztatása kivétel:
a korábbi patchablakot és gyűjtési kurzort átveszi, a kibővült rangcélokat újranyitja,
a tanításhoz pedig új, rögzített pillanatképet készít. A korábbi futás adatbázisba
mentett meccsei megmaradnak.

A Riot-kérések közös ütemezést kapnak: alapból legfeljebb 20 kérés/másodperc és
100 kérés/120 másodperc egy kulcs és route alatt. A szerver által közölt alkalmazás-
és metóduskereteket, valamint számlálókat is figyeli. A magasabb metóduslimit nem
növeli meg az alkalmazás teljes keretét. A 429-es válasznál kivárja a teljes
`Retry-After` időt, majd ugyanazt a kérést újrapróbálja. Az egyindításos frissítő
kapcsolatbontás, timeout és átmeneti szerverhiba (pl. 502/503/504) esetén is
folyamatosan próbálkozik, 5 másodpercről legfeljebb 120 másodpercre növekvő szünettel;
hosszabb várakozáskor kiírja a hátralévő időt. Így egy nagyobb adatgyűjtésnél
a várakozás normális. A már nem elérhető meccset/játékost kihagyja, a hibás
hitelesítést nem próbálja hálózati hibaként újra és újra elküldeni.
A [Riot limitkezelési leírása](https://developer.riotgames.com/docs/portal#web-apis_rate-limiting)
részletezi az egymás mellett érvényes kereteket.

Az updater a patchablak legfrissebb patchéhez tartozó hivatalos
[Data Dragon hőslistát](https://developer.riotgames.com/docs/lol#data-dragon) is
frissíti. A meglévő belső azonosítók megmaradnak, az új hősök a lista végére
kerülnek; az Aatrox=0 index változatlan. A hiányzó portrékat is letölti.
A bővített katalógus, a hozzá illő méretű modellek és az új konfiguráció csak
a teljes tanítás, QA és kalibráció után kerülnek a helyükre, ugyanazzal a
helyreállítható cserefolyamattal. Új hős esetén minden rangot együtt kell frissíteni.
A `--no-catalog` megtartja a telepített listát, például korlátozott vagy helyi
újratanításhoz. A választott katalógusban ismeretlen hősök meccsei kimaradnak.

Sikeres frissítéskor a `data/runtime_config.json` tárolja a használt patchablakot,
a hőskatalógus verzióját és méretét, valamint az adatprofilt. Ezt a `configs.py`
a következő alkalmazásindításkor betölti; a tényleges hősszámot a publikált
hőslistából olvassa. Az API-kulcsot és adatbázis-beállításokat nem írja át.

## Hősök beolvasása a saját LoL-kliensből

Nyisd meg a LoL-kliens hősválasztását, majd az Erospredictorban kattints a
**LoL-kliens beolvasása** gombra. A már rögzített választások és a tiltások
a helyi kliensből érkeznek; nincs OCR, képernyőkép-feltöltés vagy folyamatos képfeldolgozás.
Ez a gomb csak olvas, nem választ és nem tilt hőst a kliensben.

A megjelenő ellenőrző ablakban az elérhető szerepkör-statisztika alapján előre
kitölti a javasolt pozíciókat; ezeket ellenőrizheted és javíthatod. Az ellenfél
listájának sorrendjét a program nem tekinti lane-sorrendnek. Ha az oldal sem biztos, válaszd ki a saját csapatod
Kék/Piros oldalát is. A **Draft kitöltése** gomb az ellenőrzött drafttal felváltja
a főablak korábbi választásait és tiltásait. A még nem véglegesített hover kimarad.

Az **Automatikus draft** kapcsoló alapból ki van kapcsolva. Bekapcsolva 1,5
másodpercenként, háttérben olvassa a helyi kliens rögzített pickjeit és banjait.
Változáskor frissíti a mezőket, hőscsere esetén is. Azonos állapotnál megtartja
a már elkészült ajánlást/predikciót; draftváltozás után a korábbi eredményt törli.
Az automatikus adatbetöltés nem indít ajánlási vagy magyarázó modellfuttatást.
Kikapcsolás után egy későn beérkező válasz sem írhatja át a mezőket.

Normal Draft (400), Ranked Solo/Duo (420), Ranked Flex (440) és 5v5 custom
ugyanazt a beolvasási utat használja. Kezeli a kliens explicit `team: 1/2`
oldalkódját és a platform `teamId: 100/200` formátumát. A már befejezett pick
akcióból akkor is kiolvassa a hőst, ha a névsor sora még nem frissült.
Customban csak a ténylegesen közölt hősökkel dolgozik; a hiányzó botokat nem
találja ki. A státusz a látható és a besorolt hősök számát külön mutatja.

A saját csapat kliens által közölt pozíciói elsőbbséget kapnak. Az ellenfélnél
az összes kiválasztott hőst együtt vizsgálja: legfeljebb 120 lehetséges kiosztás
közül keresi a szerepgyakoriságok szorzata alapján legjobb, ütközésmentes
felállást. Automatikus szerepkörhöz a hős saját megfigyeléseinek legalább **5%-a**
kell azon a pozíción. Részleges ellenfélcsapatnál ez alatti elhelyezést nem fogad
el az ütközés feloldásához. Csak mind az öt látható ellenfélnél engedi el a küszöböt,
ha nincs teljes megengedett kiosztás; ezt **⚠ kényszer** jelöléssel és magyarázattal
mutatja. Saját klienspozíció és kézi megerősítés továbbra is elsőbbséget kap.
Így például Braum Support mellett Xerath kerülhet Midre. A pozíció
alatti `~61%` a hős megfigyelt szerepgyakorisága; az egérmutatóval a teljes
ötpozíciós eloszlás és a mintaszám is látható. Ez becslés, nem biztos szerepkör
és nem győzelmi esély. Saját kliensadatnál `kliens`, jóváhagyott javításnál `kézi`
jelölés jelenik meg. Kis mintát külön jelez; megfigyelés nélküli szerepet nem
tölt ki kitalált százalékkal.

Adathiánynál vagy bizonytalan oldalnál a **LoL-kliens beolvasása** gomb kézi
ellenőrző ablaka segít. Az oldal külön a **Beállítások / LoL-kliens** lapon is
megadható, a követés kikapcsolása nélkül. A kézzel megerősített pozíciót az aktuális draftban
megjegyzi; új sessionnél új kiosztást készít. A kliens későbbi egyértelmű saját
szerepköre felülírja a kézi korrekciót. Kézi ellenőrzés közben az automata követés
szünetel. Az **Új Draft** kikapcsolja a követést és törli a mezőket.

A szerepkör-statisztika a **`data/champion_roles.json`** fájlban található, stabil
Riot-hősazonosítókkal, rangonként és MIXED összesítéssel. Minden hőshöz megfigyelt
darabszám, öt gyakoriság és teljes mintaszám tartozik. Az adatbázisban szerepkör
szerint rendezett csapatlistákból készül; a kiválasztott rang adatai hiányában a
MIXED összesítést használja. A tanítás a rögzített tanítóadatokból újra elkészíti,
és a modellekkel együtt publikálja. Külön, csak olvasó adatbázis-export is elérhető,
így nem kell megvárni a teljes újratanítást:

```powershell
.\env\Scripts\python.exe .\erospredictor\export_role_distribution.py
# Opcionális konkrét patchablak:
.\env\Scripts\python.exe .\erospredictor\export_role_distribution.py --patches 16.17 16.18 16.19 16.20 16.21
```

Az export alapból a konfigurált patcheket és régiót használja. Nem módosítja az
adatbázist, nem indít scrape-et vagy tanítást, és nem kell hozzá Riot API-kulcs.
Az automatikus draftkövetés is csak a helyi LCU-t használja.

Ha az automatikus keresés nem találja a klienst, a **Beállítások / LoL-kliens**
lapon a **LoL lockfile…** gombbal
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

## Saját hősök, listák és a főablak

A főablakon a draft, a csapat/pozíció választása, a hőspool és a fő műveletek
maradnak. A **Beállítások / előzmények…** ablak külön lapokra rendezi az
ajánlási/elemzési kapcsolókat, a LoL-kliens beállításait és a személyes előzményeket.

A **Hőseim és listák…** szerkesztőben egy hős egyszerre kaphat **Játszom**,
**Komfort** és **Tanulnám** jelölést. Az **Összes saját hős** ezek uniója;
emellett több külön elnevezett saját lista is létrehozható. A **Listatag** oszlop
az éppen kiválasztott saját listát szerkeszti. Az előzményből hozzáadás gomb
csak a már betöltött saját meccseket használja, új lekérést nem indít.

A főablakon kiválasztható az aktív lista. A **Csak ebből ajánlj** kapcsoló
alapból kikapcsolt; bekapcsolva már a számítás előtt korlátozza a jelölteket.
A pickek, banok és off-meta szűrő továbbra is érvényesek. Üres listánál nem
vált át észrevétlenül az összes hősre. A címkék megjelennek az ajánlási listán,
a modell százalékait nem módosítják.

A listák a `%LOCALAPPDATA%/Erospredictor/champion_pools.json` fájlba kerülnek,
stabil Riot-hősazonosítókkal, így egy katalógus-átrendezés nem cseréli ki a hősöket.
Csak a **Mentés** írja felül a helyi listát; a Mégse eldobja a szerkesztést.

## Bansegéd a saját hőspoolhoz

Válassz saját listát és pozíciót, majd kattints a **Bansegéd** gombra.
A mentett lane-párosítások és szerepgyakoriságok alapján legfeljebb öt ötletet
mutat. Párosításonként legalább **20 meccs** kell; kisebb mintából nem ad
szélsőséges ajánlást. A saját pool hősei, a már pickelt és bannolt hősök kimaradnak.
Figyelembe veszi a pool lefedettségét, a kedvezőtlen párosítás erősségét,
a mintanagyságot és az ellenfél gyakoriságát az adatmintában.

A részletek a pool adott hősének teljes meccses győzelmi arányát és mintaszámát
mutatják, nem egy ban feltételezett győzelmi esélyét. A rangsorolási segédpontot
nem jeleníti meg százalékos predikcióként. Ez a funkció csak helyi statisztikát
olvas, nincs neurális inferencia, API-hívás vagy automatikus kliensbeli tiltás.

## Opcionális ajánlás több ellenfél-kiosztásra

A **Beállítások / Ajánlás és elemzés** lapon kapcsolható be a
**Bizonytalan ellenfélpozíciók vizsgálata (első 5)**. Alapból kikapcsolt.
A normál ajánlás első öt jelöltjét legfeljebb három ütközésmentes ellenfél-kiosztásban
vizsgálja újra, az aktuális saját csapatot megtartva. A kézzel megerősített
ellenfélpozíciókat is tiszteletben tartja. A lehetséges automatikus pozícióknál
ugyanaz az 5%-os küszöb és teljes ellenfélcsapatnál jelölt kivétel érvényes.

Az első öt sorrendjét a legrosszabb vizsgált becslés, egyezéskor a súlyozott átlag
adja. A fő százalék az aktuális draft becslése marad; alatta külön látható a
vizsgált kiosztások minimuma és maximuma. Ez nem konfidenciaintervallum.
A kiosztáslinkre kattintva megnyithatók a konkrét felállások és relatív súlyaik.
Két különböző használható kiosztás hiányában normál sorrendet és magyarázatot ad.
Legfeljebb 15 extra kétoldali predikció készül; az alapfelület és a bansegéd
bekapcsolás nélkül nem futtatja ezt az elemzést.

## Saját meccselőzmények

Az előzményvezérlők a **Beállítások / Saját előzmények** lapra kerültek.

Indításkor a program megpróbálja a bejelentkezett saját fiók előzményeit a helyi
LoL-kliensből beolvasni. Az **Előzmények frissítése** gombbal 100, 500 vagy 1000
legutóbbi meccset kérhetsz; a gomb betöltés közben megszakítja a beolvasást.
A statisztika csak Ranked Solo/Duo és Ranked Flex meccsekből számol, legalább
5 perces mérkőzésekkel. A kiválasztott szám az átnézendő összes játékmódot jelenti,
nem ennyi rangsorolt meccs garantált megtalálását.

Egyes kliensverziók minden lapozásra ugyanazt az első 20 meccset adják vissza.
Ezt felismeri és leállítja a program. Látható üzenetben jelzi például a
**20/1000** ténylegesen elérhető meccset; az előzményfelirat súgójában a forrás,
az átnézett meccsek, az azok között talált rangsorolt meccsek és a gyorsítótár
mérete külön olvasható. A 20 friss meccsből kapott nulla rangsorolt találat
nem állítás a fiók összes korábbi meccséről.

A **Régebbi meccsek (Riot)** kapcsoló alapból ki van
kapcsolva, nem mentődik el, és bekapcsolása után külön frissítés szükséges.
Ez külső HTTPS-lekérést használ a telepített kliens saját régiójához beállított
Riot-szerverre, a bejelentkezési tokennel és a saját fiók PUUID-jával.
A token csak memóriában marad; más szolgáltatóhoz nem küldi, átirányítást nem követ.
A fejlesztői Riot API-kulcsot egyik előzményútvonal sem használja. A régebbi
adatok elérhetősége a Riot szolgáltatásától függ; az 1000 meccs itt is felső korlát.
A külső útvonal transportja mockokkal tesztelt; az élő, tokenes lekérés még
nem ellenőrzött. Az integráció mintája a
[LeagueAkari saját klienses előzménykezelése](https://github.com/LeagueAkari/LeagueAkari/blob/dev/src/shared/http-api-axios-helper/sgp/match-history-query.ts).

A hálózati hibákra korlátozott újrapróbálás van, 429 esetén tiszteletben tartja
a `Retry-After` várakozást. Az oldalanként mentett, fiókonként elkülönített helyi
gyorsítótár megmarad megszakításkor; a következő frissítés újrahasználja.
Más játékosok nevét és fiókazonosítóját nem tárolja. Fiókváltáskor nem keveri
össze a személyes statisztikákat.

A **Játszott hősök jelölése az előzményből** kapcsoló legalább 3 meccsnél mutat rendszeresen játszott
hőst, legalább 20 meccs és 55% győzelmi arány felett erősebb keretet kap.
A lane-párosítás és szinergia mintanagysággal együtt jelenik meg; 20 meccs alatt
kis mintát jelez. Ezek a jelölések a modellek százalékait és sorrendjét nem módosítják.
A `services/feature_flags.py` kapcsolójával a teljes funkció továbbra is
szüneteltethető. A helyi draftbeolvasás ettől függetlenül működik.

## Opcionális magyarázat az első öt ajánláshoz

Először kérj normál **AJÁNLÁS**-t. A **Miért? – Kijelölt hős + első 5**
gomb külön, második lépésként számol; kikapcsolt állapotban nincs extra inferencia.
A felület a hosszabb számítások alatt is reagál.

A részletes ablak elején külön **Kijelölt hős** szakaszban látható az ajánlási
listában kiválasztott hős. Utána az első öt ajánlás következik, az eredeti
helyezésükkel. Ha a kijelölt hős köztük van, csak egyszer jelenik meg; ha nincs,
összesen hat különböző hős elemzése látható. A kijelölt hős az egyszerű nyilas
hatások kikapcsolása mellett is bekerül a részletes elemzésbe.

A részletek megmutatják a modellek győzelmiesély-becslését és tényleges súlyát,
az ajánlott hős becsült többletét az üres pozícióhoz képest, valamint azt,
hogyan változik ez a többlet egy-egy ismert társ vagy ellenfél elhagyásakor.
Ez feltételes modellérzékenység, nem összeadható vagy okozati bizonyíték.
A lane-párosítás és ADC–Support szinergia történeti győzelmi aránya külön,
a mérkőzésszámmal együtt látható. A lane-statisztika teljes meccsek kimenetelét
méri, nem az ösvény megnyerésének esélyét. Hiányzó mintát nem pótol kitalált értékkel.
Draft-, rang- vagy szűrőváltás után előbb új ajánlást kell kérni.

## Egyszerű nyilas hőshatások a főablakon

A **Beállítások / Ajánlás és elemzés** lapon lévő **Egyszerű hőshatások**
kapcsoló alapból ki van kapcsolva, ilyenkor
nincs hozzáadott hatásszámítás. Bekapcsolva a hősök és szerepköreik mellett
zöld **▲**, piros **▼**, illetve semleges **→** jelzi a hatást. Az erősebb
szín nagyobb eltérést jelent, a **fehér keret** pedig a fókuszhős lane-ellenfelét
emeli ki. A **● Fókusz** felirat jelöli a kiválasztott pozíciót.

Predikcióhoz válaszd ki lent a csapatot és szerepkört, töltsd ki ott a vizsgált
hőst, majd nyomd meg a **PREDIKCIÓ** gombot. Az aktuális győzelmi becslés mellett
megjelennek az adott hőshöz viszonyított hatások. Üres fókuszpozíció mellett
a normál predikció továbbra is lefut, csak a nyilas elemzéshez kér hősválasztást.

Ajánlásnál először az első jelölt a fókusz. A listában bármelyik ajánlott hős
nevére kattintva átválthatsz rá, anélkül, hogy a draftba beírná. A kijelölt ajánlás
színes hátteret kap. Ez egyetlen hőst vizsgál, nem indítja el a részletes
elemzést. Teljes draftnál legfeljebb 20 gyors, kétoldali predikciót számol.

A szám **százalékpontban** mutatja, hogyan változik a fókuszhős becsült többlete
egy másik ismert hős jelenlétében. Pozitív: kedvező a kiválasztott csapat és hős
számára; negatív: kedvezőtlen. Ez modellérzékenység, nem biztos lane-győzelem vagy
összeadható magyarázat. Draft-, rang-, csapat- vagy szerepkörváltás törli a korábbi
nyilakat, így elavult eredmény nem marad a hősök mellett.

## Golden QA és ensemble minimum

Az ajánlási golden teszt részpontot ad. A legjobb elvárt jelölt Top 5-ben 1,
Top 10-ben 0,75, Top 20-ban 0,5 pontot ér; ezután 0-t. A legelőrébb álló
kerülendő jelölt Top 3-ban 0, a 4–8. helyen 0,5, azon kívül 1 pontot ér.
A meglévő kritériumok átlagát veszi; egy közeli hibás helyezés nem nullázza
az egész esetet. A tanítás QA-eredménye ezt a részpontot használja.
A predikciós golden teszt elfogadási tartományai és pontozása változatlanok.

Minden ténylegesen használható ensemble-komponens legalább **8%** súlyt kap
normalizálás után. Ez az aktuális predikcióra, ajánlásra és magyarázatra is
érvényes, régi kalibrációs fájlok mellett is. Hiányzó vagy hibás komponens
nem kap kitalált hozzájárulást; a megmaradt komponensek újranormalizálódnak.
Mindkét kalibrációs út már a minimumot betartó súlyokat menti. A golden
adatfájl kézi eseteit a pontozás változtatása nem írja át.

## Ellenőrzések

```powershell
Push-Location erospredictor
..\env\Scripts\python.exe -m unittest discover -v
Pop-Location
```

Az automatizált tesztek helyi fixture-ökkel és mockokkal működnek; nem indítanak
éles adatgyűjtést, adatbázisírást vagy teljes modell-újratanítást.
