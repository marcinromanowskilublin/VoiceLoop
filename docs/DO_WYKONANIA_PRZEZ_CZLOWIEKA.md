# Ręczny wkład człowieka — prototyp dual-track

Stan na 20 października 2026. Ten dokument jest aktualizowany po każdym etapie
prac nad prototypem. Oddziela zadania, które można zweryfikować mechanicznie,
od decyzji wymagających znajomości prywatnych danych, kosztu pomyłki albo
docelowego komputera.

## Niezmienna granica prototypu

- Prototyp porównuje baseline pięciu osi z kaskadą 2+2+1 wyłącznie w cieniu.
- Wynik arbitra nie zmienia odpowiedzi, planu, akcji ani potwierdzenia.
- Demonstrator ma wyłącznie chronione tokenem endpointy `GET`.
- Indeks epizodów żyje w osobnej kolekcji Qdrant.
- Zbiór syntetyczny sprawdza kontrakt techniczny, nie jakość prywatnej pamięci.

## Decyzje człowieka wymagane przed lokalnym pomiarem

- [ ] Uruchomić lokalny endpoint embeddingów i Qdrant na pętli zwrotnej.
- [ ] Zatwierdzić, że wskazana maszyna może przetwarzać prywatne epizody.
- [ ] Wykonać kopię bazy SQLite i snapshot kolekcji Qdrant przed indeksowaniem.
- [ ] Uruchomić indexer najpierw bez `--apply` i przejrzeć liczby.
- [ ] Określić dopuszczalny budżet P95, CPU/RAM i liczbę równoległych pomiarów.

## Decyzje człowieka wymagane przed oceną jakości

- [ ] Przygotować prywatny gold set bez commitowania treści do repozytorium.
- [ ] Dla każdego przykładu zatwierdzić oczekiwane źródła i klasę ryzyka.
- [ ] Zatwierdzić znaczenie werdyktów `agreement`, `conflict` i `uncertain`.
- [ ] Ustalić regułę non-inferiority osobno dla czasu, decyzji, osób i abstencji.
- [ ] Ręcznie przejrzeć każdy konflikt dotyczący decyzji lub pytania high-impact.

## Decyzje nadal blokujące jakikolwiek live recall

- [ ] Zatwierdzić politykę stop/abstain na prywatnych przykładach.
- [ ] Zatwierdzić retencję trace’ów oraz starych rekordów bez TTL.
- [ ] Zatwierdzić plan canary, rollback i operatora odpowiedzialnego za wyłączenie.
- [ ] Osobno zlecić zmianę ścieżki odpowiedzi; ten prototyp jej nie zawiera.

## Ręczny przegląd komunikacji publicznej

- [ ] Obejrzeć trzy diagramy README na GitHubie w jasnym i ciemnym motywie.
- [ ] Potwierdzić, że etykiety osi i werdyktów są zrozumiałe dla odbiorcy.
- [ ] Potwierdzić, że README nie sugeruje produkcji, automatycznego awansu ani
  stuprocentowej trafności.
- [ ] Zweryfikować z czytnikiem ekranu sens alt textów i kolejność nagłówków.
- [ ] Zdecydować, czy publicznie pokazywać zrzut panelu obok diagramów
  architektury; kod nie rozstrzyga tej decyzji komunikacyjnej.
- [ ] Przejrzeć publiczny raport `context-evidence-v1` i potwierdzić, że
  syntetycznych liczb nie przedstawiamy jako jakości danych prywatnych.
- [ ] Po scaleniu wybrać numer wersji, utworzyć tag i opublikować GitHub Release;
  CI przygotowuje artefakt, ale nie publikuje wydania automatycznie.

## Dziennik etapów

| Etap | Stan | Co zweryfikowano | Pozostały wkład człowieka |
|---|---|---|---|
| 0. Projekt i granice | zakończony | Osobna kolekcja, dual-track shadow, deterministyczny arbiter i read-only UI są granicą implementacji. | Zatwierdzić później prywatne kryteria jakości; nie są zgadywane w kodzie. |
| 1. Indexer epizodów | zakończony | Kolekcja kandydacka ma osobną nazwę; indexer jest jawny, domyślnie dry-run, odrzuca kolekcję live i pomija niezmienione punkty. | Uruchomić dry-run na kopii danych, potem osobno zdecydować o `--apply`. |
| 2. Runtime dual-track | zakończony | Baseline RRF5 i kaskada 2+2+1 biegną równolegle po wspólnej bramce czasu; arbiter ma trzy jawne werdykty; limit współbieżności odrzuca nadmiar zamiast tworzyć kolejkę; trace nie zawiera tekstu ani jawnych ID. Wynik jest ignorowany przez ścieżkę odpowiedzi. | Zatwierdzić później znaczenie werdyktów na prywatnym gold secie i budżet zasobów. |
| 3. Raport i gold syntetyczny | zakończony | CLI porównuje rzeczywisty RRF5 z 2+2+1 na tej samej izolowanej kolekcji, raportuje metryki obu torów i zawsze zwraca `promotion_allowed=false`. Repo zawiera 40 zrównoważonych przykładów syntetycznych z jawnym czasem obserwacji. | Zastąpić kopię zbioru syntetycznego prywatnymi przykładami poza repo i ręcznie zatwierdzić etykiety. |
| 4. Demonstrator read-only | zakończony | Chronione tokenem endpointy `GET` pokazują wyłącznie pokrycie indeksu, agregaty i content-free trace’y. Panel nie ma przycisku uruchamiającego pomiar, indeksowanie, zmianę flag ani akcję. | Operator powinien potwierdzić, że widok jest zrozumiały i nie sugeruje awansu prototypu do live recall. |
| 5. Weryfikacja końcowa | zakończony | Pełny Linux suite: 1032 zaliczone, 3 pominięte; Ruff czysty; pełny przypięty lock i lokalny venv bez znanych podatności. Niezależny review nie znalazł P0, a wskazane P1 naprawiono i objęto regresjami. | Nadal potrzebne są prywatne etykiety, pomiar na docelowym Windows oraz jawna decyzja operatora o uruchomieniu shadow. |
| 6. README i diagramy | zakończony | Repo przedstawia VoiceLoop jako local-first context layer; głos jest jednym z adapterów; trzy SVG opisują prawdziwe granice dual-track i czasu. | Wykonać ręczny przegląd renderowania, dostępności i publicznych sformułowań powyżej. |
| 7. Publiczny pakiet dowodów | zakończony | Jedna komenda uruchamia bez sieci realny SQLite i silnik Qdrant local, indeksuje dane idempotentnie, mierzy FTS/RRF5/2+2+1, pokazuje agreement/conflict/abstention/unbounded, a prawdziwy `CommandExecutor.submit()` zapisuje `awaiting_confirmation` bez wywołania instrumentowanego handlera. CI publikuje JSON, Markdown i HTML jako artefakt. | Traktować pięć pomiarów czasu tylko jako smoke-run, zebrać liczby na docelowym Windows, nagrać narrację demonstratora i dopiero po scaleniu opublikować wybrany release/tag. |
| 8. Przybliżony czas i ownership telemetrii | zakończony | Wspólny parser buduje bounded plan i jeden `search_text` dla polskich form względnych. Cascade zapisuje role przed retrieval i przekazuje ownership zamiast dublować zapis dual-track; standalone runner nadal zapisuje je domyślnie. Summary agreguje precyzję czasu, bramkę relewancji i rolę epistemiczną bez treści. | Na prywatnych przykładach zatwierdzić szerokości przybliżonych okien, progi pewności i znaczenie nowych rozkładów; kod ich nie promuje automatycznie. |
