# Documentazione Tecnica — temporalXes2plan: Dal Log XES alla Generazione PDDL

## Panoramica Architetturale

Il progetto implementa una pipeline che trasforma un **event log XES** (standard per il process mining) in file **PDDL** (Planning Domain Definition Language) che codificano il processo scoperto come un problema di pianificazione automatica. La pipeline si compone di 4 macro-fasi:

1. **Parsing del log XES** → `xes_parser.py` (classe `Parser`)
2. **Decision Mining** → `decision_mining.py` (funzioni standalone)
3. **Codifica PDDL** → `pddl_encoder.py` (classe `Encoder`)
4. **Orchestrazione** → `main.py`

---

## 1. Entry Point — `src/main.py`

### Flusso principale

```python
if __name__ == "__main__":
    args = parse_arguments()
    
    # 1. Parsing del log
    parser = Parser(args.xes_name, args.log_coverage, args.discovery_algorithm,
                    use_activity_classifier=args.use_activity_classifier)
    
    # 2. Lettura stato iniziale e goal personalizzati
    custom_init = utils.read_state_file(init_file_path)
    custom_goal = utils.read_state_file(goal_file_path)
    
    # 3. Encoding PDDL
    encoder = Encoder(parser, pddl_name, init=custom_init, goal=custom_goal)
    encoder.generate_domain(output_path=domain_file_path)
    encoder.generate_problem(problem_name=pddl_name, output_path=problem_file_path)
    
    # 4. Invocazione del planner
    subprocess.run(['python', call_script_path], check=True)
```

### Argomenti CLI

| Argomento | Default | Descrizione |
|-----------|---------|-------------|
| `--xes_name` | `sepsis.xes` | Nome del file XES |
| `--log_coverage` | `0.001` | Soglia copertura varianti |
| `--discovery_algorithm` | `inductive` | Algoritmo di scoperta Petri net |
| `--use_activity_classifier` | `False` | Combina `concept:name` + `lifecycle:transition` |
| `--init` / `--goal` | paths in `pddl/input/` | File con predicati init/goal custom |

---

## 2. Parsing del Log — `src/xes_parser.py` (classe `Parser`)

### 2.1 Costruttore `__init__`

Il costruttore orchestra l'intera pipeline di parsing in sequenza:

```python
def __init__(self, log_path, coverage_percentage, discovery_algorithm, intervals=None, use_activity_classifier=False):
    self.log = self._load_and_filter_log(coverage_percentage)  # Step 1
    self._split_train_test()                                    # Step 2
    self._discover_petri_net()                                  # Step 3
    self._extract_basic_properties()                            # Step 4
    self._initialize_attributes()                               # Step 5
    self._compute_structure_and_probabilities()                  # Step 6
```

---

### 2.2 Caricamento e Filtraggio — `_load_and_filter_log()`

**Flusso dettagliato:**

1. **Importazione XES**: Usa `pm4py.objects.log.importer.xes.importer.apply(self.log_path)` per caricare il log.

2. **Copia del lifecycle log**: `self.full_lifecycle_log = copy.deepcopy(log)` — salva una copia completa prima di qualsiasi filtraggio.

3. **Activity Classifier** (se `use_activity_classifier=True`):
   - Per ogni evento in ogni traccia, combina `concept:name` + `lifecycle:transition` → `"ActivityName (Lifecycle)"`
   - Esempio: `"Accepted"` + `"In Progress"` → `"Accepted (In Progress)"`
   - Serve per log dove `concept:name` da solo non identifica univocamente un evento (es. BPIC 2013).

4. **Filtraggio lifecycle** (se `use_activity_classifier=False`):
   - Se esiste l'attributo `lifecycle:transition`, mantiene solo gli eventi `"complete"`.

5. **Salvataggio full log**: `self.full_log = log` — log completo (post-classifier, pre-coverage).

6. **Filtraggio per copertura varianti**: `pm4py.filter_variants_by_coverage_percentage(log, coverage_percentage)` — rimuove le varianti rare che non raggiungono la soglia cumulativa di copertura.

---

### 2.3 Split Train/Test — `_split_train_test()`

```python
self.train_df, self.test_df = pm4py.split_train_test(self.log, train_percentage=0.8)
self.log = self.train_df  # Il training set diventa il log usato per la scoperta
```

L'80% del log viene usato per scoprire la Petri net e le proprietà del processo; il 20% è riservato per valutazioni.

---

### 2.4 Scoperta Petri Net — `_discover_petri_net()`

```python
DISCOVERY_ALGORITHMS = {
    'alpha': pm4py.discovery.discover_petri_net_alpha,
    'inductive': pm4py.discovery.discover_petri_net_inductive,
    'heuristics': pm4py.discovery.discover_petri_net_heuristics,
    'ilp': pm4py.discovery.discover_petri_net_ilp
}
```

La funzione selezionata viene invocata sul training log e produce:
- `self.petrinet` — la rete di Petri
- `self.initial_marking` — marking iniziale (token iniziali)
- `self.final_marking` — marking finale
- `self.transitions`, `self.places`, `self.edges` — set estratti dalla rete

---

### 2.5 Estrazione Proprietà Base — `_extract_basic_properties()`

Itera su tutte le transizioni della rete:

- **Transizioni etichettate** (`transition.label is not None`): il nome viene sanitizzato con `utils.sanitize_name()` e aggiunto a `self.activities`.
- **Transizioni silenziose** (`transition.label is None`): viene assegnato un nome `tau_1`, `tau_2`, ... e salvato in `self.silent_transitions` (Dict[Transition → str]).

Inoltre estrae:
- `self.start_activities`: dizionario `{attività_sanitizzata: frequenza}` dal full log.
- `self.end_activities`: analogo per le attività finali.

---

### 2.6 Inizializzazione Attributi — `_initialize_attributes()`

```python
self.attributes = {sanitize_name(attr) for attr in pm4py.get_event_attributes(self.full_log)
                   if attr not in Parser.IGNORED_ATTRIBUTES}
self.attribute_categories = self._categorize_attributes()
```

**`_categorize_attributes()`** classifica ogni attributo (evento e traccia) in:
- `'boolean'` — tutti i valori sono `bool`
- `'numerical'` — tutti i valori sono `int` o `float` (non bool)
- `'categorical'` — tutto il resto

Attributi ignorati (costante `IGNORED_ATTRIBUTES`): `case:concept:name`, `concept:name`, `time:timestamp`, `lifecycle:transition`, `org:resource`, `org:group`, `variant-index`, `Resource`, `org:role`.

---

### 2.7 Computazione Struttura e Probabilità — `_compute_structure_and_probabilities()`

Questa è la fase più complessa. Esegue in sequenza:

#### 2.7.1 Estrazione Predecessori — `extract_predecessors()`

Costruisce un grafo di dipendenze **dirette** tra attività basato sulla struttura della Petri net:

1. `_build_place_input_mapping()`: per ogni place, trova le transizioni in ingresso (archi transizione→place).
2. `_build_predecessor_mapping()`: per ogni arco place→transition, collega le transizioni in ingresso al place con la transizione target.

Risultato: `self.predecessors = {activity: [lista predecessori diretti]}`

#### 2.7.2 Probabilità Decision Points — `compute_decision_points_probabilities()`

1. **Identificazione XOR-split**: un place con più di una transizione uscente è un decision point.
2. **Calcolo frequenze transizioni**: `_compute_activity_frequencies()` conta quante volte l'attività B segue l'attività A nel log (`df_counts[A][B] = count`).
3. **Calcolo probabilità**: `_compute_decision_probabilities()` per ogni decision point:
   - Trova le attività predecessori (non-tau) usando `_add_non_tau_predecessors()` ricorsivamente.
   - Conta quante volte ciascuna transizione uscente viene eseguita dopo i predecessori.
   - Normalizza i conteggi in probabilità.

Risultato: `self.decision_points_probabilities = {place_name: {activity: probability}}`

#### 2.7.3 Identificazione Paralleli — `identify_parallels()`

Identifica AND-split: transizioni con più place uscenti che non sono decision points.

1. Calcola `place_outgoing_transitions` e `transition_outgoing_places`.
2. Identifica i place che sono XOR-split (`decision_point_places`).
3. Per ogni transizione con più place uscenti (e nessuno di essi è un XOR-split), raccoglie le attività successive come parallele.

Risultato: `self.parallels = {activity_fork: [lista attività parallele]}`

#### 2.7.4 Grafo Transizioni Dirette — `_build_direct_transition_graph()`

Rimuove i place dalla Petri net e connette direttamente le transizioni:
- Per ogni place, collega tutte le attività in ingresso con tutte le attività in uscita.

Risultato: `self.direct_transition_graph = {activity: [lista successori diretti]}`

#### 2.7.5 Decision Mining — invocazione di `decision_mining.py`

```python
rules, intervals, samples = discover_all_decision_rules(self.log_path, log=self.full_log)
self.attribute_domains = get_attribute_domains(samples, intervals)
self.decision_samples = samples
```

---

## 3. Decision Mining — `src/decision_mining.py`

### 3.1 Funzione principale: `discover_all_decision_rules()`

**Input**: path del log + log pre-caricato opzionale.
**Output**: `(all_decision_rules, all_intervals, remapped_samples)`

#### Passo 1: Preparazione dati — `load_and_prepare_log()`

1. Converte il log pm4py in DataFrame pandas.
2. Aggiunge colonna `next_activity` (attività successiva per ogni evento).
3. Auto-detecta i tipi di feature:
   - **Numeriche**: colonne con dtype numerico
   - **Booleane**: colonne bool o colonne categoriche con soli valori True/False
   - **Categoriche**: tutto il resto
4. Forward-fill per gruppo (case): `df.groupby('case:concept:name')[features].ffill()`

#### Passo 2: Identificazione Decision Points — `find_decision_points()`

Un'attività è un decision point se:
- Ha più di un `next_activity` unico
- Appare almeno `min_instances` volte (default: 100)

#### Passo 3: Analisi Decision Points (2 passi)

**Pass 1 — Feature Importance**: per ogni decision point, addestra un `DecisionTreeClassifier` per raccogliere l'importanza delle feature a livello globale.

**Pass 2 — Training finale**: addestra il decision tree finale con le top-K feature globali (default K=4).

`run_analysis_for_decision_point()`:
1. Filtra il DataFrame per il decision point corrente.
2. One-hot encode delle variabili categoriche.
3. Addestra `DecisionTreeClassifier(max_depth=2, min_samples_leaf=150, class_weight='balanced')`.
4. Estrae le guardie con `extract_simple_guards()`.

#### Passo 4: Estrazione Guardie — `extract_simple_guards()`

Percorre ricorsivamente l'albero decisionale e per ogni foglia (classe = attività successiva) costruisce la congiunzione delle condizioni del percorso:

- **Booleane**: `(attr)` o `(not (attr))`
- **Categoriche**: `(attr evalue)` (prefisso `e` per compatibilità PDDL)
- **Numeriche**: vengono convertite in intervalli discretizzati usando `utils.discretize_value()`. Es: `(amount lte_500_0)` o `(crp gte_28_5_lte_147_5)`

#### Passo 5: Consolidamento Intervalli

La funzione `_find_matching_intervals()` rimappa le soglie estratte dagli alberi in **intervalli non-overlapping** che coprono l'intero dominio numerico:
- `lte_X` (da -∞ a X)
- `gte_X_lte_Y` (da X a Y)
- `gte_X` (da X a +∞)

`_remap_samples_to_consolidated_intervals()` duplica i samples per ogni intervallo consolidato che matcha la condizione originale.

#### Passo 6: Domini degli Attributi — `get_attribute_domains()`

Per ogni attributo nelle precondizioni dei samples, raccoglie l'insieme dei possibili valori. Per gli attributi numerici con intervalli consolidati, genera i nomi degli intervalli.

---

## 4. Codifica PDDL — `src/pddl_encoder.py` (classe `Encoder`)

### 4.1 Costruttore

```python
def __init__(self, parser, domain_name, min_confidence=0.1, init=None, goal=None, minimal_preconditions=False):
```

Il costruttore esegue:

1. **Scoperta relazioni attributo→attività**: `parser.discover_attribute_activity_relationships()` — identifica quando un valore di attributo correla significativamente con l'esecuzione di certe attività.

2. **Scoperta effetti attività→attributo**: `parser.discover_activity_attribute_effects()` — identifica come le attività modificano i valori degli attributi.

3. **Estrazione tau activities**: filtra le attività che iniziano con `tau_`.

4. **Elaborazione Decision Points**: `self._process_decision_points()` normalizza le probabilità dei rami.

5. **Elaborazione Decision Preconditions**: dal `parser.decision_samples`, estrae le precondizioni categoriche per ogni attività.

6. **Mappa parallelismo**: `_build_parallel_activity_map()` costruisce una mappa bidirezionale di quali attività possono eseguire in parallelo.

---

### 4.2 Generazione Domain — `generate_domain()`

Genera il file `domain.pddl` con questa struttura:

```
(define (domain <name>)
  (:requirements :strips :typing :universal-preconditions :conditional-effects :negative-preconditions)
  (:types ...)
  (:constants ...)
  (:predicates ...)
  (:action exec_<activity1> ...)
  (:action exec_<activity2> ...)
  ...
)
```

#### 4.2.1 Tipi — `_generate_pddl_type_definitions()`

- `activity` — tipo base per tutte le attività
- `<attr>_type` — un tipo per ogni attributo numerico/categorico rilevante (usato nelle precondizioni o effetti)

#### 4.2.2 Costanti — `_generate_pddl_constants_definitions()`

- Nomi di tutte le attività (incluse tau) come costanti di tipo `activity`
- Valori possibili per ogni attributo rilevante (dal `parser.attribute_domains`)

#### 4.2.3 Predicati — `_generate_pddl_predicate_definitions()`

Predicati fissi:
- `(completed ?a - activity)` — l'attività è stata completata
- `(enabled ?a - activity)` — l'attività è abilitata

Predicati per attributi:
- Booleani: `(attr)` (proposizione senza parametri)
- Numerici/Categorici: `(attr ?v - attr_type)` (con parametro tipizzato)

#### 4.2.4 Azioni — `_generate_pddl_action_definition()`

Per ogni attività, genera una o più varianti PDDL. Il processo di generazione è il più complesso:

**Caso 1: Measurement Variants** — se l'attività corrisponde a un attributo non-booleano (es. `crp` corrisponde all'attributo `crp`), genera un'azione separata per ogni possibile valore dell'attributo:
- `exec_crp-lte_28_5`, `exec_crp-gte_28_5_lte_147_5`, `exec_crp-gte_147_5`

**Caso 2: Decision Variants** — se l'attività appare in `self.decision_preconditions`, genera una variante per ogni set di precondizioni + un fallback senza precondizioni categoriche:
- `exec_activity_v1`, `exec_activity_v2`, ..., `exec_activity_fallback`

**Caso 3: OR Precondition Variants** — se le precondizioni hanno condizioni OR (es. più predecessori possibili), genera una variante per ogni combinazione:
- `exec_activity_v0`, `exec_activity_v1`, ...

**Caso 4: Azione singola** — nessuna variante necessaria.

---

### 4.3 Precondizioni — `_collect_preconditions()`

Per ogni azione, le precondizioni includono:

1. **`(enabled <action>)`** — sempre presente.
2. **Predecessore completato** (se non è start activity e `minimal_preconditions=False`):
   - Cerca i predecessori nel `direct_transition_graph`.
   - Se un solo predecessore: `(completed pred)`
   - Se multipli: usa il primo (o genera varianti).
3. **Condizioni basate su attributi** — dalla `attr_activity_relationships`: se un attributo correla con l'esecuzione dell'attività con probabilità ≥ `min_confidence`.

La deduplicazione avviene alla fine per evitare precondizioni duplicate.

---

### 4.4 Effetti — `_collect_effects()`

Per ogni azione, gli effetti includono:

1. **`(completed <action>)`** — completamento base.
2. **Abilitazione successori**:
   - Se l'azione è un decision point: usa effetti condizionali `(when (attr val) (enabled successor))`.
   - Altrimenti: abilita tutti i successori con `(enabled succ)`.
3. **Disabilitazione corrente**: `(not (enabled <action>))`.
4. **Effetti su attributi**:
   - Per measurement actions: rimuove il vecchio valore (`forall/when/not`) e imposta il nuovo.
   - Per attributi booleani: `(attr)` o `(not (attr))`.

---

### 4.5 Generazione Problem — `generate_problem()`

Genera il file `problem.pddl`:

```
(define (problem <name>_prediction)
  (:domain <domain_name>)
  (:init ...)
  (:goal ...)
)
```

#### Init Section — `_generate_init_section()`

- Se ci sono predicati custom (da file): li include direttamente.
- Altrimenti: `(enabled <first_start_activity>)`.

#### Goal Section — `_generate_goal_section()`

- Se ci sono predicati custom: li include.
- Altrimenti: `(completed <random_end_activity>)` — sceglie casualmente un'attività finale valida.

---

## 5. Funzioni Utility — `src/utils.py`

### Funzioni chiave usate nella pipeline:

| Funzione | Scopo |
|----------|-------|
| `sanitize_name(name)` | Rende un nome compatibile PDDL (lowercase, sostituisce spazi/caratteri speciali con `_`) |
| `discretize_value(attr, value, intervals)` | Converte un valore numerico in un nome di intervallo (es. `gte_28_5_lte_147_5`) |
| `read_state_file(path)` | Legge predicati init/goal da file |
| `compute_enabled_activities_after_prefix(parser, prefix)` | Replay su Petri net per calcolare attività abilitate |
| `compute_initial_state(parser, prefix, trace_events)` | Calcola lo stato iniziale completo per il planning |
| `extract_base_activity_name(action_name)` | Estrae il nome base da varianti (es. `crp-lte_28_5` → `crp`) |

### `discretize_value()`

Questa funzione è fondamentale per la codifica degli attributi numerici. Dato un valore e gli intervalli (split points), determina in quale intervallo cade e restituisce il nome dell'intervallo come stringa PDDL-compatibile.

---

## 6. Diagramma di Flusso Completo

```
XES Log File
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Parser.__init__()                                   │
│                                                      │
│  ┌───────────────────────────────────────────────┐  │
│  │ _load_and_filter_log()                         │  │
│  │  • Import XES (pm4py)                          │  │
│  │  • Activity Classifier (opzionale)             │  │
│  │  • Filtro lifecycle (solo 'complete')          │  │
│  │  • Filtro copertura varianti                   │  │
│  └───────────────────────────────────────────────┘  │
│                    │                                  │
│                    ▼                                  │
│  ┌───────────────────────────────────────────────┐  │
│  │ _split_train_test() → 80/20                    │  │
│  └───────────────────────────────────────────────┘  │
│                    │                                  │
│                    ▼                                  │
│  ┌───────────────────────────────────────────────┐  │
│  │ _discover_petri_net()                          │  │
│  │  • Alpha/Inductive/Heuristics/ILP Miner       │  │
│  │  → petrinet, initial_marking, final_marking   │  │
│  └───────────────────────────────────────────────┘  │
│                    │                                  │
│                    ▼                                  │
│  ┌───────────────────────────────────────────────┐  │
│  │ _extract_basic_properties()                    │  │
│  │  • Attività (sanitized names)                  │  │
│  │  • Tau transitions (tau_1, tau_2, ...)         │  │
│  │  • Start/End activities + frequenze            │  │
│  └───────────────────────────────────────────────┘  │
│                    │                                  │
│                    ▼                                  │
│  ┌───────────────────────────────────────────────┐  │
│  │ _initialize_attributes()                       │  │
│  │  • Categorizzazione: bool/numerical/categorical│  │
│  └───────────────────────────────────────────────┘  │
│                    │                                  │
│                    ▼                                  │
│  ┌───────────────────────────────────────────────┐  │
│  │ _compute_structure_and_probabilities()         │  │
│  │  • Predecessori (grafo dipendenze)             │  │
│  │  • Probabilità XOR-split                       │  │
│  │  • AND-split (paralleli)                       │  │
│  │  • Grafo transizioni dirette                   │  │
│  │  • Decision Mining (→ decision_mining.py)      │  │
│  │    → attribute_domains, decision_samples       │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  Encoder.__init__(parser)                            │
│                                                      │
│  • discover_attribute_activity_relationships()       │
│  • discover_activity_attribute_effects()             │
│  • Process decision points & preconditions           │
│  • Build parallel activity map                       │
└─────────────────────────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
┌──────────────────┐  ┌──────────────────┐
│ generate_domain()│  │generate_problem()│
│                  │  │                  │
│ • Types          │  │ • :init          │
│ • Constants      │  │ • :goal          │
│ • Predicates     │  │                  │
│ • Actions        │  │                  │
│   (variants)     │  │                  │
└──────────────────┘  └──────────────────┘
          │                     │
          ▼                     ▼
    domain.pddl           problem.pddl
```

---

## 7. Strutture Dati Chiave

### Parser (dopo inizializzazione completa)

| Attributo | Tipo | Descrizione |
|-----------|------|-------------|
| `self.activities` | `Set[str]` | Tutte le attività (nomi sanitizzati, incluse tau) |
| `self.silent_transitions` | `Dict[Transition, str]` | Map transizione → nome tau |
| `self.predecessors` | `Dict[str, List[str]]` | Predecessori diretti per attività |
| `self.decision_points_probabilities` | `Dict[str, Dict[str, float]]` | Probabilità rami XOR |
| `self.parallels` | `Dict[str, List[str]]` | Fork → attività parallele |
| `self.direct_transition_graph` | `Dict[str, List[str]]` | Grafo successori diretti |
| `self.attribute_domains` | `Dict[str, Set[str]]` | Valori possibili per attributo |
| `self.decision_samples` | `List[Dict]` | Samples per decision mining |
| `self.attribute_categories` | `Dict[str, str]` | Tipo per attributo |
| `self.intervals` | `Dict[str, List[float]]` | Soglie discretizzazione numerica |

### Encoder (attributi principali)

| Attributo | Tipo | Descrizione |
|-----------|------|-------------|
| `self.attr_activity_relationships` | `Dict[str, Dict[str, Dict[str, ...]]]` | Influenza attributi su attività |
| `self.activity_attr_effects` | `Dict[str, Dict[str, Any]]` | Effetti attività su attributi |
| `self.decision_preconditions` | `Dict[str, List[Set[str]]]` | Precondizioni data-driven |
| `self.processed_decision_points` | `Dict[str, Dict[str, float]]` | Probabilità normalizzate |
| `self.split_actions` | `Dict[str, Any]` | Varianti generate per OR |
| `self.effect_alternatives` | `Dict[str, Any]` | Alternative effetti per varianti |

---

## 8. Punti Critici per Modifiche

1. **Aggiungere nuovi attributi da considerare**: modifica `IGNORED_ATTRIBUTES` in `Parser` e la logica di `_categorize_attributes()`.

2. **Modificare la logica di generazione precondizioni**: `Encoder._collect_preconditions()` è il punto centrale. Il flag `minimal_preconditions` già supporta una modalità semplificata.

3. **Cambiare la strategia di discretizzazione**: `utils.discretize_value()` + la generazione degli intervalli in `decision_mining.py` (`_find_matching_intervals()`).

4. **Aggiungere nuovi tipi di effetti PDDL**: `Encoder._collect_effects()` e i suoi metodi helper.

5. **Modificare il decision mining**: i parametri in `DEFAULT_CONFIG` (profondità albero, min_samples, top_features) e `extract_simple_guards()` per cambiare come le condizioni vengono tradotte.

6. **Supportare nuovi planner**: il formato PDDL generato usa `:requirements :strips :typing :universal-preconditions :conditional-effects :negative-preconditions`. Per planner diversi potrebbe servire modificare `generate_domain()`.
