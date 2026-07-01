# Setup e avvio dei test — temporalXes2plan

Guida completa per configurare il progetto su una nuova macchina ed eseguire la suite di test.

---

## 1. Prerequisiti di sistema

### Sistema operativo
Ubuntu 22.04 / Debian 12 o compatibile. Su macOS alcune dipendenze cambiano nome (vedi note).

### Dipendenze di sistema (installare via apt)

```bash
# Compilatore e build tools
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    bison \
    flex \
    perl \
    git \
    curl

# Librerie richieste da OPTIC (planner temporale)
sudo apt-get install -y \
    coinor-libcbc-dev \
    coinor-libclp-dev \
    coinor-libcoinutils-dev \
    libbz2-dev \
    zlib1g-dev
```

> **macOS**: usa Homebrew — `brew install cmake bison flex coin-or-tools`. `bison` va messo in PATH prima di quello di sistema: `export PATH="$(brew --prefix bison)/bin:$PATH"`.

### Miniconda

```bash
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh -b -p "$HOME/miniconda3"
# Rendi conda disponibile nella sessione corrente
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
```

---

## 2. Clonare la repository

```bash
git clone <URL-della-repo> temporalXes2plan
cd temporalXes2plan
```

---

## 3. Ambiente Conda

Crea e popola l'ambiente dal file `environment.yaml`:

```bash
~/miniconda3/bin/conda env create -f environment.yaml
```

L'ambiente si chiama **`temporalXes2Plan`** (Python 3.14) — non rinominarlo, è hardcoded in `CLAUDE.md` e usato in tutti i comandi.

Verifica che sia stato creato:

```bash
~/miniconda3/bin/conda env list
# deve apparire: temporalXes2Plan   /home/<utente>/miniconda3/envs/temporalXes2Plan
```

> **Non attivare mai l'ambiente** con `conda activate` — la shell è non-interattiva. Usa sempre `conda run -n temporalXes2Plan ...` (vedi Sezione 5).

---

## 4. Build dei planner

### 4a. OPTIC (planner temporale — obbligatorio per i test di integrazione)

OPTIC è incluso nella repo sotto `optic/`. Va compilato dalla sorgente.

```bash
cd optic

# Step 1: genera i Makefile con CMake
./run-cmake-release

# Step 2: compila (può richiedere 3-10 minuti)
./build-release

cd ..
```

Il binario risultante è `optic/release/optic/optic-clp`.

Verifica:

```bash
./optic/release/optic/optic-clp --help 2>&1 | head -5
```

#### Errori comuni durante la build di OPTIC

| Errore | Causa | Soluzione |
|---|---|---|
| `Could not find CLP` | Librerie CoinOR mancanti | `sudo apt-get install coinor-libclp-dev coinor-libcoinutils-dev` |
| `zlib.h not found` | zlib dev mancante | `sudo apt-get install zlib1g-dev` |
| `bison: command not found` | bison non installato | `sudo apt-get install bison flex` |
| Linker error `libz.so` | conda non espone libz | `export LIBRARY_PATH=$HOME/miniconda3/envs/temporalXes2Plan/lib:$LIBRARY_PATH` poi ricompila |

#### Alternativa: build via web UI (già integrata nel codice)

Il progetto espone un endpoint web che lancia la build di OPTIC automaticamente. Se preferisci non compilare manualmente, avvia il server (`python -m web.app`) e usa il pannello di amministrazione.

---

### 4b. Fast Downward (planner classico — opzionale)

Usato solo dai test che specificano `planner="fast_downward"`. Non richiesto per la suite evaluation.

```bash
./setup_submodules.sh
# Scarica i submoduli git e compila Fast Downward (~5 min)
```

Il binario si trova in `src/downward/`.

---

## 5. Esecuzione dei test

### Struttura generale

```
tests/
├── encoding/          # Encoding PDDL (domain builder, effect duplicator, …)
├── evaluation/        # Harness di valutazione (piano, metriche, report, …)
├── parsing/           # Parser XES/CSV
├── planning/          # Wrapper planner
├── web/               # API Flask
└── test_core_utils.py # Utility condivise
```

### Comando base (tutti i test unit — veloci, nessun file XES necessario)

```bash
~/miniconda3/bin/conda run -n temporalXes2Plan python -m pytest
```

`pytest.ini` aggiunge automaticamente `src/` e `tests/` al `PYTHONPATH`.

### Comando consigliato (con output verboso)

```bash
~/miniconda3/bin/conda run -n temporalXes2Plan python -m pytest -v
```

### Eseguire solo un sottoinsieme

```bash
# Solo i test di evaluation
~/miniconda3/bin/conda run -n temporalXes2Plan python -m pytest tests/evaluation/ -v

# Solo un file
~/miniconda3/bin/conda run -n temporalXes2Plan python -m pytest tests/evaluation/test_plan_validator.py -v

# Solo una classe
~/miniconda3/bin/conda run -n temporalXes2Plan python -m pytest tests/evaluation/test_plan_validator.py::TestVariantEffects -v

# Solo un test
~/miniconda3/bin/conda run -n temporalXes2Plan python -m pytest tests/evaluation/test_plan_validator.py::TestVariantEffects::test_precondition_violation_with_exact_effects_fails -v
```

### Test di integrazione (richiedono file XES reali)

I test di integrazione sono marcati con `@pytest.mark.integration` e **saltati per default**. Richiedono il file `logs/sample_clinic.xes` (già presente nella repo).

```bash
# Esegui SOLO i test di integrazione
~/miniconda3/bin/conda run -n temporalXes2Plan python -m pytest -m integration -v

# Esegui tutti (unit + integration)
~/miniconda3/bin/conda run -n temporalXes2Plan python -m pytest -m "" -v
```

---

## 6. Parametri pytest utili

| Flag | Significato | Esempio |
|---|---|---|
| `-v` | Output verboso (nome di ogni test) | `-v` |
| `-q` | Output minimale (solo conteggio) | `-q` |
| `-x` | Ferma al primo fallimento | `-x` |
| `--tb=short` | Traceback abbreviato | `--tb=short` |
| `--tb=long` | Traceback completo (default) | `--tb=long` |
| `-k "pattern"` | Filtra per nome (substring/regex) | `-k "validator"` |
| `-m "marker"` | Filtra per marker | `-m "not integration"` |
| `-s` | Non catturare stdout (print visibili) | `-s` |
| `--no-header` | Ometti intestazione di sessione | `--no-header` |
| `--durations=10` | Mostra i 10 test più lenti | `--durations=10` |

### Esempi pratici

```bash
# Esegui veloci, ferma al primo errore, traceback corto
~/miniconda3/bin/conda run -n temporalXes2Plan \
    python -m pytest -x --tb=short -q

# Solo i test evaluation, verbose, con print
~/miniconda3/bin/conda run -n temporalXes2Plan \
    python -m pytest tests/evaluation/ -v -s

# Tutti i test che contengono "variant" nel nome
~/miniconda3/bin/conda run -n temporalXes2Plan \
    python -m pytest -k "variant" -v

# Report completo: tempi + coverage (se coverage è installata)
~/miniconda3/bin/conda run -n temporalXes2Plan \
    python -m pytest --durations=10 -v
```

---

## 7. Verifica setup completo

Esegui questo blocco in ordine — ogni step deve dare exit code 0.

```bash
# 1. Ambiente conda OK
~/miniconda3/bin/conda run -n temporalXes2Plan python -c "import pm4py, sklearn, pandas; print('deps OK')"

# 2. Import moduli progetto OK
~/miniconda3/bin/conda run -n temporalXes2Plan --cwd src \
    python -c "from evaluation.eval_api import EvalAPI; print('imports OK')"

# 3. OPTIC disponibile
test -f optic/release/optic/optic-clp && echo "OPTIC OK" || echo "OPTIC MANCANTE — build necessaria"

# 4. Suite unit test (260 test, ~3 secondi)
~/miniconda3/bin/conda run -n temporalXes2Plan python -m pytest -q
```

Output atteso dell'ultimo comando:
```
260 passed, N warnings in ~3s
```

---

## 8. Note su OPTIC e i test di evaluation

Il modulo `evaluation/` usa OPTIC come planner di default (`planner="optic"`) nella funzione `run_evaluation.py`. I **test unit** di evaluation **mockano** il planner (nessun OPTIC necessario). I **test di integrazione** invocano OPTIC realmente — assicurarsi che il binario sia compilato prima di eseguirli.

Per verificare che OPTIC sia rilevato dal codice:

```bash
~/miniconda3/bin/conda run -n temporalXes2Plan --cwd src \
    python -c "from planning.optic import is_available; print('OPTIC available:', is_available())"
```
