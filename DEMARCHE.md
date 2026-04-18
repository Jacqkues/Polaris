# Polaris — Démarche Hackathon

> Benchmarking Small Language Models in the Real World — 2026-04-18
> Équipe : Polaris (3 personnes)

---

## 1. Contexte

Hackathon organisé autour du benchmarking de **Small Language Models (SLMs)** sur une tâche de **génération de code Python Polars** à partir de requêtes en langage naturel. Les modèles sont évalués dans un environnement Dockerisé avec GPU, sur des critères de correctness, vitesse et empreinte mémoire.

### Le problème à résoudre

Générer du code Polars **exécutable et correct** qui transforme des tables TPC-H selon une demande utilisateur. Exemple :

> *"Group the orders by customer and return the 10 with the highest total revenue."*

Le code généré doit être valide, s'exécuter sans erreur, et produire un `DataFrame` dont le hash correspond à la référence attendue.

---

## 2. Contrainte de scoring — notre boussole

Formule officielle :

```
Score = N / (T × VRAM^0.1 × RAM^0.01)
```

| Facteur | Effet | Verdict |
|---|---|---|
| **N** — réponses correctes | Numérateur linéaire | **Domine tout le reste** |
| **T** — temps total de génération | Linéaire dénominateur | Fortement pénalisant |
| **VRAM^0.1** | 10× VRAM = 1.26× pénalité | Quasi ignorable |
| **RAM^0.01** | 10× RAM = 1.02× pénalité | Ignorable |

### Ce qu'on a déduit

- La **correctness écrase tout** : gagner +30% de `matches` avec un modèle qui consomme 3× plus de VRAM reste un gain net de **×2.4** sur le score.
- La **vitesse est le 2ᵉ levier** : on privilégie le greedy decoding (`do_sample=False`) qui est déterministe ET rapide.
- **Pas de chasse à la VRAM** : la quantization agressive est une perte de temps pour ce scoring.

→ Toute notre démarche découle de cette lecture : **prompt engineering + bon modèle de code** avant toute micro-optimisation mémoire.

---

## 3. Stratégie globale

### Étape 1 — Construire notre propre harness d'évaluation

**Décision clé prise tôt** : ne pas dépendre de polars.bench pour chaque test. Push → clone → run → attendre les résultats, c'est trop lent pour itérer. On a donc construit **notre propre pipeline d'évaluation local**, qui reproduit exactement la métrique officielle.

Composants développés :

- **`dataset/gen_tpch.py`** — génère les tables TPC-H (customer, orders, lineitem, etc.) localement
- **`dataset/executor.py`** — exécute le code Python généré dans un scope restreint, capture la variable `result`, gère les exceptions proprement
- **`dataset/hashing.py`** — hash canonique d'un DataFrame Polars (order-independent sur les lignes si besoin, invariant aux types numériques proches) pour comparer les sorties
- **`data/seeds.jsonl`** — un dataset de **questions seeds validées**, chacune avec : `question` en langage naturel, `tables` nécessaires, `reference_code` (la solution de référence), `expected_output_hash`, `tags` (filter, join, groupby…), `difficulty` (d1→d5)
- **`benchmark.py`** — orchestrateur qui tourne un modèle sur le dataset et sort un rapport 4-tiers

### Les 4 tiers de succès

1. `generated` : le modèle a sorti du texte non vide
2. `parses` : le code est syntaxiquement valide (AST parse)
3. `runs` : le code s'exécute sans exception et produit un `DataFrame`
4. `matches` : le hash du DataFrame correspond à la référence → c'est le **N** du score officiel

Avec `latency_sec` par question → on a la vraie métrique N/T localement.

### Étape 2 — Diviser pour régner sur les modèles

Trois membres, trois modèles testés **en parallèle** sur le même harness :

- **LiquidAI LFM2-8B-A1B** — MoE 8B total / 1B actif, parie sur la vitesse
- **Google Gemma 4 E2B-it** — 2.3B effective params, le plus récent, bon score code
- **3ᵉ modèle** — exploratoire, abandonné après premier test

L'idée : tester vite sur 10 seeds, comparer les scores, **garder le gagnant** pour itérer ensuite (prompt engineering, few-shot, etc.) avant la soumission finale.

---

## 4. Setup technique

### Infrastructure

- **VM RunPod** fournie par les orgas : RTX 5090, 32GB VRAM, Ubuntu, Python 3.12, CUDA 13.0
- **Accès** : SSH + Jupyter + proxy HTTP (port 9000) pour exposer le serveur FastAPI
- **Repo GitHub** : [Jacqkues/Polaris](https://github.com/Jacqkues/Polaris)

### Architecture de soumission

```
┌─────────────┐       push repo          ┌──────────────┐
│  Developer  ├──────────────────────────▶│    GitHub    │
└─────────────┘                            └──────┬───────┘
                                                  │ clone
                                                  ▼
┌───────────────────┐    questions       ┌──────────────────┐
│   polars.bench    │◀───────────────────┤  RunPod (GPU)    │
│  (submission UI)  │    réponses        │  uvicorn main:app│
└───────────────────┘───────────────────▶└──────────────────┘
```

Le repo doit exposer un `main:app` FastAPI que le runner de polars.bench appelle pour chaque question.

### Organisation du code

```
Polaris/
├── main.py              # Serveur FastAPI pour soumission (modèle actuel : LFM2)
├── benchmark.py         # Benchmark local — baseline LFM2
├── benchmark_gemma.py   # Benchmark local — Gemma 4 E2B (fork)
├── dataset/
│   ├── executor.py      # Exécute le code généré et capture le DataFrame
│   ├── hashing.py       # Hash canonique du DataFrame pour comparaison
│   └── gen_tpch.py      # Génère les tables TPC-H locales
├── data/
│   ├── seeds.jsonl      # Questions + code référence + hash attendu
│   └── tpch/            # Tables TPC-H
└── runs/                # Résultats de benchmarks sauvés en JSON
```

---

## 5. Modèles testés

| Modèle | Params | Décodage | VRAM (fp16) | Spécificité |
|---|---|---|---|---|
| LiquidAI LFM2-8B-A1B | 8B / 1B actif (MoE) | Greedy | ~16 GB | MoE : rapide à l'inférence |
| Google Gemma 4 E2B-it | 5B (2.3B effective) | Greedy | ~10 GB | LiveCodeBench v6 : 44% |
| *(3ᵉ — exploratoire, abandonné)* | — | — | — | Non retenu après premier run |

### Pourquoi ces choix

- **LFM2-8B-A1B** : la MoE activate seulement 1B params par token → on espère la vitesse d'un 1B avec la qualité d'un 8B.
- **Gemma 4 E2B** : le plus petit de la famille Gemma 4, optimisé on-device, supporte le rôle système nativement. LiveCodeBench raisonnable pour un 2B effective.
- **Pas de modèles >30B** : le gain en correctness ne compense probablement pas l'explosion de T.

### Paramètres communs imposés

- `do_sample=False` (greedy) — déterministe + plus rapide que sampling
- `use_cache=True` — KV cache, critique pour la latence
- `dtype=torch.float16` — divise par 2 la VRAM sans impact notable sur la correctness
- `torch.inference_mode()` — pas de gradient, inférence plus rapide
- `max_new_tokens=256` — assez pour une requête Polars typique, pas trop long pour T

### Prompt système utilisé (baseline)

```
Return only valid Python Polars code.
No markdown fences.
Assign the final Polars DataFrame to result.
Available datasets: {schema JSON}
```

Court, directif, injecte le schéma des tables. Le modèle sait ainsi quels noms de colonnes sont disponibles.

---

## 6. Expériences & résultats

> Benchmark : 10 seeds TPC-H, prompt système minimal (schéma uniquement), greedy decoding, `max_new_tokens=256`.

### Run 1 — LFM2-8B-A1B (baseline)

| Métrique | Valeur |
|---|---|
| Questions évaluées | 10 |
| `parses` | 8/10 (80.0%) |
| `runs` | 2/10 (20.0%) |
| **`matches` (= N)** | **1/10 (10.0%)** |
| Latence moyenne | 1.57s |
| T (total estimé) | ~15.7s |
| VRAM | ~16 GB |
| **Score estimé** | **≈ 0.048** |

**Par difficulté** : d1 25% · d2→d5 = 0%
**Par tag** : filter 14%, select 33%, tout le reste à 0%

### Run 2 — Gemma 4 E2B-it

| Métrique | Valeur |
|---|---|
| Questions évaluées | 10 |
| `parses` | 10/10 (100.0%) |
| `runs` | 4/10 (40.0%) |
| **`matches` (= N)** | **3/10 (30.0%)** |
| Latence moyenne | 2.86s |
| T (total estimé) | ~28.6s |
| VRAM | ~10 GB |
| **Score estimé** | **≈ 0.082** |

**Par difficulté** : d1 75% · d2→d5 = 0%
**Par tag** : string 100%, select 67%, limit 50%, sort 33%, filter 29%

### Comparaison & choix

| | LFM2-8B-A1B | **Gemma 4 E2B** |
|---|---|---|
| matches | 10% | **30%** (×3) |
| Latence | 1.57s | 2.86s (×1.8 plus lent) |
| VRAM | 16 GB | 10 GB |
| **Score** | 0.048 | **0.082 (×1.7)** |

**Décision** : on part sur **Gemma 4 E2B** pour la suite. Le gain en correctness (×3) écrase largement la perte en vitesse (×1.8), exactement comme la formule le prédit. Bonus : moins de VRAM → plus de marge si on veut augmenter `max_new_tokens` ou batcher.

### Pattern d'échec commun (diagnostic)

Les deux modèles hallucinent massivement des **APIs Polars obsolètes** :

| Erreur hallucinée | API correcte moderne |
|---|---|
| `df.with_column(...)` | `df.with_columns(...)` |
| `pl.desc(...)` | `pl.col(x).sort(descending=True)` |
| `df.agg(...)` | `df.group_by(...).agg(...)` |
| `expr.contains(...)` | `expr.str.contains(...)` |
| `df.len` (attribut) | `df.height` ou `len(df)` |
| `expr.dense` (window rank) | `.rank(method="dense")` |

Les modèles ont été entraînés sur du Polars **pré-1.0**. C'est le **blocker principal** sur notre correctness — et donc l'axe d'optimisation le plus rentable.

Autres échecs récurrents :
- **ColumnNotFoundError** sur des jointures : le modèle invente des colonnes ou lit mal le schéma des tables
- **SyntaxError** (LFM2 uniquement, 2/10) : générations tronquées ou malformées

---

## 7. Optimisations explorées

> Basées sur le diagnostic de la section 6 : le blocker principal est l'hallucination d'APIs Polars obsolètes.

### Prompt engineering (priorité #1)

- [ ] **Ajouter un cheatsheet Polars moderne** dans le system prompt (liste des APIs valides + celles à NE PAS utiliser)
- [ ] **Few-shot examples** couvrant les tags où on pourrit : `agg`, `groupby`, `join`, `window`
- [ ] Enrichir le schéma injecté avec les **types** de colonnes, pas juste les noms
- [ ] Instruction plus explicite sur `result = ...` (variable finale)

### Inférence

- [ ] Réduire `max_new_tokens` (256 → 200) si les réponses tiennent largement dedans → gain sur T
- [ ] `torch.compile` pour accélérer la génération (si stable avec Gemma)
- [ ] Tester `enable_thinking=True` sur les difficultés hautes (d3-d5) : trade-off T vs N ?

### Post-traitement / boucle d'auto-correction

- [ ] Si `exec_result.error` contient `ColumnNotFoundError` : re-prompter avec l'erreur + liste des colonnes valides
- [ ] Si `AttributeError` sur une API obsolète : retry avec injection de la bonne API
- [ ] Budget token max pour éviter d'exploser T sur les retries

---

## 8. Takeaways

*(à remplir en fin de journée)*

### Ce qui a marché

### Ce qui n'a pas marché

### Surprises

---

## 9. Pistes non explorées (faute de temps)

*(à remplir)*

- Fine-tuning léger sur un set d'exemples Polars
- Quantization 4-bit / 8-bit comparée à fp16
- Ensembling de plusieurs modèles
- Auto-correction : le modèle réécrit son code si l'exec échoue

---

## 10. Conclusion

*(à remplir pour la présentation finale — 5 min demo)*
