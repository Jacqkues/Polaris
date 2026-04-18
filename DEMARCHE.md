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
- **Mistral-7B** — exploratoire, abandonné après premier test

L'idée : tester vite sur 10 seeds, comparer les scores, **garder le gagnant** pour itérer ensuite (prompt engineering, few-shot, etc.) avant la soumission finale.

---

## 4. Setup technique

### Infrastructure

- **VM RunPod** fournie par les orgas : RTX 5090, 32GB VRAM, Ubuntu, Python 3.12, CUDA 13.0
- **Accès** : SSH + Jupyter + proxy HTTP (port 9000) pour exposer le serveur FastAPI
- **Repo GitHub** : [Polaris](https://github.com/Jacqkues/Polaris)

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

> Benchmark local sur seeds TPC-H (10 seeds initialement, **étendu à 25 seeds** entre Run 2 et Run 3 pour couvrir plus de patterns : agg, dt, cast, multi-join, rename, n_unique, etc.). Greedy decoding, `max_new_tokens=256–512`.

### Run 1 — LFM2-8B-A1B (baseline initial, 10 seeds)

| Métrique | Valeur |
|---|---|
| Questions évaluées | 10 |
| `parses` | 8/10 (80.0%) |
| `runs` | 2/10 (20.0%) |
| **`matches` (= N)** | **1/10 (10.0%)** |
| Latence moyenne | 1.57s |
| VRAM | ~16 GB |
| **Score estimé** | **≈ 0.048** |

**Par tag** : filter 14%, select 33%, tout le reste à 0% · **Par difficulté** : d1 25% · d2→d5 = 0%

### Run 2 — Gemma 4 E2B, prompt minimal (10 seeds)

| Métrique | Valeur |
|---|---|
| Questions évaluées | 10 |
| `parses` | 10/10 (100.0%) |
| `runs` | 4/10 (40.0%) |
| **`matches` (= N)** | **3/10 (30.0%)** |
| Latence moyenne | 2.86s |
| VRAM | ~10 GB |
| **Score estimé** | **≈ 0.082** |

**Par tag** : string 100%, select 67%, limit 50%, sort 33%, filter 29% · **Par difficulté** : d1 75% · d2→d5 = 0%

**Verdict** : Gemma bat LFM2 sur le score (×1.7) grâce à une correctness 3× plus haute, malgré une latence 1.8× plus lente — la formule récompense bien la correctness.

### Run 3 — Gemma 4 E2B, prompt v2 (25 seeds)

Prompt enrichi avec : cheat sheet Polars moderne (A1), 4 few-shot examples couvrant agg/join/groupby/window (A2), schéma injecté avec types + n_rows (A3). Voir `gemma_prompt.py`.

| Métrique | Valeur |
|---|---|
| Questions évaluées | **25** |
| `parses` | 25/25 (100.0%) ✓ |
| `runs` | **21/25 (84.0%)** — très bon |
| **`matches` (= N)** | **4/25 (16.0%)** |
| Latence moyenne | 2.48s (plus rapide qu'en Run 2) |
| VRAM | ~10 GB |

**Par difficulté** : d1 57% · d2–d5 = 0% (chute brutale)
**Par tag** : unique 100%, select 40%, limit/string 33%, sort 25%, filter 15% · tous les autres (agg, groupby, join, window, date, cast, arith…) à 0%

### Pattern d'échec sur le Run 3 — le vrai blocker identifié

Sur les 21 échecs du Run 3, **17 sont des erreurs `column_set`** (colonnes attendues vs produites) et 4 sont des crashes. Exemples révélateurs :

| Seed | Question | Alias attendu | Alias produit |
|---|---|---|---|
| 003 | "Count the number of orders per status" | `n_orders` | `count` (default `pl.len()`) |
| 017 | "sum of quantity, mean price, max discount" | `sum_qty`, `avg_price`, `max_disc` | `total_quantity`, `mean_extended_price`, `max_discount` |
| 004 | "customers with the nation they belong to" | `c_name, nation` (2 cols) | toutes les colonnes du join (10+) |
| 013 | "distinct customers per..." | `distinct_customers` | `o_custkey` (pas d'alias) |

→ **Le modèle produit du Polars valide avec les BONS calculs**, mais **les mauvais noms de colonnes**. Le benchmark suit une convention d'abréviation implicite (`qty` pour quantity, `disc` pour discount, préfixes `n_`/`sum_`/`avg_`/`max_`) **non inférable depuis la seule question**. La question "sum of quantity" peut légitimement produire `sum_qty`, `total_quantity`, ou `sum_quantity` — Gemma choisit le plus verbose, le benchmark attend le plus court.

**Les 4 crashes résiduels** :

| Seed | Erreur | API hallucinée |
|---|---|---|
| 005, 009 | `ColumnNotFoundError: 'o_custkey'` | Mauvaise table (utilise customer au lieu d'orders) |
| 006 | `'DataFrame' has no attribute 'len'` | `df.len` (pré-1.0) — **pourtant listée dans le cheat sheet** |
| 008 | `'Expr' has no attribute 'dense'` | `.dense` (pré-1.0) — **pourtant listée dans le cheat sheet** |

→ Le cheat sheet A1 réduit les hallucinations mais n'est **pas toujours respecté** : 2/4 crashes correspondent à des anti-patterns explicitement interdits dans le prompt.

### Comparaison globale

| | LFM2 baseline (10 seeds) | Gemma v0 (10 seeds) | **Gemma v2 (25 seeds)** |
|---|---|---|---|
| parses | 80% | 100% | 100% |
| runs | 20% | 40% | **84%** ✓ |
| **matches** | 10% | **30%** | 16% (⚠ dataset 2.5× plus dur) |
| Latence | 1.57s | 2.86s | 2.48s |
| VRAM | 16 GB | 10 GB | 10 GB |

**À lire avec précaution** : Run 3 est sur 2.5× plus de seeds, dont beaucoup de d2–d5 qui n'existaient pas dans les 10 premiers. Le **drop de matches% n'est pas une régression** du prompt v2, c'est l'ajout de seeds plus durs. Comparé seed par seed sur les 10 communs, v2 est équivalent à v0 (même matches totaux).

**Ce que le prompt v2 a vraiment apporté** :
- `runs` : 40% → 84% (gros gain : le code s'exécute beaucoup plus souvent, les APIs hallucinées baissent)
- Latence : -13% (paradoxalement plus rapide — peut-être mieux ancré, génère moins de tokens inutiles)
- Matches : plafond atteint sur les d1, bloqué sur d2+ par la convention d'aliasing

### Run 4 — Cascade de génération + retry exec

Architecture en 4 niveaux pour adapter la stratégie à la difficulté de chaque question :

| Niveau | Quand | Latence | Commentaire |
|---|---|---|---|
| **L1 — Fast** | default | ~2.5s | Prompt v2 normal, greedy decoding |
| **L2 — Constrained** | Si L1 fail `looks_ok()` statique | ~5-8s | Grammar CFG Polars (auto-skip si `llguidance` indispo) |
| **L3 — Retry hallucination** | Si L2 fail | ~3-4s | Re-prompt avec les anti-patterns détectés en L1 |
| **L4 — Exec retry (×N)** | Si le code final échoue à l'exécution mock | ~3-4s/retry | Re-prompt avec l'**erreur Polars réelle** + liste des colonnes valides |

Points clés d'implémentation :
- **`looks_ok(code, tables)`** : validation **statique** (AST + regex d'anti-patterns + vérif colonnes vs schéma). Utilisable à la fois côté serveur (sans tables) et benchmark.
- **`try_mock_execute(code, schema)`** : crée des DataFrames **vides** avec les bons types, exécute le code. Catch `ColumnNotFoundError`, `DuplicateError`, `SchemaError`, `AttributeError` — sans avoir besoin des vraies données.
- **Retry feedback** : l'erreur Polars est injectée dans le prompt comme message user. Les erreurs Polars sont bien formées (listent les colonnes valides sur ColumnNotFoundError, suggèrent `suffix=` sur DuplicateError), donc le modèle a tout pour corriger.
- **Mémoire courte** : chaque retry voit l'erreur courante + la tentative précédente uniquement, pas tout l'historique. Choix pragmatique (focus + contrôle de T).

### Découverte majeure : polars.bench n'utilise PAS TPC-H

Dès les premiers tests sur la plateforme officielle, les erreurs révèlent des schémas **très différents** de nos 25 seeds locaux :

| Erreur polars.bench | Colonnes visibles | Dataset reconnu |
|---|---|---|
| `ColumnNotFoundError: "category"; valid: [order_id, customer_id, freight...]` | Northwind |
| `valid: [payment_id, staff_id, rental_id, payment_date]` | Sakila |
| `valid: [actor_id, film_id, film_id_right]` | Sakila |

→ **Le benchmark officiel tire des questions de datasets style Spider** (Northwind, Sakila, Chinook…), pas de TPC-H. Conséquences pour notre démarche :
- Notre crainte d'overfit sur les conventions d'aliasing TPC-H **n'a pas eu lieu** par chance — les conventions polars.bench sont différentes de toute façon.
- Le principal mode d'échec côté prod est **l'hallucination de noms de colonnes** (le modèle invente `user_id` au lieu de `customer_id`, `category` au lieu de `category_id`). C'est **exactement** ce que l'exec-retry attaque : la liste des colonnes valides apparaît dans l'erreur Polars et suffit au modèle pour corriger en un retry.

### Décision

**Le prompt engineering a atteint sa limite naturelle** en local. L'étape suivante — exec-retry avec feedback d'erreur — s'attaque à la **bonne classe d'erreurs pour la prod** (hallucinations de colonnes), pas à la nôtre locale (conventions d'aliasing TPC-H). Autrement dit : nos 4/25 locaux ne mesurent **pas** ce qui va vraiment limiter le score en production. La cascade complète est déployée sur `main.py`.

---

## 7. Optimisations — statut

> Brainstorming complet et grille d'analyse dans **[OPTIMISATIONS.md](./OPTIMISATIONS.md)**. Cette section trace ce qui a été **effectivement testé**.

### Implémentées et retenues

- ✅ **A1 — Cheat sheet Polars moderne** : intégré dans `gemma_prompt.py`. Liste les anti-patterns observés dans le baseline (`with_column`, `pl.desc`, `.agg` direct, `.str.contains`, `.len`, `.dense`, `df[bool]`). Résultat : `runs` 40% → 84%, mais 2/4 crashes restent sur des APIs explicitement interdites → **effet réel mais partiel** (le modèle ignore parfois la règle).
- ✅ **A2 — Few-shot chat-format (4 exemples)** : intégrés en tant que tours `user`/`assistant` dans le prompt (pas en texte monolithique). Couvre filter+sort, groupby+agg, join+agg+head, window+rank. Résultat : structures correctes sur les cas simples.
- ✅ **A3 — Schéma enrichi avec types + n_rows** : `format_schema()` dans `gemma_prompt.py` génère un bloc lisible (`- customer (1500 rows) / c_custkey: Int64 / ...`). Bien plus lisible que `json.dumps`. Impact difficile à isoler mais contribue probablement à la baisse des `ColumnNotFoundError`.

### En cours / en attente

- 🔄 **Grammar-constrained (`--constrained`, L2 de la cascade)** : flag implémenté (`GemmaModel.generate_constrained` via Outlines CFG + `build_grammar`). Backend `llguidance` parfois instable sur la VM — la cascade skip L2 auto si indispo, dégénère en fast → retry sans crash.
- ✅ **Retry avec feedback exec (L4 de la cascade)** : `try_mock_execute` + `run_cascade_with_exec_retry` dans `gemma_cascade.py`. Construit des DataFrames vides depuis le schéma, exécute le code, catch les erreurs Polars avant de répondre, re-prompt avec l'erreur comme feedback. Jusqu'à N retries (env `POLARIS_MAX_EXEC_RETRIES`, défaut 2). Branché sur `main.py`. Cible directement les hallucinations de noms de colonnes observées sur polars.bench.
- 🔄 **Pipeline SFT / LoRA** (côté Jacques) : dataset Spider → Polars via transpileur + fine-tuning LoRA. Intéressant si on veut aller au-delà du retry.

### Écartées après analyse (pas testées)

- ❌ **Self-consistency (B1)** — vote sur N échantillons : math du scoring interdit (T ×3-5 tue le score même si N monte).
- ❌ **Quantization agressive 4/8-bit** — `VRAM^0.1` dans la formule rend le gain négligeable (1.26× au mieux), risque de perte de N.
- ❌ **RAG dynamique** — trop lourd à mettre en place sur 6h (embedder + index + tuning).
- ❌ **Transpileur SQL → Polars from scratch** — 5-6h de dev, distribution shift Spider ≠ TPC-H, risque trop élevé.

### Observations qui émergent des runs

- Le prompt engineering seul touche un plafond dès lors que les erreurs ne sont plus syntaxiques mais **sémantiques** (conventions d'aliasing). Confirme le besoin d'approches complémentaires (grammar, fine-tuning).
- Le cheat sheet n'est **pas une garantie** : le modèle peut ignorer la règle. Pour les cas critiques (`df.len`, `.dense`), seule la grammaire formelle donne une garantie dure.

---

## 8. Takeaways

### Ce qui a marché

- **Construire notre propre harness avant de toucher aux modèles**. `benchmark.py` + `data/seeds.jsonl` + `dataset/executor.py` + `dataset/compare.py` reproduisent la métrique officielle → itérations locales en quelques secondes au lieu de soumettre sur polars.bench à chaque changement.
- **Diviser pour régner sur les modèles** (3 en parallèle) pour arriver vite à une décision claire : Gemma 4 E2B écrase LFM2 sur ce type de tâche.
- **Prompt engineering ciblé sur les vraies erreurs observées** (pas générique) : regarder les failures avant d'écrire le cheat sheet. Résultat : `runs` de 40% → 84%.
- **Isoler le prompt dans son propre module** (`gemma_prompt.py`) : rend les itérations propres et les diffs git lisibles.
- **Tests unitaires sans GPU** (`tests/test_gemma_prompt.py`, 21 tests) : détectent instantanément les régressions de plomberie (format schema, few-shot structure, compat avec `build_grammar`) avant même d'allumer la VM.
- **Lire la formule de scoring avec soin avant d'optimiser** : savoir que `VRAM^0.1` est quasi-plat a évité de perdre du temps en quantization.

### Ce qui n'a pas marché (ou pas comme prévu)

- **Le cheat sheet ne garantit rien** : le modèle ignore encore 2 règles sur 4 dans les crashes (`.len`, `.dense`). Conclusion : pour les cas durs, la contrainte formelle (grammaire CFG) est nécessaire — un prompt ne suffit pas.
- **La convention d'aliasing a été une surprise majeure** : on pensait que le gap `runs` → `matches` venait d'erreurs sémantiques de code, c'est en fait surtout un mismatch sur le nom des colonnes de sortie.
- **Le dépendancy management sur la VM** a brûlé du temps (`torchvision` absent → fix via `AutoTokenizer`, `llguidance` absent → grammar non testée dans le temps imparti).

### Surprises

- **Gemma 4 E2B est étonnamment fort sur le Polars moderne** pour un 2.3B effective params (`parses` 100% dès le baseline). Le choix par défaut de `AutoProcessor` dans la doc Google tire toute la chaîne multimodale (vidéo, audio, image) même pour du texte pur → `AutoTokenizer` = même résultat, zéro dep supplémentaire.
- **Le `--constrained` (grammar CFG) est puissant en théorie mais fragile en pratique** : Outlines demande `llguidance` comme backend, non installé par défaut. À prévoir pour toute future soumission.
- **Le format JSON du plan logique Polars est déprécié** (mais fonctionnel) : idée de transpileur SQL → code Polars via `LazyFrame.serialize(format="json")` reste viable si on le code.
- **Le mock-exec sur DataFrame vide est suffisant** pour catch les 3 grandes familles d'erreurs runtime (ColumnNotFoundError, DuplicateError sur join, AttributeError). Pas besoin des vraies données — le plan lazy Polars valide les colonnes/types AVANT la première lecture de data. Ça rend l'exec-retry faisable même dans un serveur qui n'a jamais accès aux tables réelles.
- **polars.bench n'utilise pas TPC-H** : on a découvert ça tardivement par les erreurs remontées (Northwind, Sakila). Notre benchmark local était un proxy sémantiquement raisonnable (mêmes familles de patterns) mais pas structurellement (schémas différents). Leçon : **ne jamais supposer que le proxy d'éval reproduit la distribution officielle** — chercher les erreurs réelles dès que possible.

---

## 9. Pistes non explorées (faute de temps)

- ~~**Retry loop avec feedback d'erreur**~~ → **Fait** (L4 de la cascade, via mock-exec).
- **Fine-tuning LoRA sur dataset synthétique** : pipeline démarré par Jacques (Spider loader + SQL oracle + build_sft). Intéressant si on veut dépasser le plafond du retry. Dataset Spider **aligne** même avec les schémas de polars.bench (découvert tardivement).
- **Historique de retry cumulé** : au lieu de ne voir que la tentative précédente, le modèle verrait tout l'historique des essais + erreurs. Potentiel léger, coût T certain, pas prioritaire.
- **Self-correction sémantique** (non couvert par mock-exec) : code qui tourne mais output shape incorrect. Nécessiterait d'exécuter sur des données réelles, ce que le serveur ne peut pas faire sans les tables.
- **Ensembling Gemma + LFM2** : envoyer les deux, prendre celle qui passe mock-exec. Coûte T, peu de marge.
- **Quantization AWQ / GPTQ** : VRAM quasi ignorable dans la formule, non prioritaire.

---

## 10. Conclusion

Notre démarche a été guidée par **une lecture précise du scoring** (`N / (T · VRAM^0.1 · RAM^0.01)` → correctness écrase tout), ce qui nous a permis d'éviter les fausses pistes (quantization, self-consistency) et de concentrer les 6h disponibles sur les leviers à vrai ROI.

**Ce qu'on livre** :
- Un **harness d'évaluation local complet** (25 seeds TPC-H, 4 tiers de scoring, comparison tolérant aux floats et à l'ordre de colonnes).
- Un **pipeline Gemma 4 E2B** avec prompt engineering ciblé : `runs` 40% → 84%, latence 2.86s → 2.48s.
- Une **cascade de génération à 4 niveaux** (`gemma_cascade.py`) qui adapte la stratégie à la difficulté : fast → grammar-constrained → retry hallucination → **retry via mock execution**. La dernière étape catch les `ColumnNotFoundError` / `DuplicateError` par exécution sur DataFrames vides, puis re-prompt Gemma avec l'erreur Polars réelle comme feedback.
- Un **serveur FastAPI** (`main.py`) avec logs verbeux par requête (niveau cascade atteint, erreurs, code tronqué) pour diagnostic en prod.
- Un **module prompt réutilisable** (`gemma_prompt.py`) et 69 tests unitaires sans GPU.
- Une **analyse honnête du plafond** : le vrai blocker en prod n'est PAS ce qu'on observait en local. Notre benchmark local butait sur des conventions d'aliasing TPC-H ; polars.bench utilise d'autres schémas (Northwind, Sakila) et échoue surtout sur des hallucinations de noms de colonnes — exactement ce que la cascade attaque au niveau L4.

**Le message clé** : benchmarker un SLM en production ne se résume pas à empiler des techniques. Il faut **lire la formule d'éval** pour trier les leviers, **identifier les classes d'erreur observées réellement** (pas supposer), et **choisir l'outil adapté à chacune** : prompt pour guider, grammaire pour forcer structurellement, feedback d'exec pour corriger au cas par cas. Dans un temps contraint, ce ciblage fait la différence entre 30% et 0% de score. Et surtout : **le proxy d'évaluation local n'est pas la prod** — il faut des signaux réels (logs d'erreur) pour ajuster la stratégie.
