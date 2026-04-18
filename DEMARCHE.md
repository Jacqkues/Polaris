# Polaris — Démarche Hackathon

> Benchmarking Small Language Models — 2026-04-18 · Équipe Polaris (3 personnes)

---

## 1. Problème & scoring

Génération de code Python Polars à partir d'une question en langage naturel + schéma de tables. Scoring officiel :

```
Score = N / (T × VRAM^0.1 × RAM^0.01)
```

Lecture : **la correctness (N) écrase tout**, la vitesse (T) est le 2ᵉ levier, la VRAM/RAM quasi ignorables (10× VRAM = 1.26× pénalité seulement). Toute notre démarche découle de ça : prompt engineering + bon modèle de code avant toute micro-optim mémoire.

---

## 2. Stratégie

1. **Construire un harness d'éval local** qui reproduit la métrique officielle → itérer en secondes, pas en soumissions.
2. **Tester 3 modèles en parallèle** à 3 personnes, garder le meilleur.
3. **Prompt engineering ciblé** sur les erreurs réellement observées, pas générique.
4. **Cascade adaptative** pour traiter les cas durs sans exploser T sur les cas faciles.

---

## 3. Harness d'éval local

- `dataset/gen_tpch.py` — génère les tables TPC-H
- `dataset/executor.py` — exec sandboxé du code généré, capture `result`
- `dataset/compare.py` — comparaison DataFrame tolérante aux floats + ordre de colonnes
- `data/seeds.jsonl` — 25 questions validées (question, tables, reference_code, expected_hash, tags, difficulty)
- `benchmark_gemma.py` — scoring 4-tiers : `generated → parses → runs → matches`

---

## 4. Modèles testés

| Modèle | Params | VRAM | matches (10 seeds) | Verdict |
|---|---|---|---|---|
| LFM2-8B-A1B (MoE) | 8B/1B actif | ~16 GB | 10% | Score estimé 0.048 |
| **Gemma 4 E2B-it** | 2.3B effective | ~10 GB | **30%** | **×1.7 sur le score — retenu** |
| Mistral 7B Instruct | 7B dense | ~14 GB | Abandonné après 1er run | Pas spécialisé code |

**Paramètres communs** : `do_sample=False` (greedy), `use_cache=True`, `torch.float16`, `torch.inference_mode()`. Gemma utilise `AutoTokenizer` (path text-only, évite la dep `torchvision` du chemin multimodal).

---

## 5. Progression des prompts

| Run | Technique | runs | matches | Latence |
|---|---|---|---|---|
| v1 | LFM2, prompt minimal (10 seeds) | 20% | 10% | 1.57s |
| v2 | Gemma, prompt minimal (10 seeds) | 40% | 30% | 2.86s |
| v3 | Gemma + cheat sheet + 4 few-shot + schéma typé (**25 seeds**) | **84%** | 16% | 2.48s |
| v4 | v3 + cascade L1→L4 (25 seeds) | 92% | 16% | 4.11s |
| v5 | v4 + schéma strict (D) + feedback structuré (E) | 84% | 16% | 3.04s |

**Observation clé** : à partir de v3 les matches plafonnent à 4/25 sur TPC-H. Ce qui bouge, c'est la classe d'erreurs.

---

## 6. Cascade de génération

Pipeline en 4 niveaux qui adapte la stratégie à la difficulté :

| Niveau | Méthode | Coût T | Déclenché quand |
|---|---|---|---|
| **L1 fast** | Greedy + prompt v2 + schéma strict | ~2-3s | Par défaut |
| **L2 constrained** | Outlines CFG sur grammaire Lark dynamique | ~5-8s | Si L1 échoue validation statique (auto-skip si `llguidance` absent) |
| **L3 retry hallucination** | Re-prompt avec anti-patterns détectés | ~3-4s | Si L1/L2 foirent |
| **L4 exec-retry** (×2) | Mock exec sur DataFrames vides → feedback structuré de l'erreur Polars | ~3-4s / retry | Si le code final rate `try_mock_execute` |

La **validation statique** (`looks_ok`) : AST valide + pas d'anti-patterns regex + colonnes référencées ∈ schéma. Même logique côté serveur (pas de tables) et benchmark (cohérence garantie).

Le **feedback structuré** (`build_structured_feedback`) parse les erreurs Polars (`ColumnNotFoundError: "X"; valid columns: [...]`, `DuplicateError`, `SchemaError`, `AttributeError`) et les reformule en instruction actionnable pour le retry.

---

## 7. Découverte tardive : polars.bench ≠ TPC-H

Les erreurs remontées par la plateforme officielle révèlent des schémas **Spider-style** (Northwind, Sakila, Chinook) — pas TPC-H. Conséquence :

- Nos 25 seeds locaux sont un proxy **sémantique** valide mais pas **structurel**.
- Le vrai mode d'échec en prod = **hallucinations de noms de colonnes** (`user_id` au lieu de `customer_id`, `order_id` au lieu de `o_orderkey`).
- C'est **exactement** ce que L4 exec-retry attaque : la liste des colonnes valides est dans l'erreur Polars, suffit au modèle pour corriger.

Sur TPC-H, Gemma n'hallucine quasi pas grâce aux préfixes distinctifs (`c_`, `o_`) → D et L4 tombent à vide localement, mais devraient briller en prod.

---

## 8. Approches alternatives prototypées (non embarquées)

- **Grammar CFG (L2)** — implémentée, `llguidance` instable sur la VM finale. Auto-skip dans la cascade.
- **JSON QueryPlan** (`test_json_generator/`) — modèle produit un plan JSON validé Pydantic, rendu déterministe vers Polars. Propre mais les erreurs sémantiques persistent (5/5 plan_valid, 1/5 runs, 0/5 matches).
- **LoRA sur Spider** — pipeline SFT prêt (spider_loader + sql_oracle + build_sft), training pas lancé.
- **vLLM + GBNF** — version accélérée de la grammar constraint, exploratoire.

---

## 9. Ce qui est embarqué

`main.py` (FastAPI) avec :
- Gemma 4 E2B via `AutoTokenizer`
- Cascade L1→L4 avec degradation gracieuse (exceptions catchées, réponse non-vide garantie)
- Prompt v2 + schéma strict en fin de user turn (rappel passif des colonnes valides)
- Logs verbeux par requête : `[req #X] schema=... cascade=... reason=... code=...`
- Env vars : `POLARIS_MODEL_NAME`, `POLARIS_DISABLE_CONSTRAINED`, `POLARIS_MAX_EXEC_RETRIES`
- **85 tests unitaires** sans GPU (runnables en <1s)

---

## 10. Takeaways

**Ce qui a marché**
- Harness local avant de toucher aux modèles → itérations rapides
- Prompt ciblé sur les erreurs RÉELLEMENT observées (pas générique)
- Cascade avec fallback gracieux → le serveur ne plante jamais
- Tests unitaires sans GPU pour sécuriser les refactos
- Lire la formule de scoring avant d'optimiser (évite quantization, self-consistency)

**Ce qui n'a pas marché**
- Cheat sheet ignoré par le modèle sur ~50% des anti-patterns → il faut la contrainte formelle (grammar)
- Convention d'aliasing non inférable depuis la question → plafond du prompt engineering
- `llguidance` instable sur la VM → L2 pas testée en prod
- Première tentative STRICT block "use ONLY these" → lue comme "select all these" (régression). Corrigée par wording passif.

**Le message clé** : benchmarker un SLM en prod = **lire la formule** pour trier les leviers + **identifier les classes d'erreur réelles** (pas supposer) + **un outil par classe** (prompt guide, grammar force, exec-retry corrige) + **dégrader gracieusement**. Et ne pas confondre proxy local et prod — les signaux réels viennent des logs d'erreur de la plateforme.
