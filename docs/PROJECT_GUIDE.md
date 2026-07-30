# Guia do projeto — arquitetura, classes e como replicar

Documento de consulta pessoal. Explica **o que** cada pacote e classe fazem e,
principalmente, **por quê** — para servir de modelo em novos projetos de ML.

---

## 1. Filosofia

O projeto é um **monólito modular** com **arquitetura hexagonal** (ports & adapters)
e **DDD leve**. A regra central:

> O núcleo (domínio + casos de uso) não conhece frameworks. Pandas, PyTorch,
> MLflow, FastAPI vivem na borda e se conectam ao núcleo através de **portas**
> (interfaces). A dependência aponta sempre **para dentro**.

Benefícios concretos:
- Trocar o modelo (baseline ↔ PyTorch) não toca no núcleo.
- Testar o pipeline inteiro sem tocar em disco/rede, usando *fakes*.
- Trocar a fonte de dados (CSV ↔ banco) mexendo só no adapter.

### Ordem de construção (a "receita" para um novo projeto)

```
1. Domínio limpo       -> entities, value objects, regras (sem framework)
2. Portas / contratos  -> interfaces do que o núcleo precisa do mundo
3. Casos de uso        -> orquestração, dependendo só das portas
4. Infraestrutura      -> adapters concretos (pandas, torch, mlflow)
5. Interfaces          -> API / CLI (composição das dependências)
6. MLOps               -> MLflow, DVC, Docker
```

Primeiro o estável (domínio), por último o volátil (modelo, infra). Isso evita o
"notebook disfarçado de projeto".

---

## 2. Estrutura de pastas

```
src/recommender/
├── domain/            núcleo puro
│   ├── entities/          objetos com identidade / ciclo de vida
│   ├── value_objects/     objetos definidos pelo valor (imutáveis)
│   ├── repositories/      contratos de acesso a coleções do domínio
│   └── services/          regras que cruzam vários objetos
├── application/       orquestração
│   ├── ports/             contratos do que o núcleo precisa da infra
│   ├── dtos/              objetos de transferência (primitivos nas bordas)
│   └── use_cases/         casos de uso (um verbo de negócio cada)
├── infrastructure/    adapters concretos (a "borda suja")
│   ├── datasets/          leitura de dados (pandas)
│   ├── preprocessing/     encoding e estratégias de feedback
│   ├── models/            modelos e factory
│   ├── training/          loop de treino, early stopping
│   ├── evaluation/        métricas de ranking
│   ├── tracking/          MLflow (experimentos)
│   ├── registry/          MLflow Model Registry
│   └── persistence/       repositórios em memória / disco
├── interfaces/        pontos de entrada
│   ├── api/               FastAPI (inferência)
│   └── cli/               linha de comando (composição)
└── shared/            utilidades (seed, exceptions)
```

---

## 3. Domínio (`domain/`)

O coração do problema. Não importa pandas, torch, mlflow — nada externo.

### Value Objects (`value_objects/`)

Objetos definidos pelo **valor**, imutáveis (`@dataclass(frozen=True)`) e válidos
por construção (validam no `__post_init__`). Evitam "primitive obsession":
um `Rating` nunca é apenas um `float` solto.

| Classe | Papel | Regra |
|---|---|---|
| `UserId` | Identidade do usuário | inteiro positivo |
| `ItemId` | Identidade do item | inteiro positivo |
| `Rating` | Nota dada por um humano | 0.5 ≤ valor ≤ 5.0; `is_positive(≥4.0)` |
| `Score` | Relevância prevista pelo modelo | ≥ 0; não se auto-ordena |
| `Interaction` | Fato "usuário avaliou item em T" | composto de VOs; delega `is_positive` ao Rating |
| `Metrics` | Resultado de avaliação | dicionário nome→valor |
| `Hyperparameters` | Config de treino | dicionário nome→valor |

Por que `Rating` ≠ `Score`: um é **entrada humana** (fato passado), o outro é
**saída do modelo** (predição). Tipos distintos evitam confusão semântica.

### Entities (`entities/`)

Objetos com **identidade** própria (não são definidos pelos atributos).

| Classe | Identidade | Observação |
|---|---|---|
| `Item` | `item_id` | igualdade/hash só pelo id (`compare=False` nos demais campos) |
| `Recommendation` | — | resultado com `ranked()` e `top_k()`; a política de ordenação mora aqui |
| `Experiment` | `name` | **mutável** (ciclo de vida): começa sem métricas, `finish()` as adiciona |

`Recommendation` não sabe de onde vieram os scores — só ordena por eles. `Experiment`
é a única entity mutável: `__eq__`/`__hash__` manuais baseados no `name`.

### Repositories (`repositories/`)

Contratos (`typing.Protocol`) de **acesso a coleções** de objetos do domínio.
Abstraem a origem (CSV, banco, memória).

| Contrato | Operações | Por quê |
|---|---|---|
| `InteractionRepository` | `all`, `for_user` | só leitura (dados históricos) |
| `ExperimentRepository` | `save`, `get_by_name`, `all` | a app **produz** experimentos |
| `ModelRepository` | `save`, `get` | persistência simples do modelo treinado |

### Services (`services/`)

`RecommendationService.filter_already_seen`: remove da recomendação itens que o
usuário já viu. Vive num service porque **cruza** recomendação + histórico — não
cabe numa entity sozinha.

---

## 4. Application (`application/`)

### Ports (`ports/`)

Interfaces do que o núcleo precisa **da infraestrutura** (driven ports). Usamos
`typing.Protocol` (tipagem estrutural): o adapter não precisa herdar, só ter os
métodos certos.

| Porta | Contrato | Implementada por |
|---|---|---|
| `DatasetReader` | `read_items`, `read_interactions` | `MovieLensReader` |
| `RecommendationModel` | `recommend(user, k)` | `PopularityModel`, `TorchRecommendationModel` |
| `ModelTrainer` | `train(interactions, hp)` | `BaselineTrainer`, `PyTorchTrainer` |
| `MetricCalculator` | `calculate(model, test, k)` | `RankingMetricCalculator` |
| `ExperimentTracker` | `track(experiment)` | `MlflowExperimentTracker` |
| `ModelRegistry` | `register`, `load_production`, `promote` | `MlflowModelRegistry` |

`RecommendationModel` separa o **artefato de inferência** ("sabe recomendar") do
`ModelTrainer` ("sabe treinar"). Um trainer produz um model.

### DTOs (`dtos/`)

Objetos de transferência entre as bordas. Usam **primitivos**, não value objects,
porque atravessam para JSON/CLI.

- `TrainModelInput` — parâmetros de treino vindos do CLI/API.
- `RecommendationOutput` — recomendação serializável (`from_domain` converte VOs → primitivos).
- `DatasetSplit` — partições de treino e teste.

### Use Cases (`use_cases/`)

Cada um é **um verbo de negócio**, dependendo só de portas (injeção de dependência).

| Caso de uso | Faz | Depende de |
|---|---|---|
| `PreProcessDataset` | lê e divide (split por usuário) | `DatasetReader` |
| `TrainModel` | treina e persiste | `ModelTrainer`, `ModelRepository` |
| `EvaluateModel` | calcula métricas no teste | `MetricCalculator` |
| `GenerateRecommendation` | carrega modelo, recomenda, filtra vistos | `ModelRepository`, `InteractionRepository`, `RecommendationService` |

`PreProcessDataset` faz **split temporal por usuário** (leave-last-out): as
interações mais recentes de cada usuário vão para teste. Isso evita cold start
artificial na avaliação (todo usuário de teste tem histórico no treino).

---

## 5. Infrastructure (`infrastructure/`)

A "borda suja" — é onde os frameworks entram. Cada adapter traduz o mundo externo
em objetos do domínio.

### `datasets/`
- `MovieLensReader` — lê `movies.csv`/`ratings.csv` com pandas e **traduz cada linha
  em `Item`/`Interaction`**. O núcleo nunca vê um DataFrame.
- `PandasInteractionRepository` — serve as interações em memória (implementa `InteractionRepository`).

### `preprocessing/`
- `IndexEncoder` — mapeia ids esparsos (com buracos) para índices densos `0..N-1`,
  exigência das camadas de embedding.
- `FeedbackStrategy` (Strategy) — `ExplicitFeedbackStrategy` (usa a nota) vs
  `ImplicitFeedbackStrategy` (gostei/não gostei). Permite tratar o mesmo dataset de duas formas.

### `models/`
- `PopularityModel` / `BaselineTrainer` — baseline: conta popularidade (`Counter`)
  e recomenda os itens mais frequentes. Serve de **régua**.
- `EmbeddingRecommender` (nn.Module) — a rede: embedding de usuário e item + vieses;
  a nota prevista é o produto escalar + vieses.
- `TorchRecommendationModel` — adapter que usa a rede treinada + encoders para gerar
  uma `Recommendation` (ponte tensor → domínio).
- `model_factory.create_trainer` — **Factory Method**: string ("baseline"/"pytorch")
  → trainer concreto. Imports preguiçosos para não puxar torch sem necessidade.

### `training/`
- `PyTorchTrainer` — o loop: encoders → tensores → `DataLoader` → rede + `Adam` +
  `MSELoss` → forward/backward/step por época → `TorchRecommendationModel`.
- `EarlyStopping` — para quando a perda estaciona por `patience` épocas.

### `evaluation/`
- `metric_strategy` — funções puras `precision_at_k`, `recall_at_k`, `ndcg_at_k`.
- `RankingMetricCalculator` — orquestra as funções por usuário e agrega (média).

### `tracking/` e `registry/`
- `MlflowExperimentTracker` — registra params/metrics de um `Experiment` no MLflow.
- `MlflowModelRegistry` — versiona modelos e gerencia estágios (Staging/Production).

### `persistence/`
- `InMemoryModelRepository` / `InMemoryExperimentRepository` — implementações simples (testes/local).
- `FilePickleModelRepository` — salva/carrega o modelo em disco (usado pela API e pelo treino).

---

## 6. Interfaces (`interfaces/`)

O *composition root*: aqui as dependências concretas são criadas e injetadas.

- `api/app.py` — FastAPI. `create_app(use_case)` recebe o caso de uso (testável com fake);
  `app` é a composição padrão que o uvicorn sobe. Endpoints: `/health`,
  `/users/{id}/recommendations`.
- `cli/train.py` — pipeline completo: preprocess → train → evaluate → track no MLflow.
- `cli/evaluate.py`, `cli/preprocess.py`, `cli/recommend.py` — utilitários de linha de comando.

---

## 7. Padrões de projeto usados

| Padrão | Onde | Para quê |
|---|---|---|
| Ports & Adapters | ports + infrastructure | isolar o núcleo de frameworks |
| Repository | domain/repositories | abstrair acesso a coleções |
| Factory Method | `model_factory` | criar o modelo certo sem `if/else` espalhado |
| Strategy | `FeedbackStrategy` | trocar como o feedback é interpretado |
| DTO | application/dtos | transferir dados entre bordas com primitivos |
| Dependency Injection | use cases + `create_app` | testabilidade e troca de implementação |

---

## 8. Testes

- **Domínio e use cases:** testados com *fakes* que satisfazem as portas por
  estrutura — rápidos, sem I/O.
- **Adapters pesados** (pandas, torch, fastapi): testes usam `pytest.importorskip`
  para pular quando a lib não está instalada.
- A separação hexagonal é o que torna o núcleo testável sem mocks complexos.

---

## 9. MLOps

- **MLflow:** rastreia hiperparâmetros e métricas por experimento; Model Registry para versões.
- **DVC:** versiona dados e artefatos; `dvc.yaml` define o pipeline reprodutível (`dvc repro`).
- **Docker:** `Dockerfile` empacota a API; `docker-compose.yml` sobe API + servidor MLflow.
- **Reprodutibilidade:** `seed_everything`, `uv.lock`, pipeline DVC.

---

## 10. Checklist para replicar em um novo projeto de ML

1. **Modele o domínio primeiro.** Quais são os conceitos-núcleo? Quais têm identidade
   (entities) e quais são valores (value objects)? Valide invariantes na construção.
2. **Defina as portas.** Do que o núcleo precisa do mundo? (ler dados, treinar, avaliar,
   rastrear, persistir). Escreva as interfaces antes das implementações.
3. **Escreva os casos de uso** dependendo só das portas. Teste com fakes.
4. **Implemente os adapters** (pandas/torch/mlflow) satisfazendo as portas.
5. **Sempre tenha um baseline** antes do modelo complexo — é a régua.
6. **Escolha a métrica certa para o objetivo.** Ranking ≠ regressão (MSE otimiza nota,
   não ordenação; para top-k, considere perdas de ranking).
7. **Cuide do split.** Em recomendação, split temporal **por usuário** evita cold start
   artificial e vazamento de futuro.
8. **Feche com MLOps:** rastreie experimentos, versione dados e modelo, containerize.
9. **Documente honestamente** (model card): inclusive quando o baseline vence.
