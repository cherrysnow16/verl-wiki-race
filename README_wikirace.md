# WikiRace

WikiRace는 에이전트가 시작 위키백과 페이지에서 링크를 눌러 목표 페이지까지 도달하는 환경이다. 매 스텝마다 현재 페이지의 링크 후보가 번호와 함께 주어지고, 에이전트는 그 번호로 하나를 고른다. Ray 기반 병렬 환경으로 감싸져 있어 GRPO·GiGPO의 group rollout을 지원한다.

게임 엔진은 verl-agent 트리에 이미 내장(vendoring)되어 있어 별도 설치가 필요 없다.
아래 순서대로 따라가면 새 환경에서 GiGPO/GRPO 학습까지 돌릴 수 있다.

1. [veRL 설치](#1-verl-설치)
2. [WikiGraph 데이터 내려받기](#2-wikigraph-데이터-내려받기)
3. [환경 점검](#3-환경-점검)
4. [학습 실행](#4-학습-실행-gigpo--grpo)

---

## 1. veRL 설치

WikiRace 환경은 verl-agent 위에서 돈다. 먼저 본체를 설치한다.

```bash
conda create -n verl-agent python==3.12 -y
conda activate verl-agent

pip3 install vllm==0.11.0
pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -e .          # verl-agent 저장소 루트에서
```

더 자세한 옵션은 저장소 메인 README의 Installation을 참고한다.

## 2. WikiGraph 데이터 내려받기

게임 엔진(`LLM_wikirace`)과 splits CSV는 `agent_system/environments/env_package/wikirace/vendor/`
아래에 이미 내장돼 있다. 별도 clone·설치 없이 verl-agent만 받으면 된다.

다만 위키백과 그래프 데이터(약 103MB, 549,232개 페이지)는 용량이 커서 저장소에 넣지 않는다.
HuggingFace에서 받아 vendored 데이터 폴더 아래에 푼다.

```bash
# verl-agent 저장소 루트에서
cd agent_system/environments/env_package/wikirace/vendor

hf download juliusz-ziomek/LLM-WikiRace-Benchmark wikidata.zip --repo-type dataset --local-dir data/
python -c "import zipfile; zipfile.ZipFile('data/wikidata.zip').extractall('data/wikidata/')"
```

풀고 나면 아래 두 파일이 놓인다. `load_wikigraph()`가 이걸 읽어 그래프를 만든다.

```text
env_package/wikirace/vendor/data/wikidata/
  wiki_graph_sparse_filtered.npz
  id_to_title_filtered.json.gz
```

그래프를 다른 곳에 두고 싶으면 `env.wikirace.wikigraph_path`로 경로를 넘기면 된다.

> 참고 — 연동 구조:
> ```text
> agent_system/environments/env_package/wikirace/
>   envs.py / projection.py / __init__.py   # verl-agent 래퍼
>   vendor/src/LLM_wikirace/                 # 내장 게임 엔진
>   vendor/data/splits/ , vendor/data/wikidata/
> agent_system/environments/prompts/wikirace.py   # prompt 템플릿
> agent_system/environments/env_manager.py        # make_envs() 분기
> verl/trainer/config/ppo_trainer.yaml            # env.wikirace 기본값
> ```

## 3. 환경 점검

학습을 돌리기 전에 환경이 로드되고 스텝이 도는지 먼저 확인한다.

```bash
python examples/gigpo_trainer/sample_eval_wikirace.py --episodes 2 --steps 3
```

WikiGraph를 로드하고, env를 reset한 뒤 action 0으로 몇 번 스텝한다. 정상이면 이렇게 나온다.

```text
Loading WikiGraph...

Reset
[0] game=... 'start page' -> 'target page'
    0: ...
    1: ...

projection demo: actions=[0, 0], valids=[1, 0]

Step
step 1
[0] reward=0.0 done=False status=in progress page='...'
```

## 4. 학습 실행 (GiGPO / GRPO)

점검이 끝나면 학습을 돌린다.

```bash
bash examples/gigpo_trainer/run_wikirace.sh
```

스크립트는 먼저 더미 텍스트 parquet을 만들고(`examples.data_preprocess.prepare --mode text`, 실제 관측은 env에서 나온다) 이어서 `verl.trainer.main_ppo`를 실행한다. 주요 설정은 다음과 같다.

```bash
algorithm.adv_estimator=gigpo \        # GRPO로 돌리려면 grpo
algorithm.gamma=0.95 \                 # long-horizon 할인율
algorithm.gigpo.step_advantage_w=1.0 \ # step-level advantage 가중치 (GiGPO 전용)
env.env_name=wikirace/easy \           # make_envs()는 "wikirace" 부분 문자열로 라우팅
env.wikirace.difficulty=easy \         # split 선택 (data/splits/easy.csv)
env.max_steps=30 \                     # episode 최대 스텝 (= max_path_length)
env.rollout.n=8 \                      # 같은 게임 복제 수 = group size
env.history_length=2 \                 # prompt에 넣을 방문 페이지 수
env.wikirace.max_choices=50 \          # 페이지당 링크 후보 최대 수
actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
trainer.val_only=False                 # 학습. True면 학습 없이 validation만.
```

기본 스크립트는 학습 없이 validation만 도는 값(`trainer.val_only=True`, `env.rollout.n=1`)으로 맞춰져 있다. 실제로 학습을 돌리려면 스크립트에서 두 값을 바꾼다.

- `trainer.val_only=False` — 학습을 켠다.
- `env.rollout.n`을 1보다 크게(예: 8) 둔다. GRPO·GiGPO는 같은 상태에서 rollout을 여러 번 만들어 advantage를 그룹 안에서 정규화하기 때문에, group size가 1이면 학습 효과가 없다.

학습이 시작되면 매 스텝마다 아래 같은 로그가 찍힌다. `episode/reward`·`episode/success_rate`로 에이전트가 목표 페이지에 얼마나 도달하는지 볼 수 있다.

```text
step:1 - actor/pg_loss:... - actor/grad_norm:... - actor/kl_loss:...
         - episode/valid_action_ratio:... - episode/reward/mean:...
         - episode/success_rate:... - episode/length/mean:...
```

# How It Works

한 step:
```text
env.reset() -> GameState(current_page, target_page, available_choices, history)
  -> WikiRaceEnvironmentManager.build_text_obs()   # prompt 생성
  -> 모델이 응답 생성
  -> wikirace_projection()                         # action 번호 추출
  -> envs.step()                                   # 선택한 링크로 이동
  -> reward, done, info
```

## **Action.**
모델은 링크 제목이 아니라 번호를 출력한다. prompt는 `Reply with the number of your choice (0 to N)`으로 끝나고, 링크는 번호로 나열된다:
```text
0. United States
1. History
2. Mathematics
```

## **Projection.**
`wikirace_projection(actions, neighbors_list)`는 모델 응답에서 클릭할 링크 번호를 뽑는다. 방식은 원본 `LLM_wikirace.parse_answer`와 같고, 위에서부터 먼저 맞는 규칙을 쓴다.

1. 응답 속 숫자 중 실제 링크 번호로 쓸 수 있는 값, 여러 개면 마지막 숫자 사용
2. 쓸 만한 숫자가 없으면, 응답에 나온 링크 제목 중 가장 뒤에 언급된 것의 번호
3. 그래도 없으면 0 사용

`valid`은 1번 규칙으로 뽑았을 때만 1이고 나머지는 0이다. invalid action penalty에 쓰인다.
```python
# nb = ["United States","History","Mathematics","Science","Art"]  (범위 0~4)
wikirace_projection(["I choose 2"],                 [nb])  # ([2], [1])
wikirace_projection(["go 7 no, 3"],                 [nb])  # ([3], [1])  # 범위 안 마지막 숫자
wikirace_projection(["click History then Science"], [nb])  # ([3], [0])  # 마지막 언급 제목
wikirace_projection(["I don't know"],               [nb])  # ([0], [0])
```
`neighbors_list`는 `WikiRaceEnvironmentManager`가 현재 states의 `available_choices`로 넘긴다. 이후 envs 단계에서도 범위 밖 index는 crash 없이 0으로 보정한다.

## **Reward.**

| 조건 | reward | done |
| --- | --- | --- |
| 목표 도착 (`status == "completed"`) | `10.0` | True |
| 실패 (`status == "failed"`) | `0.0` | True |
| 진행 중 (`in progress`) | `0.0` | False |

성공은 `info["won"]`(= `status == "completed"`)로 노출된다.

## **Batch / group.**

환경은 두 축으로 늘어난다.

- `env_num` — 배치에 담기는 서로 다른 게임 수. train은 `data.train_batch_size`, val은 `data.val_batch_size`.
- `group_n` — 같은 게임을 몇 개로 복제할지. `env.rollout.n` 값이며 GRPO/GiGPO 그룹 rollout에 쓴다.

`reset()`은 게임 `env_num`개를 뽑아 각각 `group_n`번 복제하고, `step()`은 `env_num × group_n`개의 정수 action을 리스트로 받는다.

## **Split.**

게임을 train/val로 나누는 방식.

- `get_possible_game_ids(difficulty)`로 게임 목록을 얻어 `train_ratio` 지점에서 자른다. 앞쪽이 train, 뒤쪽이 val.
- `val_train_ratio=0.0`이면 val이 전체 게임을 쓴다.

WikiGraph는 `ray.put(load_wikigraph())`로 한 번만 로드해서 train/val 환경이 `wikigraph_ref`로 공유한다.
