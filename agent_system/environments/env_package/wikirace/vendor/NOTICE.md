# Vendored WikiRace engine

`src/LLM_wikirace/`의 게임 엔진(config, data_utils, game_engine, wikirace_env)과
`data/splits/`의 스플릿 CSV는 LLM-WikiRace-Benchmark 에서 가져와 verl-agent 트리에
내장(vendoring)한 것이다. 다른 환경(sokoban, webshop 등)과 같은 구성으로, 별도 설치
없이 verl-agent 안에서 바로 돌리기 위함이다.

- 출처: https://github.com/JuliuszZiomek/LLM-WikiRace-Benchmark (MIT License)
- 포함 파일: 학습 env 실행에 필요한 최소 모듈만. 추론/평가용(llm_inference_engine,
  wikirace_evaluator 등)은 제외.
- 위키백과 그래프 데이터(`wiki_graph_sparse_filtered.npz`, `id_to_title_filtered.json.gz`)는
  용량이 커서 포함하지 않는다. `data/wikidata/` 아래에 별도로 내려받는다 (README 참고).
