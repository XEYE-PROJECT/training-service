# Changelog

Formato [Keep a Changelog](https://keepachangelog.com/es/1.1.0/); versiones [SemVer](https://semver.org/lang/es/).
Las entradas nuevas van en "Unreleased"; `bash release.sh X.Y.Z` las convierte en una versión y
crea el tag que publica la GitHub Release, que es lo que hace que RunPod reconstruya el endpoint.

## [Unreleased]

### Añadido
- Worker de entrenamiento reconstruido: un contenedor por training (docker en local, RunPod
  Serverless en producción), mismo job JSON y mismo webhook en ambos; embeddings con
  sentence-transformers y enriquecimiento LLM (local, Groq o Gemini, con lotes y reintentos)
  cacheado por elemento.
- Callback final con reintentos, secreto del webhook solo por entorno (fail fast si falta o es
  de desarrollo contra un backend https), Sentry en los entrypoints.
- Calidad: ruff + mypy, pytest (worker, reporter, cli, handler de RunPod, contratos con el
  backend), gitleaks y Trivy en CI; releases por tag semver.

[Unreleased]: https://github.com/XEYE-PROJECT/training-service/compare/master...HEAD
