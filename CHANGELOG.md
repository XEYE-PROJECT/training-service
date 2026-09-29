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
- Datos hacia el LLM (sección D): opt-out por lista (`list.llm_enrichment=false` salta el paso
  LLM), tope de tokens de salida y temperatura configurados en Groq/Gemini (antes fijos),
  medidor de tokens con tarifa y tope de gasto por job (`LLM_MAX_COST_PER_JOB`), y coste real
  (`cost.llm`, `cost.runpod`) más `usage` en el webhook `completed`.
- Seguridad del job: token de webhook por entrenamiento (`webhook_token`, HMAC del backend) en
  lugar de un secreto en el entorno; validación de `callback_url` (`https://` salvo
  `CALLBACK_ALLOW_HTTP`, allowlist `CALLBACK_ALLOWED_HOSTS`) y allowlist de modelos de
  embeddings (`EMBEDDING_MODELS_ALLOWED`, fijada en las imágenes).
- Imágenes: usuario `xeye` sin root, `torch` fijado también en la GPU, GGUF verificado por
  SHA-256 y modelos de sentence-transformers por revisión, `HF_HUB_OFFLINE=1` en ejecución,
  `.dockerignore`.

### Cambiado
- `WEBHOOK_SECRET` desaparece del worker (y del endpoint de RunPod): el backend ya no comparte su secreto.

[Unreleased]: https://github.com/XEYE-PROJECT/training-service/compare/master...HEAD
