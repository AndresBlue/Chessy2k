[English](/README.md) / [Español](/docs/i18n/README_es.md)

Documentación detallada adicional aún en inglés en docs/.

# Chessy

Asistente de ajedrez de escritorio local: toma una captura de pantalla de un tablero digital, recupera el FEN con un modelo de visión y obtiene una sugerencia de jugada de **Stockfish**, **Reckless** o **Maia-3**.

```
Screenshot → board crop → 64-square classifier → FEN → UCI engine → suggested move + think time
```

Sin dependencia de la nube en el bucle principal. Los motores y el checkpoint de visión se incluyen en el repositorio (los pesos de Maia-3 se almacenan en caché local en la primera configuración).

## Características

- Superposición de escritorio Qt (`PySide6`) con captura de región del tablero y análisis automático
- Inferencia de visión a partir de un clasificador ResNet de casillas incluido
- Tres motores seleccionables en la interfaz:
  - **Stockfish** — fuerza completa o humanizado (`UCI_Elo` + selección MultiPV)
  - **Reckless** — motor UCI competitivo con la misma capa de humanización en Python
  - **Maia-3** — políticas neuronales de estilo humano a Elo configurable
- Indicaciones adaptativas de tiempo de pensamiento y libro de aperturas Polyglot opcional
- CLI y superficie FastAPI opcional para flujos de trabajo con FEN / captura de pantalla

## Requisitos

- Python 3.10+ (se recomienda 3.10 para builds Torch CUDA)
- Windows x86-64 con AVX2 para los binarios de motor incluidos (Linux/macOS: colocar binarios compatibles bajo `engines/`)
- Opcional: GPU NVIDIA para inferencia más rápida de visión / Maia-3

## Inicio rápido

```bash
# From the repository root
python -m pip install -e ".[dev]"

# Optional: install Maia-3 and cache weights into data/maia3/cache
python scripts/setup_maia3.py --model maia3-23m

# Desktop overlay
python -m src.app.ui_qt
# or on Windows: start_chessy_overlay.bat
```

### CLI

```bash
# FEN → Stockfish
python app.py --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# FEN → Reckless
python app.py --fen "..." --engine reckless

# Screenshot → move (uses data/checkpoints/vision/best.pt by default)
python app.py --image board.png --side white
```

Las rutas de los binarios se resuelven automáticamente desde `engines/stockfish/` y `engines/reckless/`. Se pueden sobrescribir con `--stockfish-path`, `--reckless-path` o las variables de entorno `CHESSY_STOCKFISH_PATH` / `CHESSY_RECKLESS_PATH` / `CHESSY_CLASSIFIER_CHECKPOINT`.

## Estructura del proyecto

| Path | Rol |
|------|-----|
| `src/app/` | Superposición UI, CLI, analizador, resolución de rutas |
| `src/vision/` | Inferencia captura de pantalla → FEN |
| `src/chess_core/` | Utilidades FEN, legalidad, seguimiento de partida |
| `src/engines/` | Stockfish, Reckless, Maia-3, humanización |
| `src/search/` | Libro de aperturas Polyglot |
| `engines/` | Binarios UCI incluidos |
| `third_party/maia3/` | Paquete Maia-3 incluido (o clonado por el script de setup) |
| `config/default.yaml` | Configuración por defecto (rutas relativas al repo) |
| `data/checkpoints/vision/best.pt` | Pesos de visión incluidos |
| `docs/` | Arquitectura, motores, configuración, desarrollo |

## Configuración

Consulta [Configuración](../CONFIGURATION.md) (EN). Notas específicas de motores y comportamiento de humanización: [Motores](../ENGINES.md) (EN).

## Pruebas

```bash
pytest tests/ -v
# Engine binary smoke tests
pytest tests/ -v -m integration
```

## Documentación

- [Arquitectura](../ARCHITECTURE.md) (EN)
- [Motores](../ENGINES.md) (EN)
- [Configuración](../CONFIGURATION.md) (EN)
- [Desarrollo](../DEVELOPMENT.md) (EN)
- [Contribuir](../../CONTRIBUTING.md)

## Licencia

El código de la aplicación está bajo MIT — ver [LICENSE](../../LICENSE).

Los motores empaquetados y de terceros conservan sus propias licencias:

| Componente | Licencia | Notas |
|------------|----------|-------|
| Stockfish | GPL-3.0 | Binario bajo `engines/stockfish/` |
| Reckless | AGPL-3.0 | Binario bajo `engines/reckless/` |
| Maia-3 | Upstream (CSSLab) | Ver `third_party/maia3/` |
| Checkpoint de visión | Proyecto (distribución MIT) | `data/checkpoints/vision/best.pt` |

Distribuir versiones modificadas de binarios AGPL/GPL puede imponer obligaciones de fuente para esos componentes. Chessy los invoca como procesos UCI separados y no los enlaza estáticamente.
