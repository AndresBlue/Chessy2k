# Opening book (Polyglot)

Chessy uses a Polyglot `.bin` book for fast human-like replies in the opening
when Stockfish or Reckless is selected with humanization enabled.

## Install

1. Download a Polyglot book such as `performance.bin` from
   [official-stockfish/books](https://github.com/official-stockfish/books).
2. Place it here as `data/openings/performance.bin`.

Path is configured in `config/default.yaml` under
`humanization.opening.book_path`.

Without a book file, humanization still works: the UCI engine is queried
normally and only instant book hits are skipped.
