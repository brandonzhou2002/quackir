# QuackIR: Dense Retrieval with Flock + Ollama

This short guide shows how to run a minimal dense-retrieval pipeline with the Flock extension and a local Ollama embedding model.
It assumes you’ve followed the NFCorpus setup in this [experiments guide](./experiments-nfcorpus.md).

## Install and start Ollama
+ Download and install Ollama from the [download page](https://ollama.com/download).
+ Ensure the service is running locally (default: `127.0.0.1:11434`). Start it with:

```bash
OLLAMA_NUM_PARALLEL=2 \
OLLAMA_MAX_QUEUE=2048 \
ollama serve &
```

The `ollama serve &` command starts the Ollama server in the background.
The environment variables control concurrency and memory usage.
If you encounter `server busy, please try again. maximum pending requests exceeded` error while running the script, consider increasing the `OLLAMA_MAX_QUEUE` value, or reducing the `batch_size` parameter in the `options_json` field when registering the model.
Tune these settings based on your hardware capacity and workload.

+ Pull an embedding model:

```bash
ollama pull embeddinggemma
```

## Full Python walkthrough
Paste the following Python script into a file and execute it.

```python
from quackir.flock import FlockManager
from quackir.index import DuckDBIndexer
from quackir.search import DuckDBSearcher
from quackir import IndexType
from pathlib import Path
import csv

"""
1) Configuration
    - Set table names, paths, and embedding dimension for the model you'll use.
"""
table_name = "corpus_dense"
corpus_file = "collections/nfcorpus/quackir_corpus.jsonl"
embedding_dim = 768
model_alias = "Embedder"
queries_file = "collections/nfcorpus/queries.tsv"
output_path = Path("runs/run.quackir.duckdb.dense.flock.nfcorpus.txt")
top_k = 10

"""
2) Initialize Flock + register model alias
    - Loads the extension and creates the model alias bound to the local Ollama model.
"""
flock = FlockManager()
flock.create_model(
    alias=model_alias,
    provider_model="embeddinggemma",
    provider="ollama",
    options_json='{"tuple_format":"json", "batch_size":16}',
)

"""
3) Create dense table schema
    - A DuckDB table with (id VARCHAR, embedding DOUBLE[embedding_dim]).
"""
indexer = DuckDBIndexer(flock_manager=flock)
indexer.init_table(table_name, IndexType.DENSE, embedding_dim=embedding_dim)

"""
4) Populate embeddings directly from the corpus file
    - with_flock=True invokes Flock to compute an embedding per row and insert into the table.
"""
indexer.load_table(
    table_name,
    corpus_file,
    with_flock=True,
    id_column="id",
    contents_column="contents",
    embedding_dim=embedding_dim,
)
indexer.close()

"""
5) Embed queries on-the-fly and write a TREC run file
    - embedding_search(..., with_flock=True) computes the query embedding via Flock, then scores by cosine similarity.
"""
searcher = DuckDBSearcher(flock_manager=flock)

with output_path.open("w") as out, open(queries_file) as f:
    reader = csv.reader(f, delimiter="\t")
    for qid, qtext in reader:
        hits = searcher.embedding_search(
            query_embedding=qtext,  # raw query text; Flock generates the vector
            top_n=top_k,
            table_name=table_name,
            with_flock=True,
            embedding_dim=embedding_dim,
        )
        for rank, (docid, score) in enumerate(hits, start=1):
            out.write(f"{qid} Q0 {docid} {rank} {score:.6f} QuackIR\n")

searcher.close()
```

## Evaluate with trec_eval

```bash
python -m pyserini.eval.trec_eval \
    -c -m ndcg_cut.10 collections/nfcorpus/qrels/test.qrels \
    runs/run.quackir.duckdb.dense.flock.nfcorpus.txt
```
