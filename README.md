# Kotlin MCP

Local MCP server that summarizes Kotlin and Java source files into compact structural representations, so AI agents consume fewer tokens when understanding a codebase.

## Tech Stack

- **Language**: Python 3.11+
- **MCP SDK**: `mcp[cli]` (official Anthropic)
- **Parser**: `tree-sitter` with `tree-sitter-kotlin` and `tree-sitter-java`
- **Package manager**: `uv`

## Installation

### From GitHub (recommended)

```bash
# Run directly (no install needed)
uvx --from "git+https://github.com/fordewe/fojako.git" kotlin-mcp

# Pin to a specific version
uvx --from "git+https://github.com/fordewe/fojako.git@v0.1.0" kotlin-mcp

# Or install persistently as a global CLI tool
uv tool install "git+https://github.com/fordewe/fojako.git"
```

### From source

```bash
git clone https://github.com/fordewe/fojako.git
cd fojako
uv tool install .
```

After installation, `kotlin-mcp` is available globally from any terminal.

## Register with Claude Code

### Option 1: `.mcp.json` (per-project, auto-setup)

Add a `.mcp.json` file at the root of your project:

```json
{
  "mcpServers": {
    "kotlin-summarizer": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/fordewe/fojako.git", "kotlin-mcp"]
    }
  }
}
```

Claude Code will automatically detect and start the server when you open the project.

### Option 2: Manual registration

```bash
claude mcp add kotlin-summarizer -- kotlin-mcp
```

## MCP Tools

### `summarize_file_tool`

Summarize a single `.kt` or `.java` file into its structural representation.

**Input:**
```json
{ "path": "/absolute/path/to/File.kt" }
```

**Output:**
```
File: UserRepository.kt
Package: com.example.data.repository

Classes:
  @Inject class UserRepository(
    private val dao: UserDao,
    private val api: ApiService
  )
    + suspend getUser(id: String): Flow<User>
    + suspend updateUser(user: User): Result<Unit>
    - buildQuery(filter: Filter): Query  ← private

Imports (external):
  - kotlinx.coroutines.flow.Flow
  - javax.inject.Inject
```

### `summarize_module_tool`

Recursively summarize all `.kt` and `.java` files under a directory.

**Input:**
```json
{
  "path": "/absolute/path/to/module/",
  "depth": 2
}
```

- `depth`: directory levels to recurse (default: unlimited)

**Output:** Per-file summaries grouped by package, plus a module-level overview.

**Auto-save:** The summary is automatically saved to `{path}/.kotlin-summary/{module}.md` so it can be read directly in future conversations without calling this tool again. Add `.kotlin-summary/` to your project's `.gitignore`.

To make summaries discoverable by the LLM, add this to your project's `CLAUDE.md`:

```markdown
## Codebase Summary
Read files in `.kotlin-summary/` for structural overview of Kotlin/Java modules.
```

## Skipped Files

- Generated files: `BuildConfig.kt`, `R.java`, `*Binding.kt`
- Test files: `*Test.kt`, `*Spec.kt`, `*Test.java`, `*Spec.java`

## Evaluation artifacts

This repository also holds the tasks, responses, scores, and scripts behind the paper *Structural Code Summarization as a Context Strategy for LLM-Assisted Development: A Task-Based Evaluation under a Token Budget*.

| Path | Contents |
|---|---|
| `evaluation/datasets.md`, `evaluation/datasets.json` | The five Android projects and how to clone them |
| `evaluation/tasks.json` | The 15 task templates and the scoring rubric |
| `evaluation/results/rq2_tasks.json` | The 71 tasks instantiated from those templates (seed 42) |
| `evaluation/results/rq1_metrics.csv` | Per-file compression metrics (RQ1) |
| `evaluation/results/rq2_clean_*` | The isolated run the paper reports: 135 responses, scores from both LLM judges, and the human validation scores |
| `evaluation/results/rq2_responses.json`, `rq2_judge_*`, `rq2_validation_sample.json`, `rq2_human_scores.csv` | The first run, which the paper discusses under threats to validity |
| `evaluation/*.py` | Scripts that instantiate the tasks, collect responses, run the judges, and compute the statistics |

## License

The code is released under the MIT License (see `LICENSE`). The evaluation data (`evaluation/tasks.json`, `evaluation/datasets.json`, and everything in `evaluation/results/`) is released under the Open Data Commons Attribution License v1.0 (see `LICENSE-DATA`). Code excerpts from the five evaluated projects that appear in the tasks or responses remain under those projects' own licenses.

## Citation

See `CITATION.cff`, or use GitHub's "Cite this repository" button.
