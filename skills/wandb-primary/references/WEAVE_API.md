# weave_api - High-Level Weave API for Agents

API for exploring Weave projects and analyzing evaluation results. Designed for agents.

## Quick Start

```python
from weave_tools.weave_api import init, Project

# Initialize against the remote trace server
init("entity/project")

project = Project.current()
print(project.summary())  # Start here - shows ops, objects, evals, feedback
```

## Initialization Options

````python
from weave_tools.weave_api import init

# Remote Weave server via HTTP (default)
init("entity/project")
init("entity/project", backend="http")

# Local SQLite database
init("entity/project", backend="sqlite", db_path="/path/to/weave.db")

```

## Method Reference

Cost indicators: `[fast]` = no network, `[query]` = single API call, `[loads]` = fetches data

### Project

```python
project = Project.current()  # [fast] Get current project from init()

# Discovery
project.summary() -> str                                    # [query] Overview of entire project
project.ops(name_contains=None) -> OpsView                  # [query] All traced functions
project.op(name) -> Op                                      # [fast] Single op by name (no version suffix)
project.objects(type=None, name_contains=None) -> ObjectsView  # [query] All objects
project.object(name) -> Object                              # [fast] Single object by name (no version suffix)
project.evals(limit=20) -> List[Eval]                       # [query] Recent evaluations (default 20!)
project.calls(op=None, limit=None) -> CallsView             # [query] Calls, optionally filtered by op
```

### Op (traced function, all versions)

```python
op = project.op("predict")

op.name -> str                          # [fast] Op name
op.versions() -> VersionsView           # [query] All versions
op.latest() -> OpVersion                # [query] Most recent version
op.calls() -> CallsView                 # [fast] View of calls (not loaded yet)
op.call_count() -> int                  # [query] Total calls across all versions
op.summary(shape_samples=50) -> str     # [loads] Detailed summary with I/O shapes
```

### OpVersion (specific version of an op)

```python
v = op.versions()["v3"]    # By version index
v = op.versions()["abc12"] # By digest prefix
v = op.latest()

v.name -> str              # [fast]
v.digest -> str            # [fast] Full digest
v.version_index -> int     # [fast] e.g., 3 for v3
v.ref() -> str             # [fast] Full weave:/// URI
v.short_ref() -> str       # [fast] "predict:abc123de"
v.code() -> Optional[str]  # [query] Python source code (may be None)
v.calls() -> CallsView     # [fast] Calls to this exact version
```

### Object (model, scorer, dataset - all versions)

```python
obj = project.object("MyScorer")

obj.name -> str                  # [fast]
obj.base_class -> Optional[str]  # [fast] "Scorer", "Model", etc.
obj.versions() -> VersionsView   # [query] All versions
obj.latest() -> ObjectVersion    # [query] Most recent version
obj.calls() -> CallsView         # [query] Calls referencing any version
obj.usage() -> str               # [loads] How this object is used
obj.summary() -> str             # [query] Overview with version list
```

### ObjectVersion (specific version of an object)

```python
v = obj.versions()["v5"]
v = obj.latest()

v.name -> str                    # [fast]
v.digest -> str                  # [fast]
v.version_index -> int           # [fast]
v.ref() -> str                   # [fast] Full weave:/// URI
v.short_ref() -> str             # [fast] "MyScorer:v5"
v.get() -> Any                   # [query] Fetch actual object data
v.get(expand_tables=True) -> Any # [loads] Expand table refs to row data
v.get(expand_tables=True, table_limit=100) -> Any  # [loads] Limit rows per table
v.shape(depth=3) -> ShapeSummary # [query] Structure of stored value
v.code() -> Optional[str]        # [query] Source code (if scorer/op, may be None)
v.calls() -> CallsView           # [fast] Calls referencing this version
```

### OpsView (collection of ops)

```python
ops = project.ops()
ops = project.ops(name_contains="predict")

len(ops) -> int                              # [fast] Count
ops[0] -> Op                                 # [fast] By index
ops["predict"] -> Op                         # [fast] By name
ops.filter(lambda op: op.call_count() > 10)  # [loads] Filter with predicate
ops.summary() -> str                         # [query] Table with call counts
```

### ObjectsView (collection of objects)

```python
objects = project.objects()
objects = project.objects(type="Scorer")           # Filter by base class
objects = project.objects(name_contains="Error")   # Filter by name

len(objects) -> int                                # [fast]
objects[0] -> Object                               # [fast] By index
objects["MyScorer"] -> Object                      # [fast] By name
objects.filter(predicate) -> ObjectsView           # [fast] Filter
objects.summary(include_usage=False) -> str        # [query] Grouped by type
objects.summary(include_usage=True) -> str         # [loads] Include call counts (slower)
```

### Eval (evaluation run)

```python
# Get evaluations
evals = project.evals(limit=100)  # Default is 20 - increase if needed!
eval = Eval.from_call_id("019b95e9-...")  # [query] Load specific eval

# Properties
eval.call_id -> str                    # [fast]
eval.root_call -> CallData             # [fast] The Evaluation.evaluate call
eval.model_ref -> Optional[str]        # [fast] Ref to model used
eval.dataset_ref -> Optional[str]      # [fast] Ref to dataset used

# Navigation
eval.predict_and_score_calls() -> CallsView  # [query] P&S calls
eval.model_calls() -> CallsView              # [query] Model calls (first child of each P&S)
eval.summarize() -> str                      # [loads] Full summary with shapes and feedback
```

### CallsView (lazy collection of calls)

```python
calls = project.calls(op="predict")
calls = eval.model_calls()
calls = CallsView.from_call_ids(["call_id_1", "call_id_2"])

# Counting (efficient - no data loading)
calls.count() -> int        # [query] Total count
calls.is_empty() -> bool    # [query] Fast existence check
calls.has_results() -> bool # [query] Opposite of is_empty()

# IMPORTANT: Use limit() before iterating on large datasets
calls.limit(100) -> CallsView   # [fast] Returns new view limited to N calls

# Accessing data (loads calls)
calls[0] -> CallData            # [loads] Single call
calls[:10] -> CallsView         # [loads] Slice (prefer limit() for efficiency)
list(calls) -> List[CallData]   # [loads] All calls - use limit() first!
for call in calls.limit(100):   # [loads] Iterate with limit
    ...

# Filtering (requires loaded data)
calls.filter(lambda c: c.status == "error") -> CallsView  # [loads]

# Shape analysis (loads data)
calls.output_shape(depth=3, sample=None) -> ShapeSummary  # [loads]
calls.input_shape(depth=3, sample=None) -> ShapeSummary   # [loads]
calls.feedback_shape(column, depth=3) -> ShapeSummary     # [loads]

# Feedback analysis
calls.feedback_columns() -> List[str]                # [loads] All feedback columns present
calls.feedback_summary(include_shapes=False) -> str  # [loads] Coverage table

# Scoring (writes feedback to the current backend)
calls.run_scorer(
    scorer,
    feedback_kwargs=None,
    max_concurrent=10,
) -> Iterator[Progress]  # [loads] Run a scorer on calls; writes feedback column
```

### CallData (single call)

```python
call = calls[0]
call = CallData(...)  # (Call is an alias for CallData)

# Identity
call.id -> str                    # [fast] Call ID
call.call_id -> str               # [fast] Alias for id
call.op_name -> str               # [fast] Full op ref
call.op_base_name -> str          # [fast] Just the name part
call.trace_id -> str              # [fast]
call.parent_id -> Optional[str]   # [fast]

# Data
call.inputs -> Dict[str, Any]     # [fast] Input arguments
call.output -> Any                # [fast] Return value
call.status -> str                # [fast] "success", "error", or "running"
call.exception -> Any             # [fast] Exception info if failed
call.duration_seconds -> Optional[float]  # [fast]

# Input access
call.input(key, default=None, expand=False) -> Any  # [fast, or query if expand=True]
# Example: call.input("instruction")

# Feedback
call.feedback(column, default=None) -> Any       # [query on first access, then cached]
call.feedback_meta(column) -> Optional[FeedbackMeta]  # [query] Full metadata
call.feedback_columns() -> List[str]             # [query] Columns on this call
call.feedback_keys() -> List[str]                # [query] Alias for feedback_columns

# Navigation
call.children(depth=1) -> List[CallData]         # [query] Direct children
call.first_child() -> Optional[CallData]         # [query] First child by time
call.descendants(op_name=None) -> List[CallData] # [query] All descendants, optionally filtered

# Code
call.op_code() -> Optional[str]  # [query] Source code of the op
```

### Progress (scorer run)

```python
progress.status  # "cached", "done", "error"
progress.error   # Optional[str], populated when status == "error"
progress.result  # Scorer result (if any)
progress.call_id
```

### Ref values (weave:/// refs)

```python
from weave_tools.weave_api import value_from_ref

# Object ref -> object value
obj = value_from_ref("weave:///entity/project/object/MyObj:abc123")

# Table ref -> list of rows
rows = value_from_ref("weave:///entity/project/table/abc123")

# Call ref -> CallData
call = value_from_ref("weave:///entity/project/call/019b...")
print(call.id, call.op_base_name)
```

`value_from_ref` always expands table refs inside objects.
It also caches resolved refs in memory to avoid repeat fetches.

### ShapeSummary

```python
shape = calls.output_shape()
print(shape)  # Pretty-printed tree showing structure, sizes, value distributions
shape.to_dict() -> Dict  # Export as nested dict
```

Example output:

```
Shape (50 samples, 2.3MB total):
  result (dict) 1.8MB 78.2%
    status (str) 50KB 2.2%  values: success=45, error=5
    score (number) 1KB 0.0%  range: [0.0, 1.0], mean: 0.72
```

## Common Workflows

### Explore an unfamiliar project

```python
project = Project.current()
print(project.summary())           # What's here?
print(project.ops().summary())     # What functions are traced?
print(project.objects().summary()) # What objects exist?
```

### Analyze an evaluation

```python
# Get evals - increase limit if you need more than 20!
evals = project.evals(limit=100)
print(f"Found {len(evals)} evaluations")

# Pick one and analyze
eval = evals[0]
print(eval.summarize())

# Dig into model calls
model_calls = eval.model_calls()
print(f"Total model calls: {model_calls.count()}")
print(model_calls.output_shape())
print(model_calls.feedback_summary())
```

### Find failed calls

```python
calls = project.calls(op="predict")
print(f"Total calls: {calls.count()}")

# Load a sample and filter
failed = calls.limit(1000).filter(lambda c: c.status == "error")
print(f"Failed in sample: {len(list(failed))}")

# Examine failures
for call in failed.limit(5):
    print(f"{call.id}: {call.exception}")
```

### Examine a scorer

```python
scorer = project.object("MyScorer")
print(scorer.summary())
print(scorer.latest().code())  # See the implementation
print(scorer.usage())          # Where is it used?
```

### Run a scorer on calls

Running scorers writes feedback to the initialized backend:

- `feedback_kwargs` maps scorer kwarg name -> feedback column name, and passes `call.feedback(column)` into the scorer.
- Results are stored under the scorer's name (`scorer.name` if set, otherwise the class name).

```python
import weave
from weave.flow.scorer import Scorer
from weave_tools.weave_api import Eval, Project, init

init("entity/project")  # or init(..., backend="sqlite", db_path="...")

class MyScorer(Scorer):
    name: str = "my_scorer_v1"

    @weave.op
    def score(self, output, *, prior=None):
        # output is call.output; prior is injected from feedback_kwargs below.
        return {"quality": "good" if output.get("success") else "bad"}

project = Project.current()
evals = project.evals(limit=1)
calls = evals[0].model_calls().limit(100)

# Map scorer kwargs to feedback columns: kwarg -> feedback column name.
for progress in calls.run_scorer(
    MyScorer(),
    feedback_kwargs={"prior": "some_existing_feedback"},
    max_concurrent=6,
):
    print(f"{progress.completed}/{progress.total}: {progress.status}")
    if progress.status == "error":
        print(progress.error)

# NOTE: run_scorer passes call.output and any feedback kwargs only.
# If you need inputs or trace data, pass them in via a feedback column or
# incorporate them into the model output first.
```

### Working with datasets (table refs)

Datasets store their rows as table refs (not inline data). Use `expand_tables=True` to fetch the actual rows:

```python
# Get a dataset object
dataset = project.object("my-dataset").latest()

# BAD: rows is a table ref string, not actual data
val = dataset.get()
print(val["rows"])  # "weave:///entity/project/table/abc123..."

# GOOD: expand table refs to get actual row data
val = dataset.get(expand_tables=True)
print(len(val["rows"]))  # 100
print(val["rows"][0])    # {"name": "...", "input": "...", ...}

# With limit (for large tables)
val = dataset.get(expand_tables=True, table_limit=10)  # Only fetch first 10 rows

# From an eval (dataset ref)
dataset = value_from_ref(eval.dataset_ref)
rows = dataset.get("rows", [])
# rows is a list of dicts; use rows[0].keys() to inspect
```

You can also fetch tables directly:

```python
from weave_tools.weave_api import fetch_table

table_ref = "weave:///entity/project/table/abc123..."
rows = fetch_table(table_ref, limit=100)
```

## Gotchas

### Default limits

- `project.evals(limit=20)` - Only returns 20 evals by default! Increase if needed.
- `project.calls()` - No default limit, but use `.limit()` before iterating.

### len() vs count()

```python
# BAD: Raises error on unloaded CallsView
len(calls)

# GOOD: Efficient count query
calls.count()
```

### Version suffixes

```python
# BAD: Will raise an error
project.op("predict:v3")

# GOOD: Use versions() to get specific version
project.op("predict").versions()["v3"]
```

### Object refs in call outputs

Many model outputs store large data as refs.
Avoid `weave.trace.refs.get` (it forces a full SDK login); use the helpers above:

```python
from weave_tools.weave_api import value_from_ref

output_a = value_from_ref(call.output['some_output'])
some_list = traj.get("some_list_attr", [])
print(some_list[:3])
```

### Call children are lists, not CallsView

```python
# call.children() returns a list, so use len() and list slicing.
kids = call.children()
print(len(kids))
first = kids[0] if kids else None
```

### Scorer name and Progress errors

- In Pydantic v2, define `name` with a type annotation: `name: str = "my_scorer"`.
- If you add class-level constants (e.g. `TAXONOMY`), annotate them as `ClassVar[...]`.
- Access `name` via an instance (e.g. `MyScorer().name`) to avoid `AttributeError` on the class.
- `Progress` has `.status`, `.error`, `.result` (no `.exception`).

### Avoid internal server methods

Use `Project`, `Eval`, and `CallsView` helpers. The `client.server.*` methods are
low-level and easy to mis-call.

### Feedback outputs vs scorer call refs

- `call.feedback("col")` returns the latest feedback output directly.
- `call.feedback_meta("col").scorer_call_ref` is a _call ref_ (use `value_from_ref`).

### run_scorer kwargs

`run_scorer` only supports `feedback_kwargs` (mapping scorer kwarg name -> feedback column).
It does not accept `scorer_kwargs`, `column_mapping`, or `additional_scorer_kwargs`.


### CallData vs CallsView feedback summaries

- `CallsView.feedback_summary(include_shapes=True)` works on call collections.
- `CallData.feedback_summary()` is for a single call and does not take `include_shapes`.

### Slicing vs limit()

```python
# INEFFICIENT: Fetches 20, discards 15
for eval in project.evals()[:5]:
    ...

# GOOD: Only fetches 5
for eval in project.evals(limit=5):
    ...
```
````
