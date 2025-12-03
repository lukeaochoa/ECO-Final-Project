# Copilot Agent Performance Optimization Instructions

## Core Operating Principles

You are working on **complex, multi-step tasks requiring deep reasoning, extensive context understanding, and systematic execution**. Optimize for thoroughness, strategic tool usage, and complete task resolution.

---

## CRITICAL: Tool Usage Requirements

### **1. Task Planning & Tracking (MANDATORY for complex work)**
- **USE `manage_todo_list`** at the start of ANY multi-step task
- Break down requests into specific, actionable subtasks
- Mark tasks as "in-progress" when starting, "completed" immediately when done
- Provide visibility into progress throughout execution
- **Never skip this for tasks with >3 steps or unclear scope**

### **2. Memory & Context Persistence (USE PROACTIVELY)**
- **USE `memory` tool** to save:
  - Project-specific context, preferences, and patterns
  - Important decisions and their rationale
  - Recurring requirements or constraints
  - User's coding style, naming conventions, preferences
- **READ memory at session start** to understand historical context
- Update memory when learning new important information about the user/project

### **3. Parallel Execution (MAXIMIZE EFFICIENCY)**
- **ALWAYS batch independent operations together**
- Examples of parallelizable operations:
  - Multiple file reads
  - Multiple searches (codebase, textSearch, fileSearch)
  - Reading documentation + code samples simultaneously
  - Checking multiple file paths or dependencies
- **Never make sequential calls when parallel execution is possible**
- Exception: Terminal commands must run sequentially

### **4. Proactive Research & Documentation**
- **USE `microsoft_docs_search`** and **`microsoft_code_sample_search`** for:
  - Python/ML libraries, pandas, scikit-learn, etc.
  - Azure services and APIs
  - .NET, VS Code, and Microsoft technologies
- **USE `vscode_websearchforcopilot_webSearch`** for:
  - Latest package versions and compatibility
  - Recent ML techniques and best practices
  - Error messages and troubleshooting
  - Library documentation not in Microsoft docs
- **USE `fetch_webpage`** to retrieve full documentation when search results are incomplete
- **Research BEFORE implementing**, not after encountering errors

### **5. Complex Reasoning & Delegation**
- **USE `runSubagent`** for:
  - Complex research requiring multiple search iterations
  - Tasks needing specialized domain knowledge
  - When you need to explore multiple approaches simultaneously
  - Intensive code analysis across large codebases
- Provide detailed instructions to subagents including expected output format
- Subagents are stateless - give complete context in one prompt

### **6. Comprehensive Context Gathering**
- **USE `semantic_search`** (codebase) for conceptual/fuzzy searches
- **USE `grep_search`** (textSearch) for exact string/regex patterns
- **USE `file_search`** for finding files by name/path patterns
- **USE `list_code_usages`** to understand how functions/classes are used
- **READ broadly first** - get enough context before making changes
- Parallelize context gathering across different search types

### **7. Python-Specific Tools (Pylance MCP)**
- **USE `pylanceImports`** to analyze dependencies before suggesting installations
- **USE `pylanceFileSyntaxErrors`** to validate code before execution
- **USE `pylanceRunCodeSnippet`** to test code without creating temp files
- **USE `configurePythonEnvironment`** FIRST before any Python operations
- **USE `pylancePythonEnvironments`** to understand available environments

### **8. Notebook Operations**
- **ALWAYS USE `copilot_getNotebookSummary`** before editing notebooks
- **USE `run_notebook_cell`** to execute cells and validate changes
- **USE `read_notebook_cell_output`** for detailed output inspection
- **USE `configure_notebook`** before first cell execution in a session

### **9. Error Handling & Validation**
- **USE `get_errors`** after file edits to validate changes
- **USE `runTests`** to execute unit tests when available
- **USE `test_failure`** to include test context in debugging
- Check syntax/imports/dependencies BEFORE suggesting code changes

### **10. GitHub Integration (When Applicable)**
- **USE GitHub tools** for repository operations (don't use git commands unless necessary)
- Check `get_me` first to understand permissions
- Use `search_issues` and `search_pull_requests` before creating new ones
- Use `list_*` for broad retrieval, `search_*` for targeted queries

---

## Execution Workflow for Complex Tasks

```
1. READ MEMORY → Check for project context, preferences, patterns
2. CREATE TODO LIST → Break down task into trackable steps
3. GATHER CONTEXT (PARALLEL) → Search, read files, check docs simultaneously
4. RESEARCH (PROACTIVE) → Check docs, web search, code samples BEFORE implementing
5. PLAN APPROACH → Think through solution before writing code
6. EXECUTE INCREMENTALLY → Complete one todo at a time, mark completed immediately
7. VALIDATE → Run cells, check errors, test functionality
8. UPDATE MEMORY → Save important decisions, patterns, learnings
9. CONFIRM COMPLETION → Verify all todos completed
```

---

## Response Style for Complex Work

- **Be thorough, not brief** - Complex tasks deserve detailed explanations
- **Show your reasoning** - Explain WHY you're taking specific approaches
- **Provide progress updates** - Don't go silent during long operations
- **Ask clarifying questions** when requirements are ambiguous
- **Continue until complete** - Don't stop at first obstacle, research and proceed
- **Verify your work** - Run code, check outputs, validate functionality

---

## Tools Priority Matrix

### **ALWAYS USE (Every Session)**
- `memory` (read at start, write when learning)
- `configure_python_environment` (before Python work)
- `copilot_getNotebookSummary` (before notebook edits)
- `manage_todo_list` (for multi-step tasks)

### **USE FREQUENTLY (When Relevant)**
- Parallel tool calls (batch independent operations)
- `microsoft_docs_search` + `microsoft_code_sample_search`
- `websearch` (for current information)
- Search tools (semantic, grep, file)
- `get_errors` (after edits)

### **USE STRATEGICALLY (Complex Scenarios)**
- `runSubagent` (complex research, specialized tasks)
- `list_code_usages` (understanding codebase patterns)
- GitHub tools (repository operations)
- `fetch_webpage` (deep documentation needs)

### **USE AS NEEDED (Specific Cases)**
- Container tools (Docker/container work)
- VS Code API tools (extension development)
- Azure tools (cloud resources)
- Testing tools (validation)

---

## Anti-Patterns to AVOID

❌ **Sequential operations that could be parallel**
❌ **Making changes without gathering sufficient context**
❌ **Skipping todo lists for multi-step tasks**
❌ **Forgetting to check/update memory**
❌ **Implementing before researching best practices**
❌ **Running terminal commands for file operations**
❌ **Stopping at first error instead of researching solutions**
❌ **Not validating code after making changes**
❌ **Giving up on complex tasks without trying subagents**
❌ **Asking obvious questions answerable through tool usage**

---

## Quality Checklist

Before ending your turn, verify:
- [ ] All todos marked completed (if todo list was created)
- [ ] Memory updated with new learnings
- [ ] Code validated (run cells, check errors)
- [ ] Proper documentation/comments added
- [ ] No TODO/FIXME comments left in code without tracking
- [ ] User's actual question fully answered
- [ ] Proactive suggestions provided if relevant

---

## Example Prompt Addition

**Add this to your requests:**

> "Follow the instructions in COPILOT_AGENT_INSTRUCTIONS.md. This is a complex task requiring systematic execution with full tool utilization including memory, todo lists, parallel operations, proactive research, and subagents where appropriate."

Or simply:

> "Use full agent capabilities per COPILOT_AGENT_INSTRUCTIONS.md"

---

## Notes

- This file serves as a persistent instruction set
- Reference it at the start of complex work sessions
- Update it as you discover better workflows
- The goal is **complete task resolution**, not quick partial answers
