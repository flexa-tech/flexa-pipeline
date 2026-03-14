# Quick Task 001: Automate session progress capture to Obsidian 00-Inbox - Context

**Gathered:** 2026-03-14
**Status:** Ready for planning

<domain>
## Task Boundary

Build a lightweight automation that writes structured notes to the Obsidian vault's `Claude/00-Inbox/` after each GSD task completes. Pulls from GSD's SUMMARY.md, STATE.md, and session context. Auto-indexes via brain-index.sh. Replaces the need for claude-mem.

</domain>

<decisions>
## Implementation Decisions

### Trigger Timing
- Write notes after GSD task completion (quick tasks and phase executions)
- Not session-end hooks — per-task granularity

### Relationship with Changelogs
- Coexist: Inbox = automatic, granular, per-task captures. Changelogs = curated, manual, per-session handoffs
- Both serve different purposes, no replacement

### Implementation Mechanism
- Shell script (`~/.claude/vault-capture.sh`) called by GSD workflows after executor completes
- Clean separation from GSD internals, easy to debug
- Script reads SUMMARY.md, extracts key info, writes structured note to vault, runs brain-index.sh

### Claude's Discretion
- Note format/frontmatter structure
- How to extract and compress SUMMARY.md content
- Whether to run brain-index.sh synchronously or in background

</decisions>

<specifics>
## Specific Ideas

- Vault path: `~/.claude/ObsidianVault/Claude/00-Inbox/`
- Notes should have YAML frontmatter (date, project, task type, branch, tags)
- Include: task description, what was built, key files, decisions made
- Link back to .planning/ artifacts where relevant
- Re-index vault after writing: `~/.claude/brain-index.sh`

</specifics>
