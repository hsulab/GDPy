# Commit Guidelines

- Create commits only when the user explicitly asks for them.
- Keep each commit focused on one logical change.
- Do not combine code and documentation changes in the same commit. When a task changes both, create separate commits: one for code (including its tests) and one for documentation.
- Stage files selectively and review the staged diff before each commit so unrelated or user-authored changes are not included.
- Follow the repository's Conventional Commit-style subjects: `feat`, `fix`, `docs`, `test`, `refactor`, `style`, `perf`, `build`, `ci`, `chore`, and `revert`. Use the form `type(scope): description` when a scope is useful.

# Testing Guidelines

- Keep test time minimal. Run only the smallest test selection that directly covers the changed behavior.
- Prefer an individual test or test file over a package-wide suite. Expand coverage only when a shared interface or cross-cutting change makes broader regression testing necessary.
- Do not run the full test suite unless the user requests it, targeted tests expose a wider problem, or the change is sufficiently high-risk that narrower coverage is inadequate.
- Do not rerun already-passing tests after documentation-only, formatting-only, or otherwise unrelated edits.
- Validate examples, configuration, and documentation with the narrowest relevant parser, smoke test, or build command.
- Report the exact tests run and their results.
