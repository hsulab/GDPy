# Commit Guidelines

- Create commits only when the user explicitly asks for them.
- Keep each commit focused on one logical change.
- Do not combine code and documentation changes in the same commit. When a task changes both, create separate commits: one for code (including its tests) and one for documentation.
- Stage files selectively and review the staged diff before each commit so unrelated or user-authored changes are not included.
- Follow the repository's Conventional Commit-style subjects: `feat`, `fix`, `docs`, `test`, `refactor`, `style`, `perf`, `build`, `ci`, `chore`, and `revert`. Use the form `type(scope): description` when a scope is useful.
