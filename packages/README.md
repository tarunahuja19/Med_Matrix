# 📦 Shared Packages Workspace (`packages/`)

This directory houses internal, shared packages used across the MedMatrix monorepo application layer. Storing shared components in a single workspace ensures type safety and centralized configuration.

---

## 🗺️ Shared Packages Navigation

| Package Folder | Package Name | Purpose |
| :--- | :--- | :--- |
| **[`config/`](file:///home/jemin/Projects/Med_Matrix/packages/config)** | `config` | Shared build, TypeScript, and linting rule configurations. |
| **[`shared-types/`](file:///home/jemin/Projects/Med_Matrix/packages/shared-types)** | `@kvision/shared-types` | Unified TypeScript interfaces, API schemas, and Electron IPC channels. |

---

## 🚀 Workspace Setup

These packages are managed as local dependencies using **pnpm workspaces**. When dependencies are installed, local package directories are symlinked under node modules, allowing instant updates during development.

### References in `apps/`:
In `apps/backend/package.json` or `apps/electron/package.json`, they are referenced as:
```json
"dependencies": {
  "@kvision/shared-types": "workspace:*"
}
```
This ensures both the backend and frontend are locked to the exact same types.
