# ⚙️ Config (`packages/config`)

This package is a shared space for linter, formatter, and compiler configurations within the MedMatrix monorepo. It serves to enforce stylistic and syntax rules across all frontend and backend TypeScript/JavaScript files.

---

## 🗺️ File Map & Details

### 1. [`package.json`](file:///home/jemin/Projects/Med_Matrix/packages/config/package.json)
* **Purpose:** Package definition. It exports the module structure under the workspace name `config`.
* **Details:** Uses modern `"type": "module"` configuration to support ESM imports across configs.

---

## ⚙️ How it is Used

Sub-projects within the monorepo reference configurations directly from this package in their configs (e.g. `eslint.config.js`, `tsconfig.json`, or `.prettierrc`). 

This setup allows developers to maintain consistent configurations across components, such as:
- Shared ESLint rule-sets.
- Base TypeScript configuration targets (e.g. `tsconfig.json` extending root files).
- Global Prettier rules.
