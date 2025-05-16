# AMAOS Repository Refactor Plan

This document outlines the plan for cleaning and reorganizing the AMAOS repository structure to reduce redundancy, improve maintainability, and ensure a clear architecture.

## Current Issues

The current repository structure has several issues:

1. **Redundant Directories**: Multiple deploy directories, empty sandbox directories
2. **Obsolete Test Files**: Multiple versions of orchestrator tests
3. **Duplicated Testing**: Potential duplicates like test_browser_tool.py and test_browser_tools.py
4. **Root Directory Clutter**: Test files and utility scripts in the repository root
5. **Legacy Files**: Outdated JSON files and example blueprints

## Cleanup Plan

### Files and Directories to Remove

| File/Folder | Reason | Action |
|-------------|--------|--------|
| `sandbox/` | Empty folder, redundant | Delete |
| `amaos/sandbox/` | Empty folder, not used | Delete |
| `tests/a.zip` | Test artifact | Delete |

### Files and Directories to Move

| Source | Destination | Reason |
|--------|-------------|--------|
| `tests/test_orchestrator.py` | `tests/legacy/test_orchestrator.py` | Superseded by refactored version |
| `tests/test_orchestrator_patched.py` | `tests/legacy/test_orchestrator_patched.py` | Interim test version |
| `7_Node_Agent_Blueprint_With_Examples.json` | `docs/drafts/7_Node_Agent_Blueprint_With_Examples.json` | Draft document |
| `AMAOS99.zip` | `archive/AMAOS99.zip` | Archive file |
| `test.png` and `test.txt` | `tests/assets/` | Test assets |
| `create_zip.py` | `scripts/utils/create_zip.py` | Utility script |
| Unique files from `amaos/deploy/` | `deploy/` | Consolidate deployment files |

### Files Requiring Manual Review

| File | Concern | Recommendation |
|------|---------|----------------|
| `tests/test_browser_tool.py` and `tests/test_browser_tools.py` | Potential duplication | Review both and determine which to keep |

## Directory Structure Before and After

### Before

```
/
├── amaos/
│   ├── deploy/           ← Duplicate deploy directory
│   ├── sandbox/          ← Empty directory
├── deploy/
├── sandbox/              ← Empty directory
├── tests/
│   ├── test_orchestrator.py            ← Original version
│   ├── test_orchestrator_patched.py    ← Patched version
│   ├── test_orchestrator_refactored.py ← Current version
│   ├── a.zip                           ← Test artifact
├── 7_Node_Agent_Blueprint_With_Examples.json   ← Draft document
├── AMAOS99.zip          ← Archive file
├── create_zip.py        ← Utility script
├── test.png             ← Test asset
├── test.txt             ← Test asset
```

### After

```
/
├── amaos/
├── deploy/               ← Consolidated deploy directory
├── tests/
│   ├── assets/           ← Directory for test assets
│   │   ├── test.png
│   │   ├── test.txt
│   ├── legacy/           ← Directory for legacy tests
│   │   ├── test_orchestrator.py
│   │   ├── test_orchestrator_patched.py
│   ├── test_orchestrator_refactored.py
├── scripts/
│   ├── utils/            ← Directory for utility scripts
│   │   ├── create_zip.py
├── docs/
│   ├── drafts/           ← Directory for draft documents
│   │   ├── 7_Node_Agent_Blueprint_With_Examples.json
├── archive/              ← Directory for archival files
│   ├── AMAOS99.zip
```

## Long-term Repository Organization Recommendations

To maintain a clean and organized repository structure in the long term, we recommend:

### 1. Consistent Directory Structure

- **scripts/**: For utility scripts and tools not part of the core package
  - **scripts/dev/**: Development-only utilities
  - **scripts/deploy/**: Deployment scripts
  - **scripts/utils/**: General utility scripts

- **tests/**: Well-organized test directory
  - **tests/unit/**: Unit tests for individual components
  - **tests/integration/**: Tests that verify component integration
  - **tests/e2e/**: End-to-end tests of the full system
  - **tests/assets/**: Test files and fixtures
  - **tests/legacy/**: Outdated tests preserved for reference

- **docs/**: Documentation and examples
  - **docs/api/**: API documentation
  - **docs/tutorials/**: Step-by-step guides
  - **docs/drafts/**: Work-in-progress documentation

### 2. Plugin Management

- **plugins_enabled/**: Active plugins
- **plugins_disabled/**: Inactive or experimental plugins

### 3. Version Control Best Practices

- Use `.gitignore` to exclude generated files, logs, and temporary files
- Maintain clean branch structure (main, develop, feature branches)
- Enforce code reviews for structural changes

### 4. Continuous Maintenance

- Regular scheduled cleanup sprints (quarterly)
- Automate linting and code organization checks
- Document technical debt and prioritize refactoring

### 5. Testing Strategy

- Maintain test independence
- Avoid duplicate tests with different names
- Keep test data isolated from production code

## Implementation Timeline

1. **Phase 1: Initial Cleanup** - Execute the cleanup script to perform basic organization
2. **Phase 2: Test Refactoring** - Review and consolidate test files, ensuring all tests pass
3. **Phase 3: Documentation Update** - Update documentation to reflect new structure
4. **Phase 4: CI/CD Integration** - Update CI/CD pipelines to work with new structure

## Conclusion

This refactoring plan will help ensure the AMAOS codebase remains maintainable and well-organized. The cleanup script provides a safe way to make these changes with full backup capabilities.
