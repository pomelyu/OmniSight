---
name: package-sorter
description: Automatically sort package dependencies alphabetically in package manager files. Triggers when editing or creating package manager files including requirements.txt, requirements-dev.txt, requirements.dev.txt, requirements_dev.txt, environment.yml, environment.yaml, package.json, Pipfile, pyproject.toml, Gemfile, go.mod, Cargo.toml, composer.json, and their dev/test variants (with -dev, .dev, _dev, -test, .test, _test prefixes or suffixes).
---

# Package Sorter

Sort package dependencies alphabetically in package manager files to improve readability and reduce merge conflicts.

## Supported File Formats

### Python
- **requirements.txt** - Standard Python dependencies
- **requirements-dev.txt, requirements.dev.txt, requirements_dev.txt** - Development dependencies
- **requirements-test.txt, requirements.test.txt, requirements_test.txt** - Test dependencies
- **environment.yml, environment.yaml** - Conda environment files
- **Pipfile** - Pipenv dependency file
- **pyproject.toml** - Modern Python project configuration

### JavaScript/Node.js
- **package.json** - Dependencies and devDependencies sections

### Ruby
- **Gemfile** - Ruby dependencies

### Go
- **go.mod** - Go module dependencies

### Rust
- **Cargo.toml** - Rust dependencies

### PHP
- **composer.json** - PHP dependencies

## Sorting Rules

### General Principles
1. **Preserve comments** - Keep inline and block comments with their associated packages
2. **Maintain sections** - Don't mix different dependency sections (e.g., dev vs. production)
3. **Case-insensitive sorting** - Sort alphabetically, treating uppercase and lowercase the same
4. **Preserve formatting** - Keep the same indentation, spacing, and style

### File-Specific Rules

#### requirements.txt and variants
- Sort package names alphabetically (case-insensitive)
- Keep version specifiers with their packages
- Preserve comments that appear above a package (treat as part of that package)
- Preserve inline comments (comments on the same line as a package)
- Keep blank lines between logical groups if they exist
- Don't sort lines starting with `-e` (editable installs), `-r` (include files), or `--` (pip options)

**Example:**
```
# Before
django>=4.0
# Authentication library
django-allauth==0.50.0
pytest==7.0
black==22.0

# After
black==22.0
# Authentication library
django-allauth==0.50.0
django>=4.0
pytest==7.0
```

#### environment.yml / environment.yaml (Conda)
- Sort packages under `dependencies:` section alphabetically
- Sort packages under `pip:` subsection alphabetically
- Keep pip packages as a subsection under dependencies
- Preserve channel specifications (e.g., `conda-forge::package`)

**Example:**
```yaml
# Before
dependencies:
  - python=3.9
  - numpy
  - pandas
  - pip:
    - django
    - black

# After
dependencies:
  - numpy
  - pandas
  - python=3.9
  - pip:
    - black
    - django
```

#### package.json
- Sort keys in `dependencies` object alphabetically
- Sort keys in `devDependencies` object alphabetically
- Sort keys in `peerDependencies` object alphabetically
- Don't change the order of other top-level keys

#### pyproject.toml
- Sort arrays under `[project.dependencies]` alphabetically
- Sort arrays under `[project.optional-dependencies.*]` alphabetically
- Sort arrays under `[tool.poetry.dependencies]` alphabetically
- Sort arrays under `[tool.poetry.dev-dependencies]` alphabetically

#### Pipfile
- Sort keys under `[packages]` alphabetically
- Sort keys under `[dev-packages]` alphabetically

#### Gemfile
- Sort `gem` declarations alphabetically by gem name
- Keep groups (`group :development`) separate and sort within groups
- Preserve source declarations

#### go.mod
- Sort `require` statements alphabetically
- Keep indirect dependencies separate if marked with `// indirect`

#### Cargo.toml
- Sort keys under `[dependencies]` alphabetically
- Sort keys under `[dev-dependencies]` alphabetically
- Sort keys under `[build-dependencies]` alphabetically

#### composer.json
- Sort keys in `require` object alphabetically
- Sort keys in `require-dev` object alphabetically

## Implementation

Use the bundled `scripts/sort_packages.py` script for robust sorting:

```bash
python scripts/sort_packages.py <file_path>
```

The script:
- Detects file type automatically
- Applies appropriate sorting rules
- Preserves comments and formatting
- Backs up the original file to `<file_path>.backup`
- Outputs the sorted result to the original file

For manual sorting when the script cannot be used:
1. Identify the file type and applicable section
2. Extract package lines (excluding comments, blank lines, and special directives)
3. Sort alphabetically (case-insensitive)
4. Preserve original formatting and comments
5. Reconstruct the file with sorted packages

## When to Apply

Apply sorting when:
- Creating a new package manager file
- Adding new dependencies to an existing file
- After merging changes that added dependencies
- User explicitly requests sorting
- File appears unsorted or randomly organized

Don't apply sorting when:
- User explicitly maintains a custom order for a reason
- Order has functional significance (rare, but possible in some edge cases)
- File is auto-generated with a specific order
