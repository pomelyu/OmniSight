#!/usr/bin/env python3
"""
Sort packages in package manager files alphabetically.
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import List, Tuple


def sort_requirements_txt(content: str) -> str:
    """Sort requirements.txt style files."""
    lines = content.splitlines(keepends=True)
    result = []
    package_block = []
    
    for line in lines:
        stripped = line.strip()
        
        # Skip special pip directives and editable installs
        if stripped.startswith(('-e ', '-r ', '--', '#')) or not stripped:
            # Flush current package block if any
            if package_block:
                result.extend(sorted(package_block, key=lambda x: x.lower()))
                package_block = []
            result.append(line)
        else:
            # Collect package lines
            package_block.append(line)
    
    # Flush remaining packages
    if package_block:
        result.extend(sorted(package_block, key=lambda x: x.lower()))
    
    return ''.join(result)


def sort_conda_yaml(content: str) -> str:
    """Sort environment.yml/yaml files."""
    try:
        import yaml
    except ImportError:
        print("Warning: PyYAML not installed. Install with: pip install pyyaml", file=sys.stderr)
        return content
    
    try:
        data = yaml.safe_load(content)
        
        if 'dependencies' in data and isinstance(data['dependencies'], list):
            # Separate pip dependencies from conda dependencies
            pip_deps = None
            conda_deps = []
            
            for item in data['dependencies']:
                if isinstance(item, dict) and 'pip' in item:
                    pip_deps = item
                else:
                    conda_deps.append(item)
            
            # Sort conda dependencies
            conda_deps.sort(key=lambda x: str(x).lower())
            
            # Sort pip dependencies if present
            if pip_deps and isinstance(pip_deps['pip'], list):
                pip_deps['pip'].sort(key=lambda x: str(x).lower())
                conda_deps.append(pip_deps)
            
            data['dependencies'] = conda_deps
        
        return yaml.dump(data, default_flow_style=False, sort_keys=False)
    except Exception as e:
        print(f"Warning: Could not parse YAML: {e}", file=sys.stderr)
        return content


def sort_package_json(content: str) -> str:
    """Sort package.json dependencies."""
    try:
        data = json.loads(content)
        
        # Sort dependency sections
        for key in ['dependencies', 'devDependencies', 'peerDependencies', 'optionalDependencies']:
            if key in data and isinstance(data[key], dict):
                data[key] = dict(sorted(data[key].items(), key=lambda x: x[0].lower()))
        
        return json.dumps(data, indent=2, ensure_ascii=False) + '\n'
    except Exception as e:
        print(f"Warning: Could not parse JSON: {e}", file=sys.stderr)
        return content


def sort_pyproject_toml(content: str) -> str:
    """Sort pyproject.toml dependencies."""
    try:
        import tomli
        import tomli_w
    except ImportError:
        print("Warning: tomli/tomli_w not installed. Install with: pip install tomli tomli-w", file=sys.stderr)
        return content
    
    try:
        data = tomli.loads(content)
        
        # Sort project dependencies
        if 'project' in data:
            if 'dependencies' in data['project'] and isinstance(data['project']['dependencies'], list):
                data['project']['dependencies'].sort(key=lambda x: x.lower())
            
            if 'optional-dependencies' in data['project']:
                for group in data['project']['optional-dependencies']:
                    if isinstance(data['project']['optional-dependencies'][group], list):
                        data['project']['optional-dependencies'][group].sort(key=lambda x: x.lower())
        
        # Sort poetry dependencies
        if 'tool' in data and 'poetry' in data['tool']:
            for key in ['dependencies', 'dev-dependencies', 'group']:
                if key in data['tool']['poetry'] and isinstance(data['tool']['poetry'][key], dict):
                    data['tool']['poetry'][key] = dict(sorted(data['tool']['poetry'][key].items(), 
                                                              key=lambda x: x[0].lower()))
        
        return tomli_w.dumps(data)
    except Exception as e:
        print(f"Warning: Could not parse TOML: {e}", file=sys.stderr)
        return content


def sort_pipfile(content: str) -> str:
    """Sort Pipfile dependencies."""
    try:
        import toml
    except ImportError:
        print("Warning: toml not installed. Install with: pip install toml", file=sys.stderr)
        return content
    
    try:
        data = toml.loads(content)
        
        for key in ['packages', 'dev-packages']:
            if key in data and isinstance(data[key], dict):
                data[key] = dict(sorted(data[key].items(), key=lambda x: x[0].lower()))
        
        return toml.dumps(data)
    except Exception as e:
        print(f"Warning: Could not parse TOML: {e}", file=sys.stderr)
        return content


def sort_gemfile(content: str) -> str:
    """Sort Gemfile dependencies."""
    lines = content.splitlines(keepends=True)
    result = []
    gem_block = []
    in_group = False
    
    for line in lines:
        stripped = line.strip()
        
        if stripped.startswith('gem '):
            gem_block.append(line)
        else:
            # Flush gem block if any
            if gem_block:
                result.extend(sorted(gem_block, key=lambda x: x.lower()))
                gem_block = []
            result.append(line)
            
            # Track group blocks
            if 'group ' in stripped:
                in_group = True
            elif stripped == 'end' and in_group:
                in_group = False
    
    # Flush remaining gems
    if gem_block:
        result.extend(sorted(gem_block, key=lambda x: x.lower()))
    
    return ''.join(result)


def sort_cargo_toml(content: str) -> str:
    """Sort Cargo.toml dependencies."""
    try:
        import tomli
        import tomli_w
    except ImportError:
        print("Warning: tomli/tomli_w not installed. Install with: pip install tomli tomli-w", file=sys.stderr)
        return content
    
    try:
        data = tomli.loads(content)
        
        for key in ['dependencies', 'dev-dependencies', 'build-dependencies']:
            if key in data and isinstance(data[key], dict):
                data[key] = dict(sorted(data[key].items(), key=lambda x: x[0].lower()))
        
        return tomli_w.dumps(data)
    except Exception as e:
        print(f"Warning: Could not parse TOML: {e}", file=sys.stderr)
        return content


def sort_composer_json(content: str) -> str:
    """Sort composer.json dependencies."""
    return sort_package_json(content)  # Same format as package.json


def sort_go_mod(content: str) -> str:
    """Sort go.mod require statements."""
    lines = content.splitlines(keepends=True)
    result = []
    in_require = False
    require_block = []
    
    for line in lines:
        stripped = line.strip()
        
        if stripped.startswith('require ('):
            in_require = True
            result.append(line)
        elif in_require and stripped == ')':
            # Sort and flush require block
            if require_block:
                # Separate direct and indirect dependencies
                direct = [l for l in require_block if '// indirect' not in l]
                indirect = [l for l in require_block if '// indirect' in l]
                direct.sort(key=lambda x: x.lower())
                indirect.sort(key=lambda x: x.lower())
                result.extend(direct)
                result.extend(indirect)
                require_block = []
            result.append(line)
            in_require = False
        elif in_require:
            require_block.append(line)
        else:
            result.append(line)
    
    return ''.join(result)


def detect_and_sort(file_path: Path) -> str:
    """Detect file type and apply appropriate sorting."""
    content = file_path.read_text(encoding='utf-8')
    name = file_path.name.lower()
    
    # Match requirements.txt and variants
    if re.match(r'requirements.*\.txt$', name):
        return sort_requirements_txt(content)
    elif name in ['environment.yml', 'environment.yaml']:
        return sort_conda_yaml(content)
    elif name == 'package.json':
        return sort_package_json(content)
    elif name == 'pyproject.toml':
        return sort_pyproject_toml(content)
    elif name == 'pipfile':
        return sort_pipfile(content)
    elif name == 'gemfile':
        return sort_gemfile(content)
    elif name == 'cargo.toml':
        return sort_cargo_toml(content)
    elif name == 'composer.json':
        return sort_composer_json(content)
    elif name == 'go.mod':
        return sort_go_mod(content)
    else:
        print(f"Warning: Unknown file type: {name}", file=sys.stderr)
        return content


def main():
    parser = argparse.ArgumentParser(description='Sort package dependencies alphabetically')
    parser.add_argument('file_path', type=Path, help='Path to package manager file')
    parser.add_argument('--no-backup', action='store_true', help='Do not create backup file')
    
    args = parser.parse_args()
    
    if not args.file_path.exists():
        print(f"Error: File not found: {args.file_path}", file=sys.stderr)
        sys.exit(1)
    
    # Create backup
    if not args.no_backup:
        backup_path = args.file_path.with_suffix(args.file_path.suffix + '.backup')
        shutil.copy2(args.file_path, backup_path)
        print(f"Created backup: {backup_path}")
    
    # Sort and write
    sorted_content = detect_and_sort(args.file_path)
    args.file_path.write_text(sorted_content, encoding='utf-8')
    print(f"Sorted: {args.file_path}")


if __name__ == '__main__':
    main()
