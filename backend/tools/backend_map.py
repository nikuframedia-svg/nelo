#!/usr/bin/env python3
"""
Backend Audit Tool - Contract 15 Phase 1
Maps all endpoints, services, engines and generates audit report.
"""

import os
import ast
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

BACKEND_ROOT = Path(__file__).parent.parent

# ═══════════════════════════════════════════════════════════════════════════════
# AST VISITORS
# ═══════════════════════════════════════════════════════════════════════════════

class EndpointVisitor(ast.NodeVisitor):
    """Finds FastAPI route decorators."""
    
    def __init__(self):
        self.endpoints = []
        self.current_file = ""
    
    def visit_FunctionDef(self, node):
        for decorator in node.decorator_list:
            route_method = None
            route_path = None
            
            # Handle @router.get("/path") or @app.get("/path")
            if isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Attribute):
                    if decorator.func.attr in ('get', 'post', 'put', 'delete', 'patch'):
                        route_method = decorator.func.attr.upper()
                        if decorator.args:
                            if isinstance(decorator.args[0], ast.Constant):
                                route_path = decorator.args[0].value
            
            if route_method and route_path:
                self.endpoints.append({
                    'method': route_method,
                    'path': route_path,
                    'function': node.name,
                    'file': self.current_file,
                    'line': node.lineno
                })
        
        self.generic_visit(node)


class ClassVisitor(ast.NodeVisitor):
    """Finds class definitions."""
    
    def __init__(self):
        self.classes = []
        self.current_file = ""
    
    def visit_ClassDef(self, node):
        bases = [self._get_base_name(b) for b in node.bases]
        methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
        
        self.classes.append({
            'name': node.name,
            'bases': bases,
            'methods': methods,
            'file': self.current_file,
            'line': node.lineno
        })
        self.generic_visit(node)
    
    def _get_base_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return str(node)


class FunctionVisitor(ast.NodeVisitor):
    """Finds function definitions and their calls."""
    
    def __init__(self):
        self.functions = []
        self.calls = defaultdict(list)
        self.current_file = ""
        self.current_function = None
    
    def visit_FunctionDef(self, node):
        self.functions.append({
            'name': node.name,
            'file': self.current_file,
            'line': node.lineno,
            'args': [a.arg for a in node.args.args]
        })
        
        old_func = self.current_function
        self.current_function = f"{self.current_file}::{node.name}"
        self.generic_visit(node)
        self.current_function = old_func
    
    def visit_Call(self, node):
        if self.current_function:
            call_name = self._get_call_name(node)
            if call_name:
                self.calls[self.current_function].append(call_name)
        self.generic_visit(node)
    
    def _get_call_name(self, node):
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None


class ImportVisitor(ast.NodeVisitor):
    """Finds imports."""
    
    def __init__(self):
        self.imports = []
        self.current_file = ""
    
    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append({
                'module': alias.name,
                'alias': alias.asname,
                'file': self.current_file,
                'type': 'import'
            })
    
    def visit_ImportFrom(self, node):
        module = node.module or ''
        for alias in node.names:
            self.imports.append({
                'module': module,
                'name': alias.name,
                'alias': alias.asname,
                'file': self.current_file,
                'type': 'from'
            })


# ═══════════════════════════════════════════════════════════════════════════════
# SCANNER
# ═══════════════════════════════════════════════════════════════════════════════

def scan_python_files(root: Path) -> List[Path]:
    """Find all Python files in backend."""
    files = []
    for path in root.rglob("*.py"):
        # Skip __pycache__, .git, venv, etc.
        if any(part.startswith(('.', '__')) or part in ('venv', 'env', 'node_modules') 
               for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def parse_file(filepath: Path) -> Tuple[ast.AST, str]:
    """Parse a Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        return ast.parse(source, filename=str(filepath)), source
    except Exception as e:
        return None, f"Error: {e}"


def analyze_backend():
    """Main analysis function."""
    print("=" * 80)
    print("BACKEND AUDIT - Contract 15 Phase 1")
    print("=" * 80)
    
    py_files = scan_python_files(BACKEND_ROOT)
    print(f"\nFound {len(py_files)} Python files to analyze.\n")
    
    # Collectors
    all_endpoints = []
    all_classes = []
    all_functions = []
    all_calls = defaultdict(list)
    all_imports = []
    parse_errors = []
    
    # Domain categorization
    domains = defaultdict(list)
    
    for filepath in py_files:
        rel_path = filepath.relative_to(BACKEND_ROOT)
        domain = rel_path.parts[0] if len(rel_path.parts) > 1 else 'root'
        domains[domain].append(str(rel_path))
        
        tree, source = parse_file(filepath)
        if tree is None:
            parse_errors.append((str(rel_path), source))
            continue
        
        # Endpoints
        ev = EndpointVisitor()
        ev.current_file = str(rel_path)
        ev.visit(tree)
        all_endpoints.extend(ev.endpoints)
        
        # Classes
        cv = ClassVisitor()
        cv.current_file = str(rel_path)
        cv.visit(tree)
        all_classes.extend(cv.classes)
        
        # Functions
        fv = FunctionVisitor()
        fv.current_file = str(rel_path)
        fv.visit(tree)
        all_functions.extend(fv.functions)
        for k, v in fv.calls.items():
            all_calls[k].extend(v)
        
        # Imports
        iv = ImportVisitor()
        iv.current_file = str(rel_path)
        iv.visit(tree)
        all_imports.extend(iv.imports)
    
    # Generate report
    report = generate_report(
        domains, all_endpoints, all_classes, all_functions, 
        all_calls, all_imports, parse_errors
    )
    
    # Save report
    report_path = BACKEND_ROOT / "tools" / "backend_audit_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✅ Report saved to: {report_path}")
    return report


def generate_report(domains, endpoints, classes, functions, calls, imports, errors):
    """Generate the audit report."""
    lines = []
    lines.append("# Backend Audit Report - Contract 15")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total Python files**: {sum(len(v) for v in domains.values())}")
    lines.append(f"- **Total endpoints**: {len(endpoints)}")
    lines.append(f"- **Total classes**: {len(classes)}")
    lines.append(f"- **Total functions**: {len(functions)}")
    lines.append(f"- **Parse errors**: {len(errors)}")
    lines.append("")
    
    # Domains
    lines.append("## Domains")
    lines.append("")
    for domain, files in sorted(domains.items()):
        lines.append(f"### {domain} ({len(files)} files)")
        for f in sorted(files)[:10]:
            lines.append(f"  - `{f}`")
        if len(files) > 10:
            lines.append(f"  - ... and {len(files) - 10} more")
        lines.append("")
    
    # Endpoints by domain
    lines.append("## API Endpoints")
    lines.append("")
    endpoints_by_domain = defaultdict(list)
    for ep in endpoints:
        domain = ep['file'].split('/')[0] if '/' in ep['file'] else 'root'
        endpoints_by_domain[domain].append(ep)
    
    for domain, eps in sorted(endpoints_by_domain.items()):
        lines.append(f"### {domain}")
        for ep in sorted(eps, key=lambda x: x['path']):
            lines.append(f"- `{ep['method']} {ep['path']}` → `{ep['function']}()` in `{ep['file']}`")
        lines.append("")
    
    # Key Classes (engines, services, models)
    lines.append("## Key Classes")
    lines.append("")
    
    engine_classes = [c for c in classes if 'Engine' in c['name'] or 'Estimator' in c['name']]
    service_classes = [c for c in classes if 'Service' in c['name']]
    model_classes = [c for c in classes if any(b in c['bases'] for b in ['BaseModel', 'Base', 'SQLModel'])]
    
    lines.append("### Engines & Estimators")
    for c in sorted(engine_classes, key=lambda x: x['name']):
        lines.append(f"- `{c['name']}` in `{c['file']}` (bases: {', '.join(c['bases']) or 'none'})")
    lines.append("")
    
    lines.append("### Services")
    for c in sorted(service_classes, key=lambda x: x['name']):
        lines.append(f"- `{c['name']}` in `{c['file']}`")
    lines.append("")
    
    lines.append("### Models (Pydantic/SQLAlchemy)")
    for c in sorted(model_classes[:30], key=lambda x: x['name']):
        lines.append(f"- `{c['name']}` in `{c['file']}`")
    if len(model_classes) > 30:
        lines.append(f"- ... and {len(model_classes) - 30} more models")
    lines.append("")
    
    # WP/WPX Detection
    lines.append("## R&D Work Packages")
    lines.append("")
    
    wp_files = [f for files in domains.values() for f in files if 'wp' in f.lower() or 'rd' in f.lower()]
    wpx_classes = [c for c in classes if 'WP' in c['name'] or 'Experiment' in c['name']]
    
    lines.append("### WP Files")
    for f in sorted(set(wp_files)):
        lines.append(f"- `{f}`")
    lines.append("")
    
    lines.append("### WP/Experiment Classes")
    for c in sorted(wpx_classes, key=lambda x: x['name']):
        lines.append(f"- `{c['name']}` in `{c['file']}`")
    lines.append("")
    
    # Feature Coverage Check
    lines.append("## Feature Coverage Check")
    lines.append("")
    
    features = {
        'Scheduling/APS': ['scheduling', 'heuristic', 'milp', 'cpsat', 'gantt'],
        'SmartInventory': ['inventory', 'mrp', 'rop', 'forecast', 'abc_xyz', 'stock'],
        'Duplios/PDM': ['duplios', 'pdm', 'dpp', 'lca', 'trust_index', 'gap_filling', 'compliance'],
        'Digital Twin': ['digital_twin', 'shi_dt', 'xai_dt', 'cvae', 'rul', 'deviation'],
        'Prevention Guard': ['prevention', 'guard', 'poka_yoke', 'validation'],
        'R&D': ['rd', 'experiments', 'wp1', 'wp2', 'wp3', 'wp4', 'wpx'],
        'Ops Ingestion': ['ops_ingestion', 'excel', 'ingest'],
        'Work Instructions': ['work_instruction', 'shopfloor', 'checklist'],
        'Causal/Intelligence': ['causal', 'optimization', 'what_if'],
    }
    
    all_files_lower = [f.lower() for files in domains.values() for f in files]
    all_classes_lower = [c['name'].lower() for c in classes]
    all_functions_lower = [f['name'].lower() for f in functions]
    
    for feature, keywords in features.items():
        found_files = sum(1 for kw in keywords for f in all_files_lower if kw in f)
        found_classes = sum(1 for kw in keywords for c in all_classes_lower if kw in c)
        found_functions = sum(1 for kw in keywords for fn in all_functions_lower if kw in fn)
        
        status = "✅" if found_files > 0 or found_classes > 0 else "❌"
        lines.append(f"- {status} **{feature}**: {found_files} files, {found_classes} classes, {found_functions} functions")
    
    lines.append("")
    
    # Parse Errors
    if errors:
        lines.append("## Parse Errors")
        lines.append("")
        for f, e in errors:
            lines.append(f"- `{f}`: {e}")
        lines.append("")
    
    # Potential Dead Code (functions defined but possibly not called)
    lines.append("## Potential Dead Code (needs manual verification)")
    lines.append("")
    
    called_functions = set()
    for caller, callees in calls.items():
        called_functions.update(callees)
    
    endpoint_functions = {ep['function'] for ep in endpoints}
    
    potentially_dead = []
    for fn in functions:
        if fn['name'].startswith('_'):
            continue  # Skip private
        if fn['name'] in ('__init__', '__str__', '__repr__', 'main'):
            continue
        if fn['name'] in called_functions:
            continue
        if fn['name'] in endpoint_functions:
            continue
        potentially_dead.append(fn)
    
    for fn in sorted(potentially_dead[:50], key=lambda x: x['file']):
        lines.append(f"- `{fn['name']}()` in `{fn['file']}` (line {fn['line']})")
    
    if len(potentially_dead) > 50:
        lines.append(f"- ... and {len(potentially_dead) - 50} more")
    
    lines.append("")
    lines.append("---")
    lines.append("*Report generated by backend_map.py - Contract 15*")
    
    return '\n'.join(lines)


if __name__ == '__main__':
    analyze_backend()


