#!/usr/bin/env python3
"""
code_quality_analyzer.py - Code quality analysis and refactoring recommendations
Analyze code complexity, maintainability, and provide actionable improvements
"""

import ast
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CodeMetrics:
    """Container for code quality metrics."""
    file_path: str
    lines_of_code: int = 0
    cyclomatic_complexity: int = 0
    function_count: int = 0
    class_count: int = 0
    max_function_length: int = 0
    avg_function_length: float = 0.0
    imports_count: int = 0
    documentation_coverage: float = 0.0
    code_duplication_score: float = 0.0
    maintainability_index: float = 0.0
    issues: List[str] = field(default_factory=list)


@dataclass
class QualityIssue:
    """Code quality issue with severity and recommendations."""
    file_path: str
    line_number: int
    issue_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    recommendation: str
    code_snippet: Optional[str] = None


class CodeComplexityAnalyzer:
    """Analyze code complexity using AST parsing."""
    
    def __init__(self):
        self.current_file = ""
        
    def analyze_file(self, file_path: Path) -> CodeMetrics:
        """Analyze a single Python file for complexity metrics."""
        self.current_file = str(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            metrics = CodeMetrics(file_path=str(file_path))
            metrics.lines_of_code = len([line for line in content.split('\n') if line.strip()])
            
            # Analyze AST
            self._analyze_ast(tree, metrics)
            
            # Calculate derived metrics
            self._calculate_derived_metrics(metrics, content)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return CodeMetrics(file_path=str(file_path))
    
    def _analyze_ast(self, tree: ast.AST, metrics: CodeMetrics) -> None:
        """Analyze AST for various metrics."""
        function_lengths = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                metrics.function_count += 1
                
                # Calculate function length
                if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
                    func_length = node.end_lineno - node.lineno + 1
                    function_lengths.append(func_length)
                    metrics.max_function_length = max(metrics.max_function_length, func_length)
                
                # Calculate cyclomatic complexity for this function
                complexity = self._calculate_cyclomatic_complexity(node)
                metrics.cyclomatic_complexity += complexity
                
                # Check for issues
                if func_length > 50:
                    metrics.issues.append(f"Long function '{node.name}' ({func_length} lines)")
                if complexity > 10:
                    metrics.issues.append(f"High complexity function '{node.name}' (complexity: {complexity})")
            
            elif isinstance(node, ast.ClassDef):
                metrics.class_count += 1
                
                # Check class size
                if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
                    class_length = node.end_lineno - node.lineno + 1
                    if class_length > 200:
                        metrics.issues.append(f"Large class '{node.name}' ({class_length} lines)")
            
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                metrics.imports_count += 1
        
        # Calculate average function length
        if function_lengths:
            metrics.avg_function_length = sum(function_lengths) / len(function_lengths)
    
    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity for a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            # Decision points increase complexity
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.Try):
                complexity += len(child.handlers)
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity
    
    def _calculate_derived_metrics(self, metrics: CodeMetrics, content: str) -> None:
        """Calculate derived quality metrics."""
        # Documentation coverage
        docstring_lines = len(re.findall(r'""".*?"""', content, re.DOTALL))
        comment_lines = len(re.findall(r'^\s*#', content, re.MULTILINE))
        total_doc_lines = docstring_lines + comment_lines
        
        if metrics.lines_of_code > 0:
            metrics.documentation_coverage = (total_doc_lines / metrics.lines_of_code) * 100
        
        # Maintainability Index (simplified version)
        # MI = 171 - 5.2 * ln(HV) - 0.23 * CC - 16.2 * ln(LOC)
        # Where HV=Halstead Volume, CC=Cyclomatic Complexity, LOC=Lines of Code
        if metrics.lines_of_code > 0 and metrics.cyclomatic_complexity > 0:
            import math
            log_loc = math.log(max(1, metrics.lines_of_code))
            log_cc = math.log(max(1, metrics.cyclomatic_complexity))
            
            # Simplified without Halstead Volume
            metrics.maintainability_index = max(0, 100 - (0.23 * metrics.cyclomatic_complexity) - (16.2 * log_loc))


class DuplicationDetector:
    """Detect code duplication across files."""
    
    def __init__(self):
        self.function_signatures = defaultdict(list)
        self.code_blocks = defaultdict(list)
    
    def analyze_duplication(self, files: List[Path]) -> Dict[str, List[Tuple[str, str]]]:
        """Analyze code duplication across multiple files."""
        duplications = defaultdict(list)
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                self._extract_patterns(tree, str(file_path))
                
            except Exception as e:
                logger.error(f"Error analyzing duplication in {file_path}: {e}")
        
        # Find duplications
        duplications.update(self._find_duplicate_functions())
        duplications.update(self._find_duplicate_blocks())
        
        return duplications
    
    def _extract_patterns(self, tree: ast.AST, file_path: str) -> None:
        """Extract function signatures and code blocks for comparison."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Create function signature
                args = [arg.arg for arg in node.args.args]
                signature = f"{node.name}({', '.join(args)})"
                self.function_signatures[signature].append(file_path)
            
            elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                # Extract code block patterns (simplified)
                block_hash = self._hash_ast_node(node)
                if block_hash:
                    self.code_blocks[block_hash].append((file_path, ast.unparse(node) if hasattr(ast, 'unparse') else str(node)))
    
    def _hash_ast_node(self, node: ast.AST) -> Optional[str]:
        """Create a hash for AST node structure."""
        try:
            if hasattr(ast, 'unparse'):
                code = ast.unparse(node)
                # Normalize code for comparison
                normalized = re.sub(r'\s+', ' ', code).strip()
                if len(normalized) > 50:  # Only check substantial blocks
                    return hash(normalized)
        except:
            pass
        return None
    
    def _find_duplicate_functions(self) -> Dict[str, List[str]]:
        """Find functions with identical signatures."""
        duplicates = {}
        for signature, files in self.function_signatures.items():
            if len(files) > 1:
                duplicates[f"Duplicate function: {signature}"] = files
        return duplicates
    
    def _find_duplicate_blocks(self) -> Dict[str, List[str]]:
        """Find duplicate code blocks."""
        duplicates = {}
        for block_hash, instances in self.code_blocks.items():
            if len(instances) > 1:
                files = [instance[0] for instance in instances]
                duplicates[f"Duplicate code block (hash: {abs(block_hash) % 10000})"] = files
        return duplicates


class CodeQualityAnalyzer:
    """Main code quality analyzer."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.complexity_analyzer = CodeComplexityAnalyzer()
        self.duplication_detector = DuplicationDetector()
        self.issues = []
    
    def analyze_project(self) -> Dict[str, Any]:
        """Perform comprehensive code quality analysis."""
        logger.info("Starting comprehensive code quality analysis")
        
        # Find all Python files
        python_files = list(self.project_root.glob("**/*.py"))
        python_files = [f for f in python_files if not any(exclude in str(f) for exclude in ['.venv', '__pycache__', '.git'])]
        
        logger.info(f"Analyzing {len(python_files)} Python files")
        
        # Analyze individual files
        file_metrics = {}
        for file_path in python_files:
            metrics = self.complexity_analyzer.analyze_file(file_path)
            file_metrics[str(file_path)] = metrics
        
        # Analyze duplication
        duplications = self.duplication_detector.analyze_duplication(python_files)
        
        # Generate overall project metrics
        project_metrics = self._calculate_project_metrics(file_metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(file_metrics, duplications)
        
        return {
            'project_metrics': project_metrics,
            'file_metrics': file_metrics,
            'duplications': duplications,
            'recommendations': recommendations,
            'files_analyzed': len(python_files)
        }
    
    def _calculate_project_metrics(self, file_metrics: Dict[str, CodeMetrics]) -> Dict[str, Any]:
        """Calculate aggregate project metrics."""
        if not file_metrics:
            return {}
        
        total_loc = sum(m.lines_of_code for m in file_metrics.values())
        total_functions = sum(m.function_count for m in file_metrics.values())
        total_classes = sum(m.class_count for m in file_metrics.values())
        total_complexity = sum(m.cyclomatic_complexity for m in file_metrics.values())
        
        avg_maintainability = sum(m.maintainability_index for m in file_metrics.values()) / len(file_metrics)
        avg_doc_coverage = sum(m.documentation_coverage for m in file_metrics.values()) / len(file_metrics)
        
        # Find most complex files
        complex_files = sorted(
            [(path, metrics.cyclomatic_complexity) for path, metrics in file_metrics.items()],
            key=lambda x: x[1], reverse=True
        )[:5]
        
        # Find largest files
        large_files = sorted(
            [(path, metrics.lines_of_code) for path, metrics in file_metrics.items()],
            key=lambda x: x[1], reverse=True
        )[:5]
        
        return {
            'total_lines_of_code': total_loc,
            'total_functions': total_functions,
            'total_classes': total_classes,
            'total_complexity': total_complexity,
            'average_maintainability_index': avg_maintainability,
            'average_documentation_coverage': avg_doc_coverage,
            'most_complex_files': complex_files,
            'largest_files': large_files,
            'files_with_issues': len([m for m in file_metrics.values() if m.issues])
        }
    
    def _generate_recommendations(self, file_metrics: Dict[str, CodeMetrics], 
                                duplications: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations for code improvement."""
        recommendations = []
        
        # High complexity recommendations
        for file_path, metrics in file_metrics.items():
            if metrics.cyclomatic_complexity > 50:
                recommendations.append({
                    'type': 'complexity',
                    'severity': 'high',
                    'file': file_path,
                    'description': f'High cyclomatic complexity ({metrics.cyclomatic_complexity})',
                    'recommendation': 'Consider breaking down complex functions into smaller, focused functions'
                })
            
            if metrics.max_function_length > 100:
                recommendations.append({
                    'type': 'function_length',
                    'severity': 'medium',
                    'file': file_path,
                    'description': f'Very long function ({metrics.max_function_length} lines)',
                    'recommendation': 'Break down large functions into smaller, more manageable pieces'
                })
            
            if metrics.documentation_coverage < 10:
                recommendations.append({
                    'type': 'documentation',
                    'severity': 'medium',
                    'file': file_path,
                    'description': f'Low documentation coverage ({metrics.documentation_coverage:.1f}%)',
                    'recommendation': 'Add docstrings and comments to improve code documentation'
                })
            
            if metrics.maintainability_index < 50:
                recommendations.append({
                    'type': 'maintainability',
                    'severity': 'high',
                    'file': file_path,
                    'description': f'Low maintainability index ({metrics.maintainability_index:.1f})',
                    'recommendation': 'Refactor to reduce complexity and improve code structure'
                })
        
        # Duplication recommendations
        for duplication, files in duplications.items():
            recommendations.append({
                'type': 'duplication',
                'severity': 'medium',
                'files': files,
                'description': duplication,
                'recommendation': 'Extract common functionality into shared utilities or base classes'
            })
        
        # Sort by severity
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        recommendations.sort(key=lambda x: severity_order.get(x['severity'], 3))
        
        return recommendations
    
    def generate_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate a comprehensive quality report."""
        report = []
        report.append("# CODE QUALITY ANALYSIS REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Project overview
        metrics = analysis_results['project_metrics']
        report.append("## PROJECT OVERVIEW")
        report.append(f"Total Lines of Code: {metrics['total_lines_of_code']:,}")
        report.append(f"Total Functions: {metrics['total_functions']:,}")
        report.append(f"Total Classes: {metrics['total_classes']:,}")
        report.append(f"Total Complexity: {metrics['total_complexity']:,}")
        report.append(f"Average Maintainability Index: {metrics['average_maintainability_index']:.1f}")
        report.append(f"Average Documentation Coverage: {metrics['average_documentation_coverage']:.1f}%")
        report.append(f"Files Analyzed: {analysis_results['files_analyzed']}")
        report.append(f"Files with Issues: {metrics['files_with_issues']}")
        report.append("")
        
        # Most complex files
        report.append("## MOST COMPLEX FILES")
        for file_path, complexity in metrics['most_complex_files']:
            relative_path = os.path.relpath(file_path, self.project_root)
            report.append(f"- {relative_path}: {complexity} complexity")
        report.append("")
        
        # Largest files
        report.append("## LARGEST FILES")
        for file_path, loc in metrics['largest_files']:
            relative_path = os.path.relpath(file_path, self.project_root)
            report.append(f"- {relative_path}: {loc} lines")
        report.append("")
        
        # Recommendations
        report.append("## RECOMMENDATIONS")
        recommendations = analysis_results['recommendations']
        
        by_severity = defaultdict(list)
        for rec in recommendations:
            by_severity[rec['severity']].append(rec)
        
        for severity in ['critical', 'high', 'medium', 'low']:
            if severity in by_severity:
                report.append(f"### {severity.upper()} PRIORITY")
                for rec in by_severity[severity]:
                    if 'file' in rec:
                        file_path = os.path.relpath(rec['file'], self.project_root)
                        report.append(f"- **{file_path}**: {rec['description']}")
                    else:
                        report.append(f"- **Multiple files**: {rec['description']}")
                    report.append(f"  *Recommendation*: {rec['recommendation']}")
                report.append("")
        
        # Duplications
        duplications = analysis_results['duplications']
        if duplications:
            report.append("## CODE DUPLICATIONS")
            for duplication, files in duplications.items():
                report.append(f"### {duplication}")
                for file_path in files:
                    relative_path = os.path.relpath(file_path, self.project_root)
                    report.append(f"- {relative_path}")
                report.append("")
        
        return "\n".join(report)


def main():
    """Main analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Code Quality Analyzer")
    parser.add_argument("--project-root", type=Path, default=Path("."),
                       help="Project root directory")
    parser.add_argument("--output", type=Path, default=Path("code_quality_report.md"),
                       help="Output report file")
    parser.add_argument("--json", action="store_true",
                       help="Also output results in JSON format")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run analysis
    analyzer = CodeQualityAnalyzer(args.project_root)
    results = analyzer.analyze_project()
    
    # Generate report
    report = analyzer.generate_report(results)
    
    # Save report
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Code quality report saved to: {args.output}")
    
    # Save JSON if requested
    if args.json:
        import json
        json_output = args.output.with_suffix('.json')
        
        # Convert metrics to serializable format
        serializable_results = {
            'project_metrics': results['project_metrics'],
            'duplications': results['duplications'],
            'recommendations': results['recommendations'],
            'files_analyzed': results['files_analyzed'],
            'file_metrics': {
                file_path: {
                    'lines_of_code': metrics.lines_of_code,
                    'cyclomatic_complexity': metrics.cyclomatic_complexity,
                    'function_count': metrics.function_count,
                    'class_count': metrics.class_count,
                    'max_function_length': metrics.max_function_length,
                    'avg_function_length': metrics.avg_function_length,
                    'imports_count': metrics.imports_count,
                    'documentation_coverage': metrics.documentation_coverage,
                    'maintainability_index': metrics.maintainability_index,
                    'issues': metrics.issues
                }
                for file_path, metrics in results['file_metrics'].items()
            }
        }
        
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"JSON results saved to: {json_output}")
    
    # Print summary
    print("\n" + "="*50)
    print("CODE QUALITY ANALYSIS SUMMARY")
    print("="*50)
    print(f"Files analyzed: {results['files_analyzed']}")
    print(f"Total lines of code: {results['project_metrics']['total_lines_of_code']:,}")
    print(f"Average maintainability: {results['project_metrics']['average_maintainability_index']:.1f}")
    print(f"Documentation coverage: {results['project_metrics']['average_documentation_coverage']:.1f}%")
    print(f"Total recommendations: {len(results['recommendations'])}")
    
    # Show severity breakdown
    from collections import Counter
    severity_counts = Counter(rec['severity'] for rec in results['recommendations'])
    for severity in ['critical', 'high', 'medium', 'low']:
        if severity in severity_counts:
            print(f"  {severity}: {severity_counts[severity]}")


if __name__ == "__main__":
    main()