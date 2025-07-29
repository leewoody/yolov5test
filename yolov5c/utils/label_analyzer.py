#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Label format analyzer for YOLOv5 datasets
Analyzes and validates label formats across different datasets
"""

import os
import glob
import argparse
import numpy as np
from pathlib import Path
import ast
import yaml


def analyze_label_format(label_file):
    """
    Analyze a single label file format
    
    Args:
        label_file: Path to label file
        
    Returns:
        dict: Analysis results
    """
    try:
        with open(label_file, 'r') as f:
            lines = f.read().strip().splitlines()
        
        if not lines:
            return {
                'file': label_file,
                'status': 'empty',
                'detection_lines': 0,
                'classification_line': None,
                'detection_format': None,
                'classification_format': None,
                'error': 'Empty file'
            }
        
        # Analyze detection labels (all lines except last)
        detection_lines = lines[:-1] if len(lines) > 1 else lines
        classification_line = lines[-1] if len(lines) > 1 else None
        
        # Analyze detection format
        detection_format = 'unknown'
        detection_count = 0
        
        if detection_lines:
            detection_count = len(detection_lines)
            first_line = detection_lines[0].split()
            
            if len(first_line) == 5:
                # Standard YOLO format: class x_center y_center width height
                detection_format = 'yolo_standard'
            elif len(first_line) > 6:
                # Segmentation format: class x1 y1 x2 y2 x3 y3 ...
                detection_format = 'segmentation'
            else:
                detection_format = 'invalid'
        
        # Analyze classification format
        classification_format = 'none'
        if classification_line:
            if classification_line.startswith('[') and classification_line.endswith(']'):
                # Python list format: [0, 0, 1]
                try:
                    parsed = ast.literal_eval(classification_line)
                    if isinstance(parsed, list):
                        classification_format = 'python_list'
                    else:
                        classification_format = 'invalid'
                except (ValueError, SyntaxError):
                    classification_format = 'invalid'
            else:
                # Space-separated format: 0 0 1
                parts = classification_line.split()
                if all(part.replace('.', '').replace('-', '').isdigit() or part in ['0', '1'] for part in parts):
                    classification_format = 'space_separated'
                else:
                    classification_format = 'invalid'
        
        return {
            'file': label_file,
            'status': 'valid',
            'detection_lines': detection_count,
            'classification_line': classification_line,
            'detection_format': detection_format,
            'classification_format': classification_format,
            'error': None
        }
        
    except Exception as e:
        return {
            'file': label_file,
            'status': 'error',
            'detection_lines': 0,
            'classification_line': None,
            'detection_format': None,
            'classification_format': None,
            'error': str(e)
        }


def analyze_dataset(dataset_path):
    """
    Analyze all label files in a dataset
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        dict: Dataset analysis results
    """
    dataset_path = Path(dataset_path)
    labels_path = dataset_path / 'train' / 'labels'
    
    if not labels_path.exists():
        return {
            'dataset': str(dataset_path),
            'status': 'error',
            'error': f'Labels directory not found: {labels_path}',
            'files': [],
            'summary': {}
        }
    
    # Find all label files
    label_files = glob.glob(str(labels_path / '*.txt'))
    
    if not label_files:
        return {
            'dataset': str(dataset_path),
            'status': 'error',
            'error': f'No label files found in {labels_path}',
            'files': [],
            'summary': {}
        }
    
    # Analyze each file
    file_analyses = []
    for label_file in label_files:
        analysis = analyze_label_format(label_file)
        file_analyses.append(analysis)
    
    # Generate summary
    total_files = len(file_analyses)
    valid_files = sum(1 for f in file_analyses if f['status'] == 'valid')
    error_files = sum(1 for f in file_analyses if f['status'] == 'error')
    empty_files = sum(1 for f in file_analyses if f['status'] == 'empty')
    
    detection_formats = {}
    classification_formats = {}
    
    for analysis in file_analyses:
        if analysis['detection_format']:
            detection_formats[analysis['detection_format']] = detection_formats.get(analysis['detection_format'], 0) + 1
        
        if analysis['classification_format']:
            classification_formats[analysis['classification_format']] = classification_formats.get(analysis['classification_format'], 0) + 1
    
    # Check data.yaml
    data_yaml_path = dataset_path / 'data.yaml'
    data_yaml_info = None
    if data_yaml_path.exists():
        try:
            with open(data_yaml_path, 'r') as f:
                data_yaml = yaml.safe_load(f)
                data_yaml_info = {
                    'nc': data_yaml.get('nc', 'unknown'),
                    'names': data_yaml.get('names', []),
                    'train': data_yaml.get('train', 'unknown'),
                    'val': data_yaml.get('val', 'unknown'),
                    'test': data_yaml.get('test', 'unknown')
                }
        except Exception as e:
            data_yaml_info = {'error': str(e)}
    
    summary = {
        'total_files': total_files,
        'valid_files': valid_files,
        'error_files': error_files,
        'empty_files': empty_files,
        'detection_formats': detection_formats,
        'classification_formats': classification_formats,
        'data_yaml': data_yaml_info
    }
    
    return {
        'dataset': str(dataset_path),
        'status': 'success',
        'files': file_analyses,
        'summary': summary
    }


def print_analysis_results(results):
    """
    Print formatted analysis results
    
    Args:
        results: Analysis results from analyze_dataset
    """
    print(f"\n{'='*60}")
    print(f"Dataset Analysis: {results['dataset']}")
    print(f"{'='*60}")
    
    if results['status'] != 'success':
        print(f"❌ Error: {results['error']}")
        return
    
    summary = results['summary']
    
    print(f"📊 Summary:")
    print(f"   Total files: {summary['total_files']}")
    print(f"   Valid files: {summary['valid_files']} ✅")
    print(f"   Error files: {summary['error_files']} ❌")
    print(f"   Empty files: {summary['empty_files']} ⚠️")
    
    print(f"\n🔍 Detection Formats:")
    for format_name, count in summary['detection_formats'].items():
        percentage = (count / summary['total_files']) * 100
        print(f"   {format_name}: {count} ({percentage:.1f}%)")
    
    print(f"\n🏷️ Classification Formats:")
    for format_name, count in summary['classification_formats'].items():
        percentage = (count / summary['total_files']) * 100
        print(f"   {format_name}: {count} ({percentage:.1f}%)")
    
    if summary['data_yaml']:
        print(f"\n📋 Data.yaml Info:")
        yaml_info = summary['data_yaml']
        if 'error' not in yaml_info:
            print(f"   Classes (nc): {yaml_info['nc']}")
            print(f"   Class names: {yaml_info['names']}")
            print(f"   Train path: {yaml_info['train']}")
            print(f"   Val path: {yaml_info['val']}")
            print(f"   Test path: {yaml_info['test']}")
        else:
            print(f"   Error reading data.yaml: {yaml_info['error']}")
    
    # Show sample files with issues
    error_files = [f for f in results['files'] if f['status'] != 'valid']
    if error_files:
        print(f"\n❌ Files with Issues:")
        for file_analysis in error_files[:5]:  # Show first 5 errors
            print(f"   {Path(file_analysis['file']).name}: {file_analysis['error']}")
        if len(error_files) > 5:
            print(f"   ... and {len(error_files) - 5} more")


def validate_format_consistency(results):
    """
    Validate format consistency across dataset
    
    Args:
        results: Analysis results
        
    Returns:
        dict: Validation results
    """
    if results['status'] != 'success':
        return {'valid': False, 'issues': [results['error']]}
    
    issues = []
    summary = results['summary']
    
    # Check for multiple detection formats
    if len(summary['detection_formats']) > 1:
        issues.append(f"Multiple detection formats found: {list(summary['detection_formats'].keys())}")
    
    # Check for multiple classification formats
    if len(summary['classification_formats']) > 1:
        issues.append(f"Multiple classification formats found: {list(summary['classification_formats'].keys())}")
    
    # Check for invalid formats
    if 'invalid' in summary['detection_formats']:
        issues.append(f"Invalid detection format found in {summary['detection_formats']['invalid']} files")
    
    if 'invalid' in summary['classification_formats']:
        issues.append(f"Invalid classification format found in {summary['classification_formats']['invalid']} files")
    
    # Check data.yaml consistency
    if summary['data_yaml'] and 'error' not in summary['data_yaml']:
        yaml_info = summary['data_yaml']
        if yaml_info['nc'] != 'unknown':
            # Check if number of classes matches classification format
            if 'python_list' in summary['classification_formats']:
                # Sample a file to check classification vector length
                sample_files = [f for f in results['files'] if f['classification_format'] == 'python_list']
                if sample_files:
                    try:
                        with open(sample_files[0]['file'], 'r') as f:
                            lines = f.read().strip().splitlines()
                            if lines:
                                cl_line = lines[-1]
                                parsed = ast.literal_eval(cl_line)
                                if len(parsed) != yaml_info['nc']:
                                    issues.append(f"Classification vector length ({len(parsed)}) doesn't match nc ({yaml_info['nc']})")
                    except:
                        pass
    
    return {
        'valid': len(issues) == 0,
        'issues': issues
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze YOLOv5 label formats')
    parser.add_argument('--datasets', nargs='+', required=True, help='Dataset paths to analyze')
    parser.add_argument('--validate', action='store_true', help='Validate format consistency')
    parser.add_argument('--fix', action='store_true', help='Attempt to fix format issues')
    
    args = parser.parse_args()
    
    print("🔍 YOLOv5 Label Format Analyzer")
    print("=" * 60)
    
    all_results = []
    
    for dataset_path in args.datasets:
        print(f"\nAnalyzing dataset: {dataset_path}")
        results = analyze_dataset(dataset_path)
        all_results.append(results)
        print_analysis_results(results)
        
        if args.validate:
            validation = validate_format_consistency(results)
            if validation['valid']:
                print(f"\n✅ Dataset format is consistent")
            else:
                print(f"\n❌ Format consistency issues found:")
                for issue in validation['issues']:
                    print(f"   - {issue}")
    
    # Cross-dataset comparison
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("Cross-Dataset Comparison")
        print(f"{'='*60}")
        
        detection_formats = set()
        classification_formats = set()
        
        for results in all_results:
            if results['status'] == 'success':
                detection_formats.update(results['summary']['detection_formats'].keys())
                classification_formats.update(results['summary']['classification_formats'].keys())
        
        print(f"Detection formats across all datasets: {list(detection_formats)}")
        print(f"Classification formats across all datasets: {list(classification_formats)}")
        
        if len(detection_formats) > 1:
            print("⚠️ Warning: Multiple detection formats found across datasets")
        
        if len(classification_formats) > 1:
            print("⚠️ Warning: Multiple classification formats found across datasets")


if __name__ == "__main__":
    main() 