#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data format converter for YOLOv5 classification labels
Converts between different classification label formats:
1. Space-separated: "0 0 1"
2. Python list: "[0, 0, 1]"
"""

import os
import glob
import argparse
from pathlib import Path
import ast


def convert_labels_to_space_separated(input_dir, output_dir=None):
    """
    Convert classification labels from Python list format to space-separated format
    
    Args:
        input_dir: Directory containing label files
        output_dir: Output directory (if None, overwrites input files)
    """
    if output_dir is None:
        output_dir = input_dir
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    label_files = glob.glob(os.path.join(input_dir, "*.txt"))
    
    for label_file in label_files:
        filename = os.path.basename(label_file)
        output_file = os.path.join(output_dir, filename)
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        if len(lines) >= 2:
            # Keep detection labels (first line)
            detection_line = lines[0].strip()
            
            # Convert classification label (second line)
            cl_line = lines[1].strip()
            
            if cl_line.startswith('[') and cl_line.endswith(']'):
                # Convert Python list to space-separated
                try:
                    cl_list = ast.literal_eval(cl_line)
                    cl_space = ' '.join(map(str, cl_list))
                except (ValueError, SyntaxError):
                    print(f"Warning: Could not parse classification label in {filename}: {cl_line}")
                    cl_space = cl_line
            else:
                # Already in space-separated format
                cl_space = cl_line
            
            # Write converted file
            with open(output_file, 'w') as f:
                f.write(detection_line + '\n')
                f.write(cl_space + '\n')
        
        print(f"Converted: {filename}")


def convert_labels_to_python_list(input_dir, output_dir=None):
    """
    Convert classification labels from space-separated format to Python list format
    
    Args:
        input_dir: Directory containing label files
        output_dir: Output directory (if None, overwrites input files)
    """
    if output_dir is None:
        output_dir = input_dir
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    label_files = glob.glob(os.path.join(input_dir, "*.txt"))
    
    for label_file in label_files:
        filename = os.path.basename(label_file)
        output_file = os.path.join(output_dir, filename)
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        if len(lines) >= 2:
            # Keep detection labels (first line)
            detection_line = lines[0].strip()
            
            # Convert classification label (second line)
            cl_line = lines[1].strip()
            
            if not (cl_line.startswith('[') and cl_line.endswith(']')):
                # Convert space-separated to Python list
                try:
                    cl_values = cl_line.split()
                    cl_list = [float(x) for x in cl_values]
                    cl_python = str(cl_list)
                except ValueError:
                    print(f"Warning: Could not parse classification label in {filename}: {cl_line}")
                    cl_python = cl_line
            else:
                # Already in Python list format
                cl_python = cl_line
            
            # Write converted file
            with open(output_file, 'w') as f:
                f.write(detection_line + '\n')
                f.write(cl_python + '\n')
        
        print(f"Converted: {filename}")


def validate_label_format(input_dir):
    """
    Validate the format of classification labels in the directory
    
    Args:
        input_dir: Directory containing label files
    """
    label_files = glob.glob(os.path.join(input_dir, "*.txt"))
    
    space_separated_count = 0
    python_list_count = 0
    invalid_count = 0
    
    for label_file in label_files:
        filename = os.path.basename(label_file)
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        if len(lines) >= 2:
            cl_line = lines[1].strip()
            
            if cl_line.startswith('[') and cl_line.endswith(']'):
                try:
                    ast.literal_eval(cl_line)
                    python_list_count += 1
                except (ValueError, SyntaxError):
                    invalid_count += 1
                    print(f"Invalid Python list format in {filename}: {cl_line}")
            else:
                try:
                    values = cl_line.split()
                    [float(x) for x in values]
                    space_separated_count += 1
                except ValueError:
                    invalid_count += 1
                    print(f"Invalid space-separated format in {filename}: {cl_line}")
        else:
            invalid_count += 1
            print(f"Invalid file format in {filename}: insufficient lines")
    
    print(f"\nValidation Results for {input_dir}:")
    print(f"Space-separated format: {space_separated_count}")
    print(f"Python list format: {python_list_count}")
    print(f"Invalid format: {invalid_count}")
    print(f"Total files: {len(label_files)}")


def main():
    parser = argparse.ArgumentParser(description='Convert YOLOv5 classification label formats')
    parser.add_argument('--input', required=True, help='Input directory containing label files')
    parser.add_argument('--output', help='Output directory (if not specified, overwrites input)')
    parser.add_argument('--format', choices=['space', 'python'], required=True,
                       help='Target format: space (space-separated) or python (Python list)')
    parser.add_argument('--validate', action='store_true', help='Validate label format without converting')
    
    args = parser.parse_args()
    
    if args.validate:
        validate_label_format(args.input)
    else:
        if args.format == 'space':
            convert_labels_to_space_separated(args.input, args.output)
        elif args.format == 'python':
            convert_labels_to_python_list(args.input, args.output)
        
        print(f"\nConversion completed!")
        if args.output:
            print(f"Output directory: {args.output}")
        else:
            print(f"Files overwritten in: {args.input}")


if __name__ == "__main__":
    main() 