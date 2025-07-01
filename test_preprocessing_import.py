#!/usr/bin/env python3
"""
Test the preprocessing_advanced.py file directly to find the import issue.
"""

import sys
from pathlib import Path

def test_file_syntax():
    """Test if the file has syntax errors."""
    print("🔍 Testing file syntax...")
    
    file_path = Path("app/ml/preprocessing_advanced.py")
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Try to compile the code
        compile(content, str(file_path), 'exec')
        print("✅ File syntax is valid")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error found:")
        print(f"   Line {e.lineno}: {e.text}")
        print(f"   Error: {e.msg}")
        return False
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

def test_direct_import():
    """Test importing the module directly."""
    print("\n🧪 Testing direct module import...")
    
    try:
        import app.ml.preprocessing_advanced as preprocessing
        print("✅ Module imported successfully")
        
        # Check what's available in the module
        available_functions = [name for name in dir(preprocessing) if not name.startswith('_')]
        print(f"Available functions/classes: {available_functions}")
        
        # Specifically check for our function
        if 'create_advanced_features' in available_functions:
            print("✅ create_advanced_features found in module")
            return True
        else:
            print("❌ create_advanced_features NOT found in module")
            print(f"Available items: {available_functions}")
            return False
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_function_import():
    """Test importing the specific function."""
    print("\n🎯 Testing specific function import...")
    
    try:
        from app.ml.preprocessing_advanced import create_advanced_features
        print("✅ create_advanced_features imported successfully!")
        
        # Test the function signature
        import inspect
        sig = inspect.signature(create_advanced_features)
        print(f"Function signature: {sig}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Function import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during function import: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_dependencies():
    """Check if all required dependencies are available."""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'warnings'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    return len(missing_packages) == 0

def check_file_structure():
    """Check the file structure."""
    print("\n📁 Checking file structure...")
    
    required_files = [
        "app/__init__.py",
        "app/ml/__init__.py", 
        "app/ml/preprocessing_advanced.py"
    ]
    
    all_exist = True
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_exist = False
    
    return all_exist

def analyze_function_location():
    """Check where the function is defined in the file."""
    print("\n🔍 Analyzing function location in file...")
    
    file_path = Path("app/ml/preprocessing_advanced.py")
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        function_lines = []
        for i, line in enumerate(lines, 1):
            if 'def create_advanced_features' in line:
                function_lines.append(i)
                print(f"✅ Function found at line {i}: {line.strip()}")
        
        if function_lines:
            # Check indentation - should be at top level (no indentation)
            for line_num in function_lines:
                line = lines[line_num - 1]
                if line.startswith('def '):
                    print("✅ Function is at top level (correct)")
                else:
                    indent = len(line) - len(line.lstrip())
                    print(f"⚠️  Function is indented by {indent} spaces - this might be the issue!")
        else:
            print("❌ Function not found in file")
            
    except Exception as e:
        print(f"❌ Error analyzing file: {e}")

def main():
    """Main diagnostic function."""
    print("🔧 Preprocessing Import Diagnostic Tool")
    print("=" * 50)
    
    # Check file structure
    structure_ok = check_file_structure()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check syntax
    syntax_ok = test_file_syntax()
    
    # Analyze function location
    analyze_function_location()
    
    # Test imports
    if structure_ok and deps_ok and syntax_ok:
        module_ok = test_direct_import()
        function_ok = test_function_import()
        
        print("\n" + "=" * 50)
        print("📋 DIAGNOSIS SUMMARY")
        print("=" * 50)
        
        if module_ok and function_ok:
            print("🎉 Everything working! The import should work now.")
        else:
            print("🔧 Issues found:")
            if not module_ok:
                print("   ❌ Module import failed")
            if not function_ok:
                print("   ❌ Function import failed")
    else:
        print("\n❌ Basic issues found:")
        if not structure_ok:
            print("   - Missing files")
        if not deps_ok:
            print("   - Missing dependencies")
        if not syntax_ok:
            print("   - Syntax errors")
    
    print("\n🎯 Next step:")
    print("Based on the results above, we can fix the specific issue.")

if __name__ == "__main__":
    main()