import os
import re

def get_imports_from_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    imports = re.findall(r'^\s*(?:import|from)\s+([^\s]+)', content, re.MULTILINE)
    return imports

def is_library_used(library, file_paths):
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            content = file.read()
        if re.search(r'\b' + re.escape(library.split('.')[0]) + r'\b', content):
            return True
    return False

def main():
    # Get all .py files in the current directory and subdirectories
    py_files = [os.path.join(root, file)
                for root, _, files in os.walk('.')
                for file in files if file.endswith('.py')]

    # Get all imports from .py files
    all_imports = set()
    for py_file in py_files:
        all_imports.update(get_imports_from_file(py_file))

    # Check which libraries are used
    used_libraries = {lib for lib in all_imports if is_library_used(lib, py_files)}

    # Read current requirements.txt
    with open('requirements.txt', 'r') as file:
        requirements = file.read().splitlines()

    # Filter requirements to only include used libraries
    updated_requirements = [req for req in requirements if any(req.startswith(lib) for lib in used_libraries)]

    # Write updated requirements.txt
    with open('requirements.txt', 'w') as file:
        file.write('\n'.join(updated_requirements) + '\n')

if __name__ == "__main__":
    main()