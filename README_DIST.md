
# RAG Chat Distribution Package

## System Requirements
- Windows 10/11
- Python 3.8+
- Node.js 16+

## Installation
1. Run `package_dist.bat` to create the distribution package
2. The package will contain compiled executables, no source code exposed

## Directory Structure
```
rag_chat_dist/
├── backend/          # Compiled backend executable
├── frontend/        # Built frontend assets
├── model_cache/     # AI model files
├── vectorstore/    # Vector database
└── install.bat     # Installation script
```

## Security Features
1. Python code compiled to executable
2. Frontend code minified and optimized
3. Sensitive configurations protected

## Usage
1. Copy the distribution folder to target machine
2. Run the main executable
3. No need to install Python/Node on production machines
