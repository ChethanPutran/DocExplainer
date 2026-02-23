#!/usr/bin/env bash
# set -euo pipefail

# if [[ $# -lt 1 ]]; then
#   echo "Usage: $0 <document_path> [--concepts-per-paragraph N] [--no-visualize]"
#   exit 1
# fi

# python src/core/knowlege_modelling/check_knowlege_modelling.py "$@"

# ./run_knowlege_modelling_check.sh data/report.pdf --concepts-per-paragraph 10
# ./run_knowlege_modelling_check.sh <document_path> [--concepts-per-paragraph N] [--no-visualize]
# python src/core/knowlege_modelling/check_knowlege_modelling.py /path/to/file.txt --no-visualize
python src/core/knowlege_modelling/check_knowlege_modelling.py data/report.pdf
