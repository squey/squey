#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run code coverage analysis
python $DIR/fastcov.py --lcov -o main_coverage.info \
    --exclude /usr/ /app/ third_party/ squey-utils/ external/ build/ /moc_

# Generate HTML report
SOURCE_DATE_EPOCH=$(date +%s) genhtml main_coverage.info --output-directory code_coverage_report --dark-mode

# Generate Cobertura XML report
python /usr/lib/python3.*/site-packages/lcov_cobertura/lcov_cobertura.py main_coverage.info --output code_coverage_report/cobertura-coverage.xml --demangle
