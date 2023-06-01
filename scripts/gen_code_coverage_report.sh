#!/bin/bash

# Run code coverage analysis
lcov -c -d . -o main_coverage.info

# Remove unwanted stuff
lcov --remove main_coverage.info -o main_coverage.info \
'/usr/*' \
'/app/*' \
'*/third_party/*' \
'*/squey-utils/*' \
'*/external/*' \
'*/build/*' \
'*/moc_*' \

# Generate HTML report
genhtml main_coverage.info --output-directory code_coverage_report --dark-mode

# Generate Cobertura XML report
python /usr/lib/python3.*/site-packages/lcov_cobertura/lcov_cobertura.py main_coverage.info --output code_coverage_report/cobertura-coverage.xml --demangle
