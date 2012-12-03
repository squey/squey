#!/bin/sh

# object registration
./bin/Thive_massive_register 1000 0 0 0 0 0
./bin/Thive_massive_register 10000 0 0 0 0 0
./bin/Thive_massive_register 100000 0 0 0 0 0
./bin/Thive_massive_register 1000000 0 0 0 0 0


# property registration
# varying property number
./bin/Thive_massive_register 1 1000 0 0 0 0
./bin/Thive_massive_register 1 10000 0 0 0 0
./bin/Thive_massive_register 1 100000 0 0 0 0
./bin/Thive_massive_register 1 1000000 0 0 0 0

# varying objects/properties ratio
./bin/Thive_massive_register 10 100000 0 0 0 0
./bin/Thive_massive_register 100 10000 0 0 0 0
./bin/Thive_massive_register 1000 1000 0 0 0 0
./bin/Thive_massive_register 10000 100 0 0 0 0
./bin/Thive_massive_register 100000 10 0 0 0 0


# object actors registration
# varying actors number
./bin/Thive_massive_register 1 0 1000 0 0 0
./bin/Thive_massive_register 1 0 10000 0 0 0
./bin/Thive_massive_register 1 0 100000 0 0 0
./bin/Thive_massive_register 1 0 1000000 0 0 0

# varying objects/actors ratio
./bin/Thive_massive_register 10 0 100000 0 0 0
./bin/Thive_massive_register 100 0 10000 0 0 0
./bin/Thive_massive_register 1000 0 1000 0 0 0
./bin/Thive_massive_register 10000 0 100 0 0 0
./bin/Thive_massive_register 100000 0 10 0 0 0


# property actors registration
# varying actors number
./bin/Thive_massive_register 1 1 0 1000 0 0
./bin/Thive_massive_register 1 1 0 10000 0 0
./bin/Thive_massive_register 1 1 0 100000 0 0
./bin/Thive_massive_register 1 1 0 1000000 0 0

# varying properties/actors ratio
./bin/Thive_massive_register 1 10 0 100000 0 0
./bin/Thive_massive_register 1 100 0 10000 0 0
./bin/Thive_massive_register 1 1000 0 1000 0 0
./bin/Thive_massive_register 1 10000 0 100 0 0
./bin/Thive_massive_register 1 100000 0 10 0 0


# object pbservers registration
# varying observers number
./bin/Thive_massive_register 1 0 0 0 1000 0
./bin/Thive_massive_register 1 0 0 0 10000 0
./bin/Thive_massive_register 1 0 0 0 100000 0
./bin/Thive_massive_register 1 0 0 0 1000000 0

# varying objects/observers ratio
./bin/Thive_massive_register 10 0 0 0 100000 0
./bin/Thive_massive_register 100 0 0 0 10000 0
./bin/Thive_massive_register 1000 0 0 0 1000 0
./bin/Thive_massive_register 10000 0 0 0 100 0
./bin/Thive_massive_register 100000 0 0 0 10 0


# property observers registration
# varying servers number

./bin/Thive_massive_register 1 1 0 0 0 1000
./bin/Thive_massive_register 1 1 0 0 0 10000
./bin/Thive_massive_register 1 1 0 0 0 100000
./bin/Thive_massive_register 1 1 0 0 0 1000000

# varying properties/observers ratio
./bin/Thive_massive_register 1 10 0 0 0 100000
./bin/Thive_massive_register 1 100 0 0 0 10000
./bin/Thive_massive_register 1 1000 0 0 0 1000
./bin/Thive_massive_register 1 10000 0 0 0 100
./bin/Thive_massive_register 1 100000 0 0 0 10
