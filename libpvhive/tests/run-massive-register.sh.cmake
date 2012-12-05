#!/bin/sh

run()
{
    ./Thive_massive_register $@
}

# object registration
run 1000 0 0 0 0 0
run 10000 0 0 0 0 0
run 100000 0 0 0 0 0
run 1000000 0 0 0 0 0


# property registration
# varying property number
run 1 1000 0 0 0 0
run 1 10000 0 0 0 0
run 1 100000 0 0 0 0
run 1 1000000 0 0 0 0

# varying objects/properties ratio
run 10 100000 0 0 0 0
run 100 10000 0 0 0 0
run 1000 1000 0 0 0 0
run 10000 100 0 0 0 0
run 100000 10 0 0 0 0


# object actors registration
# varying actors number
run 1 0 1000 0 0 0
run 1 0 10000 0 0 0
run 1 0 100000 0 0 0
run 1 0 1000000 0 0 0

# varying objects/actors ratio
run 10 0 100000 0 0 0
run 100 0 10000 0 0 0
run 1000 0 1000 0 0 0
run 10000 0 100 0 0 0
run 100000 0 10 0 0 0


# property actors registration
# varying actors number
run 1 1 0 1000 0 0
run 1 1 0 10000 0 0
run 1 1 0 100000 0 0
run 1 1 0 1000000 0 0

# varying properties/actors ratio
run 1 10 0 100000 0 0
run 1 100 0 10000 0 0
run 1 1000 0 1000 0 0
run 1 10000 0 100 0 0
run 1 100000 0 10 0 0


# object pbservers registration
# varying observers number
run 1 0 0 0 1000 0
run 1 0 0 0 10000 0
run 1 0 0 0 100000 0
run 1 0 0 0 1000000 0

# varying objects/observers ratio
run 10 0 0 0 100000 0
run 100 0 0 0 10000 0
run 1000 0 0 0 1000 0
run 10000 0 0 0 100 0
run 100000 0 0 0 10 0


# property observers registration
# varying servers number

run 1 1 0 0 0 1000
run 1 1 0 0 0 10000
run 1 1 0 0 0 100000
run 1 1 0 0 0 1000000

# varying properties/observers ratio
run 1 10 0 0 0 100000
run 1 100 0 0 0 10000
run 1 1000 0 0 0 1000
run 1 10000 0 0 0 100
run 1 100000 0 0 0 10
