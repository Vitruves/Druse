#!/bin/bash
cd "$(dirname "$0")/build"
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1
make -j8 2>&1
