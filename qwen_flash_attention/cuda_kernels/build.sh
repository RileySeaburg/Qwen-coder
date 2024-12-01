#!/bin/bash
/bin/mkdir -p build
cd build
/usr/bin/cmake ..
/usr/bin/make -j$(/usr/bin/nproc)
/usr/bin/sudo /usr/bin/make install
