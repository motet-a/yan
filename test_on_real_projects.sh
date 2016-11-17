#!/bin/sh -x


cd "$(dirname "$0")"

yan=$(realpath yan.py)

rm -rf test/projects
mkdir test/projects
(cd test/projects &&
        git clone https://github.com/motet-a/egc.git &&
        cd egc && ./glist_gen.py && $yan .)
