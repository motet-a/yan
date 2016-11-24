#!/bin/sh -x


cd "$(dirname "$0")"

yan=$(readlink -f -- ../yan.py)

rm -rf real_projects/
mkdir real_projects
(cd real_projects &&
        git clone https://github.com/motet-a/egc.git &&
        cd egc && ./glist_gen.py && $yan .)
