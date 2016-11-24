#!/bin/sh -ex

cd "$(dirname "$0")"

../yan.py tab_after_space.c >/dev/null &
pid=$!
sleep 2

if kill -0 "$pid" 2>/dev/null
then
    echo 'Yan is still running, test failed.'
    kill $pid
    exit 1
else
    echo 'OK'
fi
