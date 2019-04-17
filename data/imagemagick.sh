#!/bin/bash

convert           \
   -verbose       \
   -density 150   \
    sample.pdf    \
   -quality 100   \
    sample-page-%d.png

