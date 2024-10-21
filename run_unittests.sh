#!/bin/sh

for directory in $(ls -d */)
do
  cd $directory
  case $directory in
    vol1/|vol2/)
      python -m unittest discover ./test;;
    vol3/)
      for chapter in $(ls -d */)
      do
        python -m unittest discover ./test
      done
  esac
done
