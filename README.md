# CS336 Spring 2025 Assignment 1: Basics

My implementation of Assignment 1 in CS-336. 
Original tests uses resource module, which does not work on windows. Skipped the import and tests requiring it. 
I've made minor modifications to test files to force utf-8 format when opening files on windows.

Older block modules used in ablations are broken as I've not kept them up to date with changes in config files.

To do a training run using open web text, clone the repo and run main.py. This will download training and validation data from huggingface and execute a training run using the config file "best.yaml".