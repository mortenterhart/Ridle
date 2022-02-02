#!/usr/bin/env bash

## Learn representations on dataset umls.
python ridle/rbm/learn_representation.py --dataset umls

## Instance type prediction on dataset umls.
python ridle/classifier/evaluate_neural_network.py --dataset umls
