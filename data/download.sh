#!/bin/bash

if [ ! -e genre-classification-dataset-imdb.zip ]; then
    curl -L -o genre-classification-dataset-imdb.zip\
      https://www.kaggle.com/api/v1/datasets/download/hijest/genre-classification-dataset-imdb
fi

if [ ! -d genre-classification-imdb ]; then
    unzip genre-classification-dataset-imdb.zip 
    mv "Genre Classification Dataset" genere-classification-imdb
fi
