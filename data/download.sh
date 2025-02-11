#!/bin/bash

if [ ! -e genre-classification-dataset-imdb.zip ]; then
    curl -L -o genre-classification-dataset-imdb.zip\
      https://www.kaggle.com/api/v1/datasets/download/hijest/genre-classification-dataset-imdb
fi

if [ ! -d genre-classification-imdb ]; then
    unzip genre-classification-dataset-imdb.zip 
    mv "Genre Classification Dataset" genre-classification-imdb
fi

if [ ! -e bbc-full-text-document-classification.zip ]; then
    curl -L -o bbc-full-text-document-classification.zip\
      https://www.kaggle.com/api/v1/datasets/download/shivamkushwaha/bbc-full-text-document-classification
fi

if [ ! -d bbc ]; then
    unzip bbc-full-text-document-classification.zip
fi
