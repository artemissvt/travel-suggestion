# AI Travel Destination Recommendation Service

## Overview
This project is a machine learning-powered recommendation service
that generates travel destination suggestions based on natural language user preferences.
Users can describe their ideal trip (e.g. climate, activities, budget, atmosphere, season),
and the service returns destinations that best match the provided criteria.
The service is exposed through a FastAPI REST API and is designed to integrate 
with external applications such as web platforms and mobile applications.


## Features
- Natural language travel preference processing
- Kmeans model for clustering
- Destination recommendation engine
- REST API built with FastAPI
- JSON-based responses
- Session-aware requests
- Cloud deployment ready
- Integration with external applications

## Architecture
Client Application --> FastAPI --> Reccomendation Engine --> Destination Dataset

## Technology Stack
- Python
- FastAPI
- Pandas
- NumPy
- Scikit-Learn
- Uvicorn

## Deployment
Production endpoint: https://destination-suggester-b9aov.ondigitalocean.app

Interactive API documentation: https://ml-travel-suggest-2568f.ondigitalocean.app/docs
