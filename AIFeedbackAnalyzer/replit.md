# AI Feedback Analyzer

## Overview

This is a Streamlit-based web application that provides AI-powered analysis of user feedback. The application uses OpenAI's GPT-4o model to extract themes, analyze sentiment, and generate actionable insights from various types of user feedback including survey responses, app store reviews, and support chat logs.

## System Architecture

The application follows a modular Python architecture with a web-based frontend built on Streamlit. The system is designed as a single-page application that processes text input in real-time and provides interactive visualizations of the analysis results.

### Frontend Architecture
- **Framework**: Streamlit for rapid web app development
- **UI Components**: Interactive sidebar for configuration, two-column layout for input/output
- **Visualizations**: Plotly for interactive charts and graphs
- **State Management**: Streamlit session state for maintaining analyzer instances

### Backend Architecture
- **Core Processing**: Modular Python classes for text processing and feedback analysis
- **AI Integration**: OpenAI API for sentiment analysis and theme extraction
- **Text Processing**: NLTK for natural language processing tasks
- **Machine Learning**: Scikit-learn for clustering and TF-IDF vectorization

## Key Components

### FeedbackAnalyzer (`feedback_analyzer.py`)
- **Purpose**: Core AI-powered analysis using OpenAI's GPT-4o model
- **Key Features**: Batch sentiment analysis, theme extraction, emotional indicator detection
- **API Integration**: OpenAI client with configurable model selection
- **Data Processing**: Handles multiple feedback items simultaneously for efficiency

### TextProcessor (`text_processor.py`)
- **Purpose**: Text preprocessing and normalization
- **Features**: Text cleaning, tokenization, lemmatization, stopword removal
- **NLTK Integration**: Automatic download of required language resources
- **Custom Preprocessing**: Domain-specific stopwords for feedback analysis

### Main Application (`app.py`)
- **Purpose**: Streamlit web interface and orchestration
- **Features**: User input handling, configuration options, results visualization
- **State Management**: Session-based storage of analyzer instances and results
- **Export Functionality**: CSV and JSON export options

## Data Flow

1. **Input**: Users paste raw feedback text into the web interface
2. **Preprocessing**: TextProcessor cleans and normalizes the input text
3. **AI Analysis**: FeedbackAnalyzer sends processed text to OpenAI for sentiment and theme analysis
4. **Visualization**: Results are displayed using Plotly charts and Streamlit components
5. **Export**: Users can export results in CSV or JSON format

## External Dependencies

### AI Services
- **OpenAI API**: GPT-4o model for advanced text analysis and sentiment detection
- **Authentication**: API key required (configured via environment variable)

### Python Libraries
- **Streamlit**: Web application framework
- **Plotly**: Interactive data visualization
- **NLTK**: Natural language processing toolkit
- **Scikit-learn**: Machine learning utilities for clustering
- **Pandas**: Data manipulation and analysis
- **OpenAI**: Official OpenAI Python client

### NLTK Resources
- **punkt**: Sentence tokenization
- **stopwords**: Common word filtering
- **wordnet**: Word lemmatization

## Deployment Strategy

### Platform
- **Replit**: Cloud-based development and hosting platform
- **Autoscale Deployment**: Configured for automatic scaling based on demand

### Runtime Configuration
- **Python 3.11**: Modern Python runtime with full feature support
- **Nix Package Management**: Stable channel for reliable dependency management
- **Port Configuration**: Application runs on port 5000

### Process Management
- **Streamlit Server**: Built-in development server for rapid iteration
- **Workflow Automation**: Parallel task execution for efficient development

### Environment Setup
- **Dependencies**: Managed via pyproject.toml and uv.lock for reproducible builds
- **Locale Support**: glibc locales for internationalization
- **Package Manager**: UV for fast and reliable dependency resolution

## Changelog

- June 20, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.