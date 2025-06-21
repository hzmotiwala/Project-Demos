# AI Feedback Analyzer

A powerful Streamlit web application that uses OpenAI's GPT-4o to analyze user feedback and extract actionable insights. Perfect for analyzing survey responses, app store reviews, support chat logs, and other customer feedback data.

## Features

- **AI-Powered Analysis**: Uses OpenAI's latest GPT-4o model for intelligent sentiment analysis and theme extraction
- **Multiple Input Methods**: Paste text directly or upload CSV/TXT files
- **Theme Extraction**: Automatically identifies key themes and topics using machine learning clustering
- **Sentiment Analysis**: Analyzes emotional tone with confidence scores and detailed breakdowns
- **Actionable Insights**: Generates specific recommendations based on feedback patterns
- **Interactive Visualizations**: Beautiful charts showing sentiment distribution and theme frequency
- **Export Capabilities**: Download results as CSV or JSON for further analysis
- **Robust Processing**: Handles various feedback formats with intelligent text preprocessing

## Demo

![AI Feedback Analyzer Screenshot](https://via.placeholder.com/800x400?text=AI+Feedback+Analyzer+Demo)

## Quick Start

### Option 1: Deploy on Streamlit Cloud (Recommended)

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy your forked repository
4. Add your OpenAI API key in the Streamlit Cloud secrets

### Option 2: Run Locally

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-feedback-analyzer.git
cd ai-feedback-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

4. Run the application:
```bash
streamlit run app.py
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)

### Getting an OpenAI API Key

1. Visit [OpenAI's website](https://openai.com)
2. Create an account or sign in
3. Go to the API section
4. Generate a new API key
5. Copy the key (starts with `sk-`)

## Usage

1. **Input Feedback**: Paste your raw feedback text or upload a file
2. **Configure Analysis**: Adjust the number of themes to extract and minimum feedback length
3. **Analyze**: Click the "Analyze Feedback" button
4. **Review Results**: Explore sentiment analysis, key themes, and action items
5. **Export**: Download results in CSV or JSON format

### Supported Input Formats

- **Direct Text**: Paste feedback directly into the text area
- **CSV Files**: Upload CSV files (feedback should be in the first column)
- **Text Files**: Upload .txt files with feedback entries

### Example Input

```
The app crashes every time I try to upload a photo. Very frustrating!

Love the new design update, looks much cleaner and modern.

Customer support was unhelpful and took 3 days to respond.

The checkout process is confusing, I couldn't find the payment button.
```

## Architecture

### Core Components

- **`app.py`**: Main Streamlit web interface with interactive visualizations
- **`feedback_analyzer.py`**: AI-powered analysis engine using OpenAI GPT-4o
- **`text_processor.py`**: Text preprocessing and cleaning utilities
- **`.streamlit/config.toml`**: Streamlit configuration for deployment

### Key Technologies

- **Streamlit**: Web application framework
- **OpenAI GPT-4o**: Advanced language model for analysis
- **NLTK**: Natural language processing
- **Scikit-learn**: Machine learning for clustering
- **Plotly**: Interactive data visualizations
- **Pandas**: Data manipulation and analysis

## API Usage

The feedback analyzer can process various types of feedback:

### Sentiment Analysis
- Classifies feedback as positive, neutral, or negative
- Provides confidence scores from -1 to 1
- Identifies emotional indicators in the text

### Theme Extraction
- Uses TF-IDF vectorization and K-means clustering
- Generates descriptive theme names using AI
- Ranks themes by frequency and relevance

### Action Items
- Creates specific, actionable recommendations
- Prioritizes issues by impact and frequency
- Links recommendations to supporting evidence

## Deployment

### Streamlit Cloud

1. Connect your GitHub repository to Streamlit Cloud
2. Add your OpenAI API key in the secrets management
3. Deploy with automatic updates on code changes

### Other Platforms

The application can be deployed on:
- **Heroku**: Use the included `requirements.txt`
- **Railway**: Automatic deployment from GitHub
- **Replit**: Direct import and run
- **Google Cloud Run**: Container-based deployment

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Error Handling

The application includes robust error handling:

- **NLTK Data**: Automatically downloads required language resources
- **API Failures**: Graceful fallback to rule-based analysis
- **File Processing**: Supports various input formats with validation
- **Network Issues**: Retry logic for API calls

## Limitations

- Requires OpenAI API key (usage costs apply)
- Best results with English text
- Large datasets may take longer to process
- API rate limits may affect processing speed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, please:
1. Check the existing [GitHub Issues](https://github.com/yourusername/ai-feedback-analyzer/issues)
2. Create a new issue with detailed information
3. Include sample data and error messages when possible

## Changelog

### Version 1.0.0
- Initial release with core functionality
- OpenAI GPT-4o integration
- Streamlit web interface
- Theme extraction and sentiment analysis
- Export capabilities

---

Built with ❤️ using Streamlit and OpenAI