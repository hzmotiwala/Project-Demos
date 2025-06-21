import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from feedback_analyzer import FeedbackAnalyzer
from text_processor import TextProcessor
import json

# Configure page
st.set_page_config(
    page_title="AI Feedback Analyzer",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = FeedbackAnalyzer()
if 'text_processor' not in st.session_state:
    st.session_state.text_processor = TextProcessor()
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

st.title("🤖 AI-Powered Feedback Analyzer")
st.markdown("Extract themes, analyze sentiment, and generate actionable insights from user feedback")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Processing options
    st.subheader("Processing Options")
    num_themes = st.slider("Number of themes to extract", 3, 10, 5)
    min_feedback_length = st.slider("Minimum feedback length (characters)", 10, 100, 20)
    
    # Export options
    st.subheader("Export Options")
    export_format = st.selectbox("Export format", ["CSV", "JSON"])

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📝 Input Feedback")
    
    # Text input area
    feedback_text = st.text_area(
        "Paste your raw feedback here:",
        placeholder="Enter survey responses, app store reviews, support chat logs, etc.\n\nSeparate multiple feedback entries with line breaks.",
        height=300,
        help="You can paste multiple feedback entries. Each line will be treated as a separate feedback item."
    )
    
    # File upload option
    uploaded_file = st.file_uploader(
        "Or upload a text file:",
        type=['txt', 'csv'],
        help="Upload a text file with feedback data. CSV files should have feedback in the first column."
    )
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            if uploaded_file.type == "text/plain":
                feedback_text = str(uploaded_file.read(), "utf-8")
            elif uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
                feedback_text = "\n".join(df.iloc[:, 0].astype(str).tolist())
            st.success("File uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    # Analysis button
    analyze_button = st.button("🔍 Analyze Feedback", type="primary", use_container_width=True)

with col2:
    st.header("📊 Analysis Results")
    
    if analyze_button and feedback_text.strip():
        with st.spinner("Analyzing feedback... This may take a few moments."):
            try:
                # Process and clean text
                feedback_items = st.session_state.text_processor.split_feedback(feedback_text)
                cleaned_feedback = st.session_state.text_processor.clean_feedback_batch(
                    feedback_items, min_length=min_feedback_length
                )
                
                if not cleaned_feedback:
                    st.error("No valid feedback items found. Please check your input.")
                else:
                    # Perform analysis
                    st.session_state.analysis_results = st.session_state.analyzer.analyze_feedback_batch(
                        cleaned_feedback, num_themes=num_themes
                    )
                    st.success(f"Analysis complete! Processed {len(cleaned_feedback)} feedback items.")
                    
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.session_state.analysis_results = None
    
    elif analyze_button and not feedback_text.strip():
        st.error("Please enter some feedback text to analyze.")

# Display results
if st.session_state.analysis_results:
    results = st.session_state.analysis_results
    
    st.header("📈 Analysis Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Feedback", results['total_feedback'])
    
    with col2:
        avg_sentiment = results['sentiment_summary']['average_sentiment']
        sentiment_label = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
        st.metric("Overall Sentiment", sentiment_label, f"{avg_sentiment:.2f}")
    
    with col3:
        st.metric("Top Themes", len(results['themes']))
    
    with col4:
        st.metric("Action Items", len(results['action_items']))
    
    # Sentiment Analysis
    st.subheader("😊 Sentiment Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Sentiment distribution pie chart
        sentiment_counts = results['sentiment_summary']['distribution']
        fig_pie = px.pie(
            values=list(sentiment_counts.values()),
            names=list(sentiment_counts.keys()),
            title="Sentiment Distribution",
            color_discrete_map={
                'positive': '#2E8B57',
                'neutral': '#FFD700',
                'negative': '#DC143C'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Sentiment scores histogram
        sentiment_scores = [item['sentiment_score'] for item in results['detailed_analysis']]
        fig_hist = px.histogram(
            x=sentiment_scores,
            nbins=20,
            title="Sentiment Score Distribution",
            labels={'x': 'Sentiment Score', 'y': 'Count'}
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Neutral")
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Themes Analysis
    st.subheader("🎯 Key Themes")
    
    themes_df = pd.DataFrame(results['themes'])
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Themes bar chart
        fig_themes = px.bar(
            themes_df,
            x='frequency',
            y='theme',
            orientation='h',
            title="Theme Frequency",
            labels={'frequency': 'Number of Mentions', 'theme': 'Theme'}
        )
        fig_themes.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_themes, use_container_width=True)
    
    with col2:
        # Themes table
        st.dataframe(
            themes_df[['theme', 'frequency', 'keywords']],
            use_container_width=True,
            hide_index=True
        )
    
    # Action Items
    st.subheader("🎯 Recommended Actions")
    
    for i, action in enumerate(results['action_items'], 1):
        with st.expander(f"Action Item {i}: {action['category']}", expanded=i <= 3):
            st.write(f"**Priority:** {action['priority']}")
            st.write(f"**Issue:** {action['issue']}")
            st.write(f"**Recommendation:** {action['recommendation']}")
            st.write(f"**Supporting Evidence:** {action['evidence_count']} feedback items")
    
    # Detailed Analysis
    st.subheader("🔍 Detailed Feedback Analysis")
    
    # Filter options
    col1, col2 = st.columns([1, 1])
    with col1:
        sentiment_filter = st.selectbox(
            "Filter by sentiment:",
            ["All", "Positive", "Neutral", "Negative"]
        )
    
    with col2:
        theme_filter = st.selectbox(
            "Filter by theme:",
            ["All"] + [theme['theme'] for theme in results['themes']]
        )
    
    # Filter detailed analysis
    filtered_analysis = results['detailed_analysis']
    
    if sentiment_filter != "All":
        filtered_analysis = [
            item for item in filtered_analysis 
            if item['sentiment'] == sentiment_filter.lower()
        ]
    
    if theme_filter != "All":
        filtered_analysis = [
            item for item in filtered_analysis 
            if theme_filter.lower() in [theme.lower() for theme in item['themes']]
        ]
    
    # Display filtered results
    st.write(f"Showing {len(filtered_analysis)} of {len(results['detailed_analysis'])} feedback items")
    
    for i, item in enumerate(filtered_analysis[:10]):  # Show first 10 items
        with st.expander(f"Feedback {i+1} - {item['sentiment'].title()} ({item['sentiment_score']:.2f})"):
            st.write(f"**Text:** {item['text']}")
            st.write(f"**Themes:** {', '.join(item['themes'])}")
            if item['key_phrases']:
                st.write(f"**Key Phrases:** {', '.join(item['key_phrases'])}")
    
    if len(filtered_analysis) > 10:
        st.info(f"Showing first 10 items. {len(filtered_analysis) - 10} more items available.")
    
    # Export functionality
    st.subheader("📤 Export Results")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("📄 Export Summary Report"):
            if export_format == "CSV":
                # Create summary CSV
                summary_data = {
                    'Metric': ['Total Feedback', 'Average Sentiment', 'Positive %', 'Neutral %', 'Negative %'],
                    'Value': [
                        results['total_feedback'],
                        f"{results['sentiment_summary']['average_sentiment']:.2f}",
                        f"{results['sentiment_summary']['distribution']['positive']}",
                        f"{results['sentiment_summary']['distribution']['neutral']}",
                        f"{results['sentiment_summary']['distribution']['negative']}"
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="Download Summary CSV",
                    data=csv,
                    file_name="feedback_analysis_summary.csv",
                    mime="text/csv"
                )
            else:
                # Export as JSON
                json_data = json.dumps(results, indent=2, default=str)
                st.download_button(
                    label="Download JSON Report",
                    data=json_data,
                    file_name="feedback_analysis_results.json",
                    mime="application/json"
                )
    
    with col2:
        if st.button("📊 Export Detailed Data"):
            # Create detailed CSV
            detailed_data = []
            for item in results['detailed_analysis']:
                detailed_data.append({
                    'Feedback': item['text'],
                    'Sentiment': item['sentiment'],
                    'Sentiment_Score': item['sentiment_score'],
                    'Themes': ', '.join(item['themes']),
                    'Key_Phrases': ', '.join(item['key_phrases'])
                })
            
            detailed_df = pd.DataFrame(detailed_data)
            csv = detailed_df.to_csv(index=False)
            st.download_button(
                label="Download Detailed CSV",
                data=csv,
                file_name="feedback_detailed_analysis.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and OpenAI")
