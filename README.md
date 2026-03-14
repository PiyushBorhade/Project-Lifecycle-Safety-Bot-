# 🏗️ Construction Safety Bot

A multi-source RAG chatbot for construction safety data that can query both text documents and CSV metrics using Azure OpenAI.

## 📋 Overview

This bot answers questions about construction safety by intelligently routing queries to:
- **TXT documents** - Safety procedures, descriptions, owners, fields
- **CSV data** - Baseline metrics and monthly performance data

Built for the Project Lifecycle Safety Bot Hackathon.

## ✨ Features

- **Smart Query Routing** - Automatically determines whether to use TXT, CSV, or both
- **Semantic Search** - Finds relevant information across documents
- **Grounded Answers** - Every response cites its sources
- **Transparent Output** - Shows routing decisions and data sources
- **Evaluation Harness** - Built-in testing with sample questions
