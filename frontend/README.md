# Multimodal RAG Frontend

React + Vite frontend application for the Multimodal RAG system.

## Features

- 📄 Document upload with drag-and-drop support
- ✅ Support for PDF, DOCX, TXT, and MD files
- 📊 Real-time upload status and results
- 🎨 Modern, responsive UI with dark mode support
- 🔄 Batch file upload capability

## Prerequisites

- Node.js 18+ and npm/yarn/pnpm

## Installation

1. Install dependencies:

```bash
npm install
# or
yarn install
# or
pnpm install
```

2. Create a `.env` file (optional, defaults to `http://localhost:8000`):

```env
VITE_API_BASE_URL=http://localhost:8000
```

## Development

Start the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
```

The app will be available at `http://localhost:5173`.

## Building for Production

Build the application:

```bash
npm run build
# or
yarn build
# or
pnpm build
```

Preview the production build:

```bash
npm run preview
# or
yarn preview
# or
pnpm preview
```

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── DocumentUpload.tsx    # Main upload component
│   │   └── DocumentUpload.css    # Component styles
│   ├── services/
│   │   └── api.ts                # API client service
│   ├── App.tsx                   # Main app component
│   ├── App.css                   # App styles
│   ├── main.tsx                  # Entry point
│   └── index.css                 # Global styles
├── index.html                    # HTML template
├── vite.config.ts                # Vite configuration
├── tsconfig.json                 # TypeScript configuration
└── package.json                  # Dependencies
```

## API Integration

The frontend connects to the backend API at `/api/v1/ingest` for document uploads. Make sure the backend is running on the configured port (default: 8000).

## Supported File Types

- PDF (`.pdf`)
- Word Documents (`.docx`)
- Text Files (`.txt`)
- Markdown (`.md`, `.markdown`)

Maximum file size: 50MB per file
