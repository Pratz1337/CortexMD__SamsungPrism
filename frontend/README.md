# CortexMD Frontend - Next.js 14.2.1

A modern, responsive frontend for the CortexMD medical diagnosis system built with Next.js, TypeScript, and TailwindCSS.

## Features

- 🏥 **Medical Diagnosis Interface** - Comprehensive form for patient data input
- 💬 **Real-time AI Chat** - Interactive chat with medical AI assistant
- 📊 **Live Processing Status** - Real-time updates during diagnosis processing
- 🔍 **UMLS Code Lookup** - Medical terminology search and normalization
- 📱 **Responsive Design** - Works on desktop, tablet, and mobile devices
- 🎨 **Medical-themed UI** - Professional healthcare application design
- ⚡ **Real-time Updates** - Live processing status and results
- 🔒 **Type Safety** - Full TypeScript support throughout

## Tech Stack

- **Framework**: Next.js 14.2.1 with App Router
- **Language**: TypeScript
- **Styling**: TailwindCSS with custom medical theme
- **State Management**: Zustand
- **Forms**: React Hook Form
- **HTTP Client**: Axios
- **Icons**: Heroicons
- **Charts**: Recharts
- **Animations**: Framer Motion
- **Notifications**: React Hot Toast

## Quick Start

### Prerequisites

- Node.js 18+ 
- npm or yarn
- Running CortexMD backend (Flask app on port 5000)

### Installation

1. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   # or
   yarn install
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env.local
   ```

4. **Start development server:**
   ```bash
   npm run dev
   # or
   yarn dev
   ```

5. **Open your browser:**
   Navigate to [http://localhost:3000](http://localhost:3000)

## Environment Variables

Create a `.env.local` file in the frontend directory:

```env
# Backend API URL
NEXT_PUBLIC_BACKEND_URL=http://localhost:5000

# For production
NEXT_PUBLIC_BACKEND_URL=https://your-backend-domain.com
```

## Project Structure

```
frontend/
├── src/
│   ├── app/                  # Next.js App Router pages
│   │   ├── layout.tsx        # Root layout component
│   │   ├── page.tsx          # Home page
│   │   └── globals.css       # Global styles
│   ├── components/           # React components
│   │   ├── diagnosis/        # Diagnosis-related components
│   │   ├── chat/            # Chat interface components
│   │   ├── layout/          # Layout components (Navbar, Footer)
│   │   └── ui/              # Reusable UI components
│   ├── lib/                 # Utilities and API functions
│   ├── store/               # Zustand state management
│   ├── types/               # TypeScript type definitions
│   └── hooks/               # Custom React hooks
├── public/                  # Static assets
├── tailwind.config.js       # Tailwind configuration
├── next.config.js           # Next.js configuration
└── package.json             # Dependencies and scripts
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint
- `npm run type-check` - Run TypeScript type checking

## Backend Integration

The frontend communicates with the Flask backend through:

- **Diagnosis API**: Submit patient data and get diagnosis results
- **Chat API**: Real-time chat with medical AI
- **UMLS API**: Medical terminology lookup and normalization
- **Status API**: Real-time processing status updates
- **Health API**: System health monitoring

### API Endpoints

- `POST /diagnose` - Submit diagnosis request
- `GET /status/{session_id}` - Get processing status
- `GET /results/{session_id}` - Get diagnosis results
- `POST /chat` - Send chat message
- `POST /ontology/normalize` - UMLS terminology lookup
- `GET /api/health` - System health status

## Key Components

### DiagnosisForm
Comprehensive form for collecting patient information including:
- Basic demographics (age, gender)
- Symptoms and medical history
- Current medications and allergies
- Vital signs
- Medical image uploads

### ChatInterface
Real-time chat interface with:
- Message history
- Typing indicators
- File upload support
- Audio recording (future feature)

### ProcessingStatus
Live status updates showing:
- Current processing stage
- Progress indicators
- Estimated time remaining
- Error handling

### DiagnosisResults
Comprehensive results display with:
- Primary diagnosis with confidence scores
- Differential diagnoses
- Treatment recommendations
- FOL verification status
- Source credibility indicators

## Styling

The application uses a custom medical-themed design system built on TailwindCSS:

- **Medical Color Palette**: Professional healthcare colors
- **Custom Components**: Reusable medical UI components
- **Responsive Design**: Mobile-first approach
- **Accessibility**: WCAG compliant design
- **Animations**: Smooth transitions and loading states

## State Management

Uses Zustand for simple, effective state management:

- **Patient Input**: Form data and file uploads
- **Diagnosis Results**: Current diagnosis session data
- **Chat Messages**: Chat history and real-time updates
- **UI State**: Loading states, errors, and notifications

## Development

### Code Style
- TypeScript for type safety
- ESLint for code quality
- Prettier for code formatting
- Component-based architecture

### Best Practices
- Server-side rendering with Next.js
- Optimized images and assets
- Progressive enhancement
- Error boundaries
- Accessibility features

## Production Deployment

### Build Process
```bash
npm run build
npm run start
```

### Environment Setup
- Configure `NEXT_PUBLIC_BACKEND_URL` for production
- Set up proper CORS on backend
- Configure SSL certificates
- Set up monitoring and logging

### Docker Support
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Run linting and type checking
6. Submit a pull request

## License

This project is part of the CortexMD medical diagnosis system developed for the Samsung Gen AI Hackathon.
