# DeepVerify - Deepfake Detection Platform

A comprehensive deepfake detection platform with multi-model analysis, heatmap visualization, and multi-language support.

## 📁 Project Structure

```
DeepFake-Detector/
├── backend/                 # FastAPI backend application
│   ├── app/                # Main application code
│   │   ├── main.py         # FastAPI app and routes
│   │   ├── models_interface.py  # Model loading and prediction
│   │   ├── auth.py         # Authentication logic
│   │   ├── database.py     # Database configuration
│   │   ├── crud.py         # Database operations
│   │   └── ...             # Other modules
│   ├── models/             # ML model files (not in git)
│   ├── data/               # Runtime data (uploads, heatmaps - not in git)
│   ├── requirements.txt    # Python dependencies
│   └── start_backend.sh    # Backend startup script
│
├── frontend/                # Next.js frontend application
│   ├── src/
│   │   ├── pages/          # Next.js pages
│   │   ├── components/     # React components
│   │   ├── lib/            # Utilities and API client
│   │   └── hooks/          # Custom React hooks
│   ├── public/
│   │   └── locales/        # Translation files (i18n)
│   ├── package.json        # Node.js dependencies
│   └── next.config.js      # Next.js configuration
│
├── data/                    # Shared data directory
│   ├── uploads/            # User uploaded images
│   └── heatmaps/           # Generated heatmap visualizations
│
├── docs/                    # Documentation
│   ├── README.md           # Backend documentation
│   ├── AUTHENTICATION.md   # Auth setup guide
│   ├── LOCAL_SETUP.md      # Local development guide
│   └── ...                 # Other documentation
│
├── scripts/                 # Utility scripts
│   ├── test_models.py      # Test model loading/prediction
│   └── view_users.py       # View database users
│
└── docker-compose.yml       # Docker configuration

```

## 🚀 Quick Start

### Backend Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
./start_backend.sh
```

The backend will run on `http://localhost:8000`

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The frontend will run on `http://localhost:3000`

## 🌐 Features

- **Multi-Model Analysis**: Uses 4 different deepfake detection models
- **Heatmap Visualization**: Grad-CAM heatmaps showing detection regions
- **Multi-Language Support**: 8 languages (EN, ES, FR, DE, HI, ZH, JA, AR)
- **User Authentication**: JWT-based authentication system
- **Dashboard**: View analysis history and results
- **Real-time Processing**: Background task processing with status updates

## 📝 Documentation

See the `docs/` directory for detailed documentation:
- `docs/README.md` - Backend overview
- `docs/LOCAL_SETUP.md` - Local development setup
- `docs/AUTHENTICATION.md` - Authentication guide
- `docs/RAILWAY_DATABASE_SETUP.md` - Railway deployment guide

## 🛠️ Development

### Testing Models

```bash
cd scripts
python3 test_models.py
```

### Viewing Users

```bash
cd scripts
python3 view_users.py
```

## 📦 Dependencies

### Backend
- FastAPI
- TensorFlow/Keras
- SQLAlchemy
- JWT authentication

### Frontend
- Next.js
- React
- TypeScript
- Tailwind CSS
- next-i18next (internationalization)

## 🔒 Security

- JWT token-based authentication
- Password hashing with bcrypt
- CORS configuration
- Input validation

## 📄 License

[Your License Here]

