# Railway Deployment Guide

## Authentication System Summary

Your application now has a complete authentication system:

### Features
- ✅ User registration (sign up) with username, email, and password
- ✅ User login with JWT token authentication
- ✅ Password hashing with bcrypt (passwords never stored in plain text)
- ✅ Protected API endpoints (upload, dashboard, jobs)
- ✅ User-specific data (each user sees only their own jobs)

### Where User Data is Stored

**Database Table: `users`**
- Username (unique)
- Email (unique)
- Hashed password (bcrypt)
- User ID
- Created timestamp

**Location**: PostgreSQL database (configured in Railway)

### Security
- Passwords are hashed using bcrypt before storage
- JWT tokens for authentication (30-day expiration)
- Tokens stored in browser localStorage
- All protected endpoints require valid JWT token

## Railway Deployment Steps

### 1. Backend Setup

1. **Create Railway Project**
   - Go to railway.app
   - Create new project
   - Add PostgreSQL database service

2. **Set Environment Variables**
   ```
   SECRET_KEY=<generate-a-strong-random-key>
   DATABASE_URL=<railway-postgres-connection-string>
   USE_CELERY=false
   ```

3. **Deploy Backend**
   - Connect your GitHub repo
   - Set root directory to `deepfake-backend/backend`
   - Railway will auto-detect Python and install dependencies
   - Set start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

### 2. Frontend Setup

1. **Create Frontend Service**
   - Add new service in Railway project
   - Set root directory to `deepfake-frontend`

2. **Set Environment Variables**
   ```
   NEXT_PUBLIC_API_BASE=https://your-backend-service.railway.app
   ```

3. **Deploy**
   - Railway will auto-detect Next.js
   - Build and deploy automatically

### 3. Database Migration

The database tables will be created automatically on first startup via:
```python
Base.metadata.create_all(bind=engine)
```

This creates:
- `users` table (for authentication)
- `jobs` table (for image analysis jobs)
- `model_results` table (for analysis results)

### 4. Important Notes

1. **SECRET_KEY**: Generate a strong random key for production:
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

2. **Database**: Railway PostgreSQL is automatically configured

3. **File Storage**: For production, consider using Railway's volume storage or S3 for uploaded images

4. **CORS**: Currently allows all origins (`*`). For production, restrict to your frontend domain:
   ```python
   allow_origins=["https://your-frontend.railway.app"]
   ```

## Testing Authentication

### Register New User
```bash
curl -X POST https://your-backend.railway.app/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","email":"test@example.com","password":"testpass123"}'
```

### Login
```bash
curl -X POST https://your-backend.railway.app/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","password":"testpass123"}'
```

### Use Token
```bash
curl -X GET https://your-backend.railway.app/api/auth/me \
  -H "Authorization: Bearer <your-token>"
```

## Frontend Routes

- `/` - Home page (upload images - requires login)
- `/login` - Login page
- `/register` - Registration page
- `/dashboard` - User's analysis history (requires login)
- `/result/[id]` - View analysis result (requires login)

## User Flow

1. New user visits site → sees "Sign Up" button
2. Clicks "Sign Up" → fills registration form
3. After registration → automatically logged in
4. Can now upload images and view dashboard
5. Existing users click "Sign In" → login page
6. After login → can access all features

## Security Checklist

- [x] Passwords hashed with bcrypt
- [x] JWT tokens for authentication
- [x] Protected endpoints require authentication
- [x] User data isolated (users see only their jobs)
- [ ] Change SECRET_KEY in production
- [ ] Restrict CORS to frontend domain
- [ ] Use HTTPS (Railway provides this automatically)
- [ ] Consider rate limiting for login/register endpoints

