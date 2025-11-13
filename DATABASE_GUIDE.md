# Database Setup Guide

## ✅ Current Status: Local SQLite Database

Your application is **already set up** to use a local SQLite database. Everything works automatically!

### What's Working Now

- ✅ **Database**: `deepfake.db` (in `deepfake-backend/backend/`)
- ✅ **Tables Created**: `users`, `jobs`, `model_results`
- ✅ **Authentication**: Ready to use
- ✅ **No Setup Required**: Works immediately

### Test It Now

1. **Start the backend**:
   ```bash
   cd deepfake-backend/backend
   source venv/bin/activate
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Register a user** (via frontend or API):
   - Go to `http://localhost:3000/register`
   - Or use curl:
     ```bash
     curl -X POST http://localhost:8000/api/auth/register \
       -H "Content-Type: application/json" \
       -d '{"username":"test","email":"test@test.com","password":"test123"}'
     ```

3. **View database**:
   ```bash
   sqlite3 deepfake-backend/backend/deepfake.db
   .tables
   SELECT * FROM users;
   ```

---

## 🚀 Setting Up Railway Database (When Ready)

### Step-by-Step Instructions

#### 1. Create PostgreSQL Database on Railway

1. Go to [railway.app](https://railway.app) and open your project
2. Click **"+ New"** button
3. Select **"Database"** → **"Add PostgreSQL"**
4. Railway automatically creates and configures PostgreSQL

#### 2. Get Database Connection String

1. Click on the **PostgreSQL service** you just created
2. Go to the **"Variables"** tab
3. Find `DATABASE_URL` - it looks like:
   ```
   postgresql://postgres:password@hostname.railway.app:5432/railway
   ```

#### 3. Connect Backend to Railway Database

1. Click on your **Backend service** in Railway
2. Go to **"Variables"** tab
3. Add/Update these variables:

   ```
   DATABASE_URL=${{Postgres.DATABASE_URL}}
   SECRET_KEY=<generate-a-random-key>
   USE_CELERY=false
   ```

   **Note**: `${{Postgres.DATABASE_URL}}` automatically references your PostgreSQL service

#### 4. Generate Secret Key

Run this command to generate a secure key:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Copy the output and use it for `SECRET_KEY`

#### 5. Deploy

1. Railway will automatically redeploy your backend
2. Check the logs to see:
   - ✅ Database connection successful
   - ✅ Tables created automatically
   - ✅ Application started

#### 6. Verify It Works

Test the registration endpoint:
```bash
curl -X POST https://your-backend.railway.app/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"test","email":"test@test.com","password":"test123"}'
```

If successful, the user is now stored in Railway's PostgreSQL!

---

## 🔄 How It Switches Between Databases

The application automatically detects which database to use:

| Environment | Database | How It Works |
|------------|----------|--------------|
| **Local** | SQLite | Uses `deepfake.db` file (default) |
| **Railway** | PostgreSQL | Uses `DATABASE_URL` environment variable |

**No code changes needed!** Just set the environment variable.

---

## 📊 Database Tables

All tables are created automatically:

### `users` Table
- Stores user accounts (username, email, hashed password)
- Used for authentication

### `jobs` Table
- Stores image analysis jobs
- Linked to users via `user_id`

### `model_results` Table
- Stores analysis results from models
- Linked to jobs via `job_id`

---

## 🔐 Security Notes

1. **Local Development**: SQLite file is in your project directory
2. **Railway Production**: PostgreSQL is managed by Railway (secure)
3. **Passwords**: Always hashed with bcrypt (never plain text)
4. **Connection**: Railway uses secure network connections

---

## 📝 Quick Reference

### Local Development
- **Database**: SQLite (`deepfake.db`)
- **Location**: `deepfake-backend/backend/deepfake.db`
- **Setup**: None needed - works automatically

### Railway Production
- **Database**: PostgreSQL (managed by Railway)
- **Setup**: Add PostgreSQL service + set `DATABASE_URL`
- **Migration**: Automatic - tables create on first run

### Switching
- **Local → Railway**: Just set `DATABASE_URL` on Railway
- **Railway → Local**: Remove `DATABASE_URL` (uses SQLite)

---

## ❓ Troubleshooting

### "Database connection failed"
- Check `DATABASE_URL` is set correctly
- Verify PostgreSQL service is running
- Check Railway logs for errors

### "Table already exists"
- This is normal - tables are already created
- Application continues working normally

### "No such table: users"
- Restart the backend service
- Tables should auto-create on startup

---

## ✅ Summary

**Right Now (Local)**:
- ✅ Using SQLite database
- ✅ All tables created
- ✅ Authentication working
- ✅ Ready to test locally

**When Ready for Railway**:
1. Add PostgreSQL service
2. Set `DATABASE_URL` environment variable
3. Deploy - that's it!

No code changes needed - the app automatically uses the right database! 🎉

