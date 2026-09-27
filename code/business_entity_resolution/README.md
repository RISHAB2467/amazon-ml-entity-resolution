# Automated Proxy Teacher Assignment System

A desktop web application that assigns substitute (proxy) teachers automatically
when a teacher is absent, and emails the assigned teacher — replacing a manual
process that is slow, error-prone, and causes double-booking.

Built with **Python + Streamlit + SQLite + Gmail SMTP**.

> **In a hurry?** Read `CONFIGURATION.md` instead. It lists the two things you
> must set up and the three that are optional, and nothing else.

---

## 1. What you need first

- **Python 3.10 or newer** — download from python.org
  During installation, tick **"Add Python to PATH"**. This matters.
- **VS Code** — download from code.visualstudio.com
- A **Gmail account** (only needed for the email feature)

Check Python installed correctly. Open Command Prompt and type:

```
python --version
```

You should see something like `Python 3.12.1`.

---

## 2. Install the project

Open Command Prompt, then:

```
cd proxy_system
pip install -r requirements.txt
```

This installs Streamlit and pandas. It takes 1–2 minutes the first time.

---

## 3. Run the project

```
streamlit run app.py
```

On Windows you can instead just double-click **`run_app.bat`**; on a Mac,
**`run_app.command`**. Both check your setup first and install anything missing.

Your browser opens automatically at `http://localhost:8501`.

The first time it runs, `proxy_system.db` is created automatically in this
folder. On the Dashboard, click **"Load sample data"** to get 6 teachers and a
full timetable so you can test everything immediately.

To stop the app, press `Ctrl + C` in the Command Prompt window.

---

## 4. Setting up email

The system will save assignments without this — email is optional. But to send
real notifications you need a Gmail **App Password**.

### Step 4a — Turn on 2-Step Verification

1. Go to https://myaccount.google.com/security
2. Turn on **2-Step Verification**. App Passwords do not exist without it.

### Step 4b — Create an App Password

1. Go to https://myaccount.google.com/apppasswords
2. Type a name like `Proxy System` and click **Create**
3. Google shows a **16-character password** like `abcd efgh ijkl mnop`
4. Copy it. Remove the spaces when you use it.

### Step 4c — Store it safely as an environment variable

**Never type this password into a Python file.**

On Windows, open Command Prompt and run these two commands (using your own
email and the 16-character password):

```
setx PROXY_EMAIL "yourname@gmail.com"
setx PROXY_PASSWORD "abcdefghijklmnop"
```

Then **close Command Prompt completely and open a new one.** `setx` only
affects windows opened afterwards. Verify:

```
echo %PROXY_EMAIL%
```

Now run `streamlit run app.py` again. The Dashboard should show
"✅ Email is configured".

### Testing email on its own

```
python email_service.py
```

This prints the email text, and sends a test message to yourself if the
credentials are set.

---

## 5. How to use the system

| Page | What it does |
|------|--------------|
| 🏠 Dashboard | Summary counts, recent assignments, email status |
| 👩‍🏫 Manage Teachers | Add, view and remove teachers |
| 📅 Manage Timetable | Set which periods each teacher teaches; view a weekly grid |
| 🔴 Mark Absence & Assign | Pick an absent teacher + date → preview → assign → email |
| 📊 History & Reports | Filter by date, see workload, export to CSV |

**Normal daily workflow for the coordinator:**

1. Go to **🔴 Mark Absence & Assign**
2. Select the absent teacher and today's date
3. Read the **Preview** — it shows who is free for each affected period
4. Click **Confirm absence and assign proxies**
5. Emails go out automatically; results appear on screen

---

## 6. Project files

```
proxy_system/
├── app.py               ← Streamlit screens and navigation
├── database.py          ← All SQLite work: the 4 tables and every query
├── proxy_logic.py       ← The assignment algorithm
├── email_service.py     ← Gmail SMTP sender
├── proxy_system.db      ← Created automatically on first run
├── requirements.txt     ← Python packages
├── run_app.bat          ← Windows: double-click to start
├── run_app.command      ← Mac: double-click to start
├── setup_email.bat      ← Windows: one-time Gmail setup
├── CONFIGURATION.md     ← What you need to set up, and what you don't
├── README.md            ← This file
├── PROJECT_INDEX.docx   ← What every file and function contains
└── TEACHER_GUIDE.md     ← Mentor's guide: explanations, questions, tests
```

Each file has ONE job. `app.py` never touches the database directly — it always
goes through `database.py`. This separation is what makes the project easy to
test and easy to explain in the viva.

---

## 7. Database design

**teachers** — teacher_id, name, email (unique), subject

**timetable** — timetable_id, teacher_id, day, period
`UNIQUE(teacher_id, day, period)` stops the same slot being entered twice.

**proxy_assignments** — assignment_id, absent_teacher_id, proxy_teacher_id,
date, period, status

**email_log** — log_id, assignment_id, sent_at, delivery_status

---

## 8. How the algorithm works

When you mark a teacher absent:

1. The date is converted to a day name (e.g. `2026-08-03` → `Monday`)
2. The system looks up every period that teacher teaches on that day
3. For each affected period, it asks the database for teachers who are:
   - not teaching in that day+period on the normal timetable, **and**
   - not already assigned a proxy duty for that exact date+period, **and**
   - not the absent teacher
4. The first free teacher (alphabetically) is assigned and saved immediately
5. Because each assignment is saved before the next period is checked, the same
   teacher can never be double-booked
6. An email is sent and the result is written to `email_log`

If nobody is free, that period is reported as `no_teacher_available` so the
coordinator can handle it manually — the system never silently skips a class.

---

## 9. Common problems

**`'streamlit' is not recognized`**
Streamlit did not install, or Python is not on PATH. Try `python -m streamlit run app.py`.

**`ModuleNotFoundError: No module named 'streamlit'`**
Run `pip install -r requirements.txt` again, in the same folder.

**Email says "Gmail rejected the login"**
You used your normal Gmail password instead of the 16-character App Password,
or you did not remove the spaces from it.

**"Email is not configured" even after setx**
You did not open a NEW Command Prompt window after running `setx`.

**Everything disappeared**
Check `proxy_system.db` is still in the folder. That file IS your data. Back it
up by copying it.

**"NO TEACHER AVAILABLE" for every period**
Every teacher is teaching in that period. Check the timetable grid on the
📅 Manage Timetable page.

---

## 10. Testing

```
python database.py        # creates tables + loads sample data
python proxy_logic.py     # runs a full assignment on sample data
python email_service.py   # previews and optionally sends a test email
```

See `TEACHER_GUIDE.md` for the full list of 20 test cases and their expected
results, and `PROJECT_INDEX.docx` for a description of every file and function.
