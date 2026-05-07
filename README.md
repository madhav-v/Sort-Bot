# 🤖 SortBot — Smart Waste Sorting System

SortBot is an AI-powered smart waste sorting simulation system built using:

- Python
- OpenCV
- PyTorch
- Dash Analytics Dashboard

The system detects waste items, classifies them into categories, animates a rotating smart dustbin, and logs all sorting events into a live analytics dashboard.

---

# 📦 Features

## Smart Dustbin Simulation
- Circular rotating smart dustbin
- Animated lid opening/closing
- Falling trash animation
- Particle effects
- Real-time AI detection panel

## AI Waste Classification
Uses MobileNetV2 pretrained model to classify:
- Plastic
- Burnable
- Cans
- Bottles
- Others

## Real-Time Dashboard
- Live event analytics
- Category distribution
- Environmental impact
- Performance metrics
- Export CSV support

## Manual Controls

Keyboard controls:

- `1` → Plastic
- `2` → Burnable
- `3` → Cans
- `4` → Bottles
- `5` → Others
- `Q` → Quit

---

# 📁 Project Structure

```text
smart_dustbin/
│
├── new.py
├── requirements.txt
│
├── dashboard/
│   ├── app_dash.py
│   └── data/
│       └── events.csv
│
└── venv/
```

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/madhav-v/Sort-Bot.git
cd Sort-Bot
```

---

## 2. Create Virtual Environment

### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running The System

## Step 1 — Start Smart Dustbin

Run:

```bash
python new.py
```

This opens:
- Camera
- AI classification window
- Smart dustbin simulation

---

## Step 2 — Open Analytics Dashboard

Open another terminal.

Activate virtual environment again.

### Mac/Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Then run:

```bash
cd dashboard
python app_dash.py
```

Dashboard URL:

```text
http://127.0.0.1:8050
```

Open it in browser.

---


## Detection Flow

1. Camera captures object
2. AI classifies object
3. Dustbin rotates toward category
4. Lid opens
5. Trash animation falls into section
6. Event saved into CSV
7. Dashboard updates automatically

---
---

# 📝 Event Logging

All events are saved inside:

```text
dashboard/data/events.csv
```

Format:

```csv
ts,category,confidence,source,note
```

Example:

```csv
2025-01-01T12:00:00,Plastic,0.92,sortbot-01,auto
```

---

# 🎮 Manual Demo Mode

Press keyboard numbers during simulation:

| Key | Category |
|-----|----------|
| 1 | Plastic |
| 2 | Burnable |
| 3 | Cans |
| 4 | Bottles |
| 5 | Others |

Useful for:
- Presentations
- Demonstrations
- Testing dashboard

---

# 🛠 Technologies Used

- Python
- OpenCV
- NumPy
- PyTorch
- TorchVision
- Dash
- Plotly
- Pandas

---
