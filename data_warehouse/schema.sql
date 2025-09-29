-- Table for image metadata
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    label TEXT,
    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed BOOLEAN DEFAULT 0
);

-- Table for training runs
CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    training_time REAL, -- in seconds
    accuracy REAL,
    loss REAL,
    history TEXT,
    model_path TEXT,
    hyperparams TEXT
);