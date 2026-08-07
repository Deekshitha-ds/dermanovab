CREATE DATABASE IF NOT EXISTS dermasense CHARACTER SET utf8mb4;
USE dermasense;

CREATE TABLE IF NOT EXISTS users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(120) NOT NULL,
  email VARCHAR(180) NOT NULL UNIQUE,
  hashed_password VARCHAR(255) NOT NULL,
  profile_picture_url VARCHAR(500),
  age INT,
  gender VARCHAR(30),
  skin_type VARCHAR(30),
  hair_type VARCHAR(30),
  sensitive_skin BOOLEAN DEFAULT FALSE,
  pregnant BOOLEAN,
  allergies VARCHAR(300),
  preferred_brands VARCHAR(300),
  monthly_budget FLOAT DEFAULT 800,
  is_admin BOOLEAN DEFAULT FALSE,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS products (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(200) NOT NULL,
  brand VARCHAR(80) NOT NULL,
  category VARCHAR(60) NOT NULL,
  price FLOAT NOT NULL,
  ingredients JSON NOT NULL,
  skin_type VARCHAR(30),
  hair_type VARCHAR(30),
  concern VARCHAR(60),
  dermatologist_tested BOOLEAN DEFAULT FALSE,
  fragrance_free BOOLEAN DEFAULT FALSE,
  paraben_free BOOLEAN DEFAULT FALSE,
  cruelty_free BOOLEAN DEFAULT FALSE,
  vegan BOOLEAN DEFAULT FALSE,
  rating FLOAT DEFAULT 4.0,
  image_url VARCHAR(500),
  description TEXT,
  purchase_link VARCHAR(500)
);

CREATE TABLE IF NOT EXISTS analyses (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL,
  mode VARCHAR(10) NOT NULL,
  image_url VARCHAR(500),
  detected_type VARCHAR(30),
  detected_issues JSON NOT NULL,
  scores JSON NOT NULL,
  face_detected BOOLEAN DEFAULT FALSE,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS recommendations (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL,
  analysis_id INT,
  product_id INT NOT NULL,
  routine_slot VARCHAR(20),
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE SET NULL,
  FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS wishlist (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL,
  product_id INT NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS history (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL,
  action VARCHAR(60) NOT NULL,
  details JSON NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
