# Zepto Data & AI Platform — Module 1

**Data Pipeline**

This module implements an end-to-end data pipeline for scraping, cleaning, validating, storing, and querying book catalogue data.

## Project Overview

The pipeline collects book information from the Books to Scrape website and transforms the scraped data into a structured dataset.

The pipeline performs the following operations:

- Scrapes book data from the catalogue.
- Extracts book title, price, rating, availability, and category.
- Cleans and validates the scraped records.
- Converts GBP prices to INR using the required fixed exchange rate.
- Creates a normalized SQLite database.
- Inserts books and categories into separate relational tables.
- Executes SQL queries for data analysis.
- Verifies the SQL JOIN result independently using Pandas.
- Generates SQL query output as a project artifact.

## Data Source

The data is collected from:

https://books.toscrape.com/

The pipeline scrapes the first five catalogue pages.

Each catalogue page contains 20 books.

Therefore, the pipeline processes:

```text
5 pages × 20 books = 100 books
```

## Dataset Fields

The cleaned dataset contains the following fields:

| Field | Description |
| --- | --- |
| title | Book title |
| price_gbp | Original book price in GBP |
| price_inr | Converted book price in INR |
| rating | Book rating from 1 to 5 |
| in_stock | Whether the book is currently in stock |
| category | Book category |

Example:

```text
Title:       A Light in the Attic
Price GBP:   51.77
Price INR:   5461.74
Rating:      3
In Stock:    True
Category:    Poetry
```

## Currency Conversion

The project uses the required fixed conversion rate:

```text
1 GBP = 105.50 INR
```

Currency calculations use Python's Decimal type with ROUND_HALF_UP to avoid floating-point rounding inconsistencies.

For example:

```text
51.77 × 105.50 = 5461.735
```

After rounding to two decimal places:

```text
5461.74 INR
```

The same Decimal-based calculation is used during validation to ensure that the stored INR values are correct.

## Data Cleaning

The cleaning stage validates the following fields:

- Title
- Price
- Rating
- Availability
- Category

Records with invalid required fields are removed rather than assigning artificial values.

The successful pipeline run produced:

```text
Raw rows:              100
Clean rows:            100
Invalid titles:        0
Invalid prices:        0
Invalid ratings:       0
Invalid availability:  0
Invalid categories:    0
```

Therefore, all 100 scraped records passed the cleaning stage.

## Dataset Validation

Before creating the database, the pipeline validates:

- Minimum number of books
- Minimum number of categories
- Required columns
- Rating range
- GBP → INR conversion
- Missing values
- Boolean stock values

The project requires at least:

```text
60 books
3 categories
```

The scraped dataset contains 100 books and therefore satisfies the minimum book requirement.

## SQLite Database

The cleaned data is stored in:

```text
zepto_catalog.db
```

The database uses two tables:

- categories
- books

### Categories Table

The categories table contains unique book categories.

```sql
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY,
    category_name TEXT UNIQUE
);
```

### Books Table

The books table contains the book information.

```sql
CREATE TABLE books (
    book_id INTEGER PRIMARY KEY,
    title TEXT,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock INTEGER,
    category_id INTEGER,
    FOREIGN KEY (category_id)
        REFERENCES categories(category_id)
);
```

The category_id column creates a relationship between the books and categories tables.

### Database Relationship

The database has the following relationship:

```text
categories
    |
    | category_id
    |
    ↓
 books
```

For example:

```text
categories
-------------------------------
category_id | category_name
1           | Poetry
2           | Fiction
3           | Mystery
```

A book can then reference its category through category_id.

This avoids storing the same category name repeatedly and provides a normalized relational structure.

## SQL Queries

The pipeline executes six SQL queries demonstrating common SQL operations.

### Query 1 — WHERE

Find books costing more than £20.

```sql
SELECT
    title,
    price_gbp,
    rating
FROM books
WHERE price_gbp > 20
ORDER BY price_gbp DESC;
```

### Query 2 — ORDER BY / LIMIT

Find the 10 most expensive books.

```sql
SELECT
    title,
    price_gbp,
    rating
FROM books
ORDER BY price_gbp DESC
LIMIT 10;
```

### Query 3 — DISTINCT

Find the different ratings present in the dataset.

```sql
SELECT DISTINCT
    rating
FROM books
ORDER BY rating;
```

### Query 4 — BETWEEN

Find books priced between £10 and £30.

```sql
SELECT
    title,
    price_gbp
FROM books
WHERE price_gbp BETWEEN 10 AND 30
ORDER BY price_gbp;
```

### Query 5 — IN

Find books with ratings of 4 or 5.

```sql
SELECT
    title,
    rating,
    in_stock
FROM books
WHERE rating IN (4, 5)
ORDER BY rating DESC, title;
```

### Query 6 — JOIN

Combine book information with category information.

```sql
SELECT
    b.book_id,
    b.title,
    b.price_gbp,
    b.price_inr,
    b.rating,
    b.in_stock,
    c.category_name
FROM books AS b
INNER JOIN categories AS c
    ON b.category_id = c.category_id
ORDER BY b.book_id;
```

## Pandas Verification

The SQL JOIN is independently reproduced using `pandas.merge()`.

The pipeline compares:

```text
SQL JOIN
```

against:

```text
pandas.merge()
```

The final verification produced:

```text
SQL JOIN == pandas.merge: True
```

This confirms that the SQL JOIN and Pandas merge generated equivalent results.

## Generated Files

After successfully running the pipeline, the following files are generated:

```text
data_pipeline/
│
├── pipeline.py
├── zepto_catalog.db
├── sql_outputs.txt
└── README.md
```

### pipeline.py

The main Python data pipeline containing:

- Web scraping
- Parsing
- Cleaning
- Validation
- Database creation
- SQL queries
- Pandas verification

### zepto_catalog.db

SQLite database containing the cleaned catalogue.

### sql_outputs.txt

Contains the executed SQL queries and their output results.

### README.md

Project documentation.

## How to Run

Create and activate the Python virtual environment if required.

Install the required packages:

```bash
pip install requests beautifulsoup4 pandas
```

Then run:

```bash
python pipeline.py
```

## Expected Successful Output

A successful run ends with:

```text
================================================================================
JOIN VERIFICATION
================================================================================
SQL JOIN == pandas.merge: True

================================================================================
DATA PIPELINE COMPLETED SUCCESSFULLY
================================================================================
SQLite database: ...\zepto_catalog.db
SQL output file: ...\sql_outputs.txt
```

The exact file paths depend on the location of the project on the user's computer.

## Technologies Used

- Python
- Requests
- BeautifulSoup
- Pandas
- SQLite
- SQL
- Decimal arithmetic

## Module 1 Completion

Module 1 successfully demonstrates an end-to-end data engineering workflow:

```text
Web Scraping
     ↓
Data Extraction
     ↓
Data Cleaning
     ↓
Data Validation
     ↓
Currency Transformation
     ↓
SQLite Database
     ↓
SQL Analysis
     ↓
Pandas Verification
```

The completed pipeline successfully processed 100 books and verified the SQL JOIN against an independent Pandas merge.
