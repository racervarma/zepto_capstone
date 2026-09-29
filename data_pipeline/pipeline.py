from pathlib import Path
import re
import sqlite3

import pandas as pd
import requests
from bs4 import BeautifulSoup
from decimal import Decimal, ROUND_HALF_UP


# ============================================================
# Configuration
# ============================================================

BASE_URL = "https://books.toscrape.com/"
CATALOGUE_URL = BASE_URL + "catalogue/page-{}.html"

PROJECT_DIR = Path(__file__).resolve().parent
DB_PATH = PROJECT_DIR / "zepto_catalog.db"
SQL_OUTPUT_PATH = PROJECT_DIR / "sql_outputs.txt"

GBP_TO_INR = 105.50
PAGES_TO_SCRAPE = 5
MINIMUM_BOOKS = 60
MINIMUM_CATEGORIES = 3

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}


# ============================================================
# HTTP / HTML helpers
# ============================================================

def get_soup(url):
    """Download a webpage and return a BeautifulSoup object."""
    response = requests.get(
        url,
        timeout=20,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        },
    )

    response.raise_for_status()

    return BeautifulSoup(response.text, "html.parser")


# ============================================================
# Parsing
# ============================================================

def parse_price(price_element):
    """
    Parse a book price into GBP.

    The source site displays prices with a pound symbol.
    The parser removes the currency symbol and extracts the
    numeric value.
    """
    if price_element is None:
        return None

    text = price_element.get_text(" ", strip=True)

    # Remove everything except digits and decimal point.
    numeric_text = re.sub(r"[^0-9.]", "", text)

    if not numeric_text:
        return None

    try:
        return float(numeric_text)
    except ValueError:
        return None


def parse_rating(rating_element):
    """Convert One/Five-style star rating text to an integer."""
    if rating_element is None:
        return None

    classes = rating_element.get("class", [])

    for class_name in classes:
        if class_name in RATING_MAP:
            return RATING_MAP[class_name]

    return None


def parse_availability(availability_element):
    """
    Convert availability text into an integer boolean representation.

    SQLite does not have a dedicated BOOLEAN storage class, so:
        1 = True / in stock
        0 = False / not in stock
    """
    if availability_element is None:
        return None

    text = availability_element.get_text(
        " ",
        strip=True,
    ).lower()

    if "in stock" in text:
        return 1

    if "out of stock" in text:
        return 0

    return None


def parse_category(book_url):
    """Extract the category from the book's detail page."""
    soup = get_soup(book_url)

    breadcrumb = soup.select("ul.breadcrumb li")

    # Typical structure:
    # Home -> Books -> Category -> Book title
    if len(breadcrumb) >= 3:
        category = breadcrumb[-2].get_text(
            strip=True
        )

        return category

    return None


# ============================================================
# Scraping
# ============================================================

def scrape_books():
    """
    Scrape the first five pages of the All Products catalogue.

    Each listing contains:
        title
        price
        star_rating
        availability

    Category is obtained from the book's detail page.
    """
    books = []

    for page_number in range(1, PAGES_TO_SCRAPE + 1):

        page_url = CATALOGUE_URL.format(page_number)

        print(
            f"Scraping catalogue page {page_number}: "
            f"{page_url}"
        )

        soup = get_soup(page_url)

        products = soup.select(
            "article.product_pod"
        )

        print(
            f"  Books found: {len(products)}"
        )

        for product in products:

            try:
                title_element = product.select_one(
                    "h3 a"
                )

                price_element = product.select_one(
                    "p.price_color"
                )

                rating_element = product.select_one(
                    "p.star-rating"
                )

                availability_element = product.select_one(
                    "p.availability"
                )

                if title_element is None:
                    raise ValueError(
                        "Title could not be parsed."
                    )

                title = (
                    title_element.get("title")
                    or title_element.get_text(
                        strip=True
                    )
                )

                relative_url = title_element.get(
                    "href"
                )

                if not relative_url:
                    raise ValueError(
                        "Book detail URL missing."
                    )

                detail_url = requests.compat.urljoin(
                    page_url,
                    relative_url,
                )

                price_text = (
                    price_element.get_text(
                        " ",
                        strip=True,
                    )
                    if price_element
                    else None
                )

                rating_text = None

                if rating_element is not None:
                    for class_name in rating_element.get(
                        "class",
                        [],
                    ):
                        if class_name in RATING_MAP:
                            rating_text = class_name
                            break

                availability_text = (
                    availability_element.get_text(
                        " ",
                        strip=True,
                    )
                    if availability_element
                    else None
                )

                category = parse_category(
                    detail_url
                )

                books.append(
                    {
                        "title": title,
                        "price": price_text,
                        "star_rating": rating_text,
                        "availability": availability_text,
                        "category": category,
                    }
                )

            except (
                requests.RequestException,
                AttributeError,
                TypeError,
                ValueError,
            ) as error:

                print(
                    f"  Skipping malformed book: {error}"
                )

                continue

    return books


# ============================================================
# Cleaning
# ============================================================

def clean_books(raw_books):
    """
    Clean the scraped records.

    If a required field cannot be parsed, the row is dropped.
    This is safer for catalog integrity than inventing a price,
    rating, availability value, or category.

    The project requires the fixed conversion:
        1 GBP = 105.50 INR

    Decimal arithmetic is used for the currency conversion so
    that binary floating-point rounding does not produce
    incorrect two-decimal INR values.
    """

    cleaned_rows = []

    failure_counts = {
        "title": 0,
        "price": 0,
        "rating": 0,
        "availability": 0,
        "category": 0,
    }

    for book in raw_books:

        title = book.get("title")

        price_gbp = parse_price_from_text(
            book.get("price")
        )

        rating = RATING_MAP.get(
            book.get("star_rating")
        )

        in_stock = parse_availability_text(
            book.get("availability")
        )

        category = book.get("category")

        # ----------------------------------------------------
        # Required field validation
        # ----------------------------------------------------

        if not title:
            failure_counts["title"] += 1
            continue

        if price_gbp is None:
            failure_counts["price"] += 1
            continue

        if rating not in {1, 2, 3, 4, 5}:
            failure_counts["rating"] += 1
            continue

        if in_stock not in {0, 1}:
            failure_counts["availability"] += 1
            continue

        if not category:
            failure_counts["category"] += 1
            continue

        # ----------------------------------------------------
        # Fixed GBP -> INR conversion
        # ----------------------------------------------------
        #
        # Required project rate:
        #     1 GBP = 105.50 INR
        #
        # Decimal is used instead of ordinary float arithmetic
        # so values exactly halfway between two paise are rounded
        # correctly using ROUND_HALF_UP.
        # ----------------------------------------------------

        price_inr = (
            Decimal(str(price_gbp))
            * Decimal("105.50")
        ).quantize(
            Decimal("0.01"),
            rounding=ROUND_HALF_UP,
        )

        cleaned_rows.append(
            {
                "title": title.strip(),

                "price_gbp": float(
                    price_gbp
                ),

                "price_inr": float(
                    price_inr
                ),

                "rating": int(
                    rating
                ),

                "in_stock": bool(
                    in_stock
                ),

                "category": category.strip(),
            }
        )

    cleaned_df = pd.DataFrame(
        cleaned_rows
    )

    # --------------------------------------------------------
    # Cleaning diagnostics
    # --------------------------------------------------------

    print("\nCleaning diagnostics")
    print("-" * 60)

    print(
        f"Raw rows:              "
        f"{len(raw_books)}"
    )

    print(
        f"Clean rows:            "
        f"{len(cleaned_df)}"
    )

    print(
        f"Invalid titles:        "
        f"{failure_counts['title']}"
    )

    print(
        f"Invalid prices:        "
        f"{failure_counts['price']}"
    )

    print(
        f"Invalid ratings:       "
        f"{failure_counts['rating']}"
    )

    print(
        f"Invalid availability:  "
        f"{failure_counts['availability']}"
    )

    print(
        f"Invalid categories:    "
        f"{failure_counts['category']}"
    )

    return cleaned_df


def parse_price_from_text(price_text):
    """Parse price text such as £51.77 into float 51.77."""
    if not price_text:
        return None

    numeric_text = re.sub(
        r"[^0-9.]",
        "",
        str(price_text),
    )

    if not numeric_text:
        return None

    try:
        return float(numeric_text)
    except ValueError:
        return None


def parse_availability_text(availability_text):
    """Convert availability text into 1/0."""
    if not availability_text:
        return None

    text = str(
        availability_text
    ).lower()

    if "in stock" in text:
        return 1

    if "out of stock" in text:
        return 0

    return None


# ============================================================
# Database
# ============================================================

def create_database(connection):
    """Create the normalized SQLite database schema."""

    cursor = connection.cursor()

    cursor.execute(
        "DROP TABLE IF EXISTS books"
    )

    cursor.execute(
        "DROP TABLE IF EXISTS categories"
    )

    cursor.execute(
        """
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY,
            category_name TEXT UNIQUE
        )
        """
    )

    cursor.execute(
        """
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
        )
        """
    )

    connection.commit()


def insert_data(connection, dataframe):
    """Insert categories and books into SQLite."""

    categories = sorted(
        dataframe["category"]
        .unique()
        .tolist()
    )

    connection.executemany(
        """
        INSERT INTO categories (
            category_name
        )
        VALUES (?)
        """,
        [
            (category,)
            for category in categories
        ],
    )

    category_df = pd.read_sql(
        """
        SELECT
            category_id,
            category_name
        FROM categories
        """,
        connection,
    )

    merged_df = dataframe.merge(
        category_df,
        left_on="category",
        right_on="category_name",
        how="inner",
    )

    book_rows = []

    for row in merged_df.itertuples(
        index=False
    ):

        book_rows.append(
            (
                row.title,
                row.price_gbp,
                row.price_inr,
                int(row.rating),
                int(row.in_stock),
                int(row.category_id),
            )
        )

    connection.executemany(
        """
        INSERT INTO books (
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            category_id
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        book_rows,
    )

    connection.commit()


# ============================================================
# SQL queries
# ============================================================

SQL_QUERIES = [
    (
        "Query 1 - SELECT / WHERE",
        """
        SELECT
            title,
            price_gbp,
            rating
        FROM books
        WHERE price_gbp > 20
        ORDER BY price_gbp DESC
        """,
    ),
    (
        "Query 2 - ORDER BY / LIMIT",
        """
        SELECT
            title,
            price_gbp,
            rating
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10
        """,
    ),
    (
        "Query 3 - DISTINCT",
        """
        SELECT DISTINCT
            rating
        FROM books
        ORDER BY rating
        """,
    ),
    (
        "Query 4 - BETWEEN",
        """
        SELECT
            title,
            price_gbp
        FROM books
        WHERE price_gbp BETWEEN 10 AND 30
        ORDER BY price_gbp
        """,
    ),
    (
        "Query 5 - IN",
        """
        SELECT
            title,
            rating,
            in_stock
        FROM books
        WHERE rating IN (4, 5)
        ORDER BY rating DESC, title
        """,
    ),
    (
        "Query 6 - JOIN",
        """
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
        ORDER BY b.book_id
        """,
    ),
]


def execute_sql_queries(connection):
    """
    Execute all required SQL queries.

    Results are printed to the console and saved to
    sql_outputs.txt so the executed query strings and
    outputs are preserved as project artifacts.
    """

    output_lines = []

    for query_name, query in SQL_QUERIES:

        cleaned_query = query.strip()

        result_df = pd.read_sql(
            cleaned_query,
            connection,
        )

        separator = "=" * 80

        output_lines.append(
            separator
        )
        output_lines.append(
            query_name
        )
        output_lines.append(
            separator
        )
        output_lines.append(
            cleaned_query
        )
        output_lines.append("")
        output_lines.append(
            result_df.to_string(
                index=False
            )
        )
        output_lines.append("")

        print("\n" + separator)
        print(query_name)
        print(separator)
        print(cleaned_query)
        print("\nOutput:")
        print(
            result_df.to_string(
                index=False
            )
        )

    SQL_OUTPUT_PATH.write_text(
        "\n".join(output_lines),
        encoding="utf-8",
    )

    print(
        f"\nSQL query results saved to:"
        f"\n{SQL_OUTPUT_PATH}"
    )


# ============================================================
# Pandas verification
# ============================================================

def verify_with_pandas(connection):
    """
    Read query results into pandas and reproduce the JOIN
    independently using pd.merge.
    """

    # --------------------------------------------------------
    # Query result 1 via pd.read_sql
    # --------------------------------------------------------

    expensive_books = pd.read_sql(
        """
        SELECT
            title,
            price_gbp,
            rating
        FROM books
        WHERE price_gbp > 20
        """,
        connection,
    )

    print("\n" + "=" * 80)
    print("PANDAS QUERY RESULT #1")
    print("=" * 80)
    print(
        expensive_books.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # JOIN result via pd.read_sql
    # --------------------------------------------------------

    sql_join = pd.read_sql(
        """
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
        ORDER BY b.book_id
        """,
        connection,
    )

    # --------------------------------------------------------
    # Read source tables into pandas.
    # --------------------------------------------------------

    books_df = pd.read_sql(
        """
        SELECT
            book_id,
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            category_id
        FROM books
        """,
        connection,
    )

    categories_df = pd.read_sql(
        """
        SELECT
            category_id,
            category_name
        FROM categories
        """,
        connection,
    )

    # --------------------------------------------------------
    # Reproduce SQL JOIN with pandas.merge.
    # --------------------------------------------------------

    pandas_join = pd.merge(
        books_df,
        categories_df,
        on="category_id",
        how="inner",
    )

    pandas_join = pandas_join[
        [
            "book_id",
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category_name",
        ]
    ]

    pandas_join = (
        pandas_join
        .sort_values("book_id")
        .reset_index(drop=True)
    )

    sql_join = (
        sql_join
        .sort_values("book_id")
        .reset_index(drop=True)
    )

    # Ensure comparable dtypes.
    sql_join["book_id"] = (
        sql_join["book_id"].astype(int)
    )

    pandas_join["book_id"] = (
        pandas_join["book_id"].astype(int)
    )

    # Compare exact values.
    matches = sql_join.equals(
        pandas_join
    )

    print("\n" + "=" * 80)
    print("SQL JOIN RESULT")
    print("=" * 80)
    print(
        sql_join.to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("PANDAS MERGE RESULT")
    print("=" * 80)
    print(
        pandas_join.to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("JOIN VERIFICATION")
    print("=" * 80)
    print(
        f"SQL JOIN == pandas.merge: {matches}"
    )

    if not matches:
        raise AssertionError(
            "The SQL JOIN result does not match "
            "the pandas.merge result."
        )

    # Save verification to the SQL output artifact.
    with SQL_OUTPUT_PATH.open(
        "a",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n\n"
            + "=" * 80
            + "\n"
        )

        file.write(
            "PANDAS JOIN VERIFICATION\n"
        )

        file.write(
            "=" * 80
            + "\n\n"
        )

        file.write(
            "SQL JOIN RESULT\n\n"
        )

        file.write(
            sql_join.to_string(
                index=False
            )
        )

        file.write(
            "\n\nPANDAS MERGE RESULT\n\n"
        )

        file.write(
            pandas_join.to_string(
                index=False
            )
        )

        file.write(
            "\n\n"
            f"SQL JOIN == pandas.merge: "
            f"{matches}\n"
        )


# ============================================================
# Validation
# ============================================================

def validate_dataset(dataframe):
    """Validate the project's minimum acceptance criteria."""

    if len(dataframe) < MINIMUM_BOOKS:
        raise RuntimeError(
            f"Only {len(dataframe)} valid books "
            f"remain. At least {MINIMUM_BOOKS} are required."
        )

    category_count = dataframe["category"].nunique()

    if category_count < MINIMUM_CATEGORIES:
        raise RuntimeError(
            f"Only {category_count} categories "
            f"were found. At least "
            f"{MINIMUM_CATEGORIES} are required."
        )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = {
        "title",
        "price_gbp",
        "price_inr",
        "rating",
        "in_stock",
        "category",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    # --------------------------------------------------------
    # Rating validation
    # --------------------------------------------------------

    if not dataframe["rating"].between(
        1,
        5,
    ).all():
        raise RuntimeError(
            "Rating contains values outside 1-5."
        )

    # --------------------------------------------------------
    # Missing-value validation
    # --------------------------------------------------------

    missing_values = (
        dataframe[
            list(required_columns)
        ]
        .isnull()
        .sum()
    )

    if missing_values.any():

        print("\nMissing values found:")

        print(
            missing_values[
                missing_values > 0
            ].to_string()
        )

        raise RuntimeError(
            "Required columns contain "
            "missing values."
        )

    # --------------------------------------------------------
    # Boolean validation
    # --------------------------------------------------------

    if not dataframe["in_stock"].isin(
        [True, False]
    ).all():

        raise RuntimeError(
            "in_stock contains values other "
            "than True or False."
        )

    # --------------------------------------------------------
    # Price conversion validation
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # clean_books() uses Decimal arithmetic with:
    #
    #     ROUND_HALF_UP
    #
    # Therefore validation MUST use the exact same
    # calculation. Do not use pandas .round(2) here.
    #
    # Example:
    #
    #     54.23 × 105.50 = 5721.265
    #
    # ROUND_HALF_UP:
    #
    #     5721.27
    #
    # --------------------------------------------------------

    expected_price_inr = []

    for price_gbp in dataframe["price_gbp"]:

        expected = (
            Decimal(str(price_gbp))
            * Decimal(str(GBP_TO_INR))
        ).quantize(
            Decimal("0.01"),
            rounding=ROUND_HALF_UP,
        )

        expected_price_inr.append(
            expected
        )

    actual_price_inr = []

    for price_inr in dataframe["price_inr"]:

        actual = Decimal(
            str(price_inr)
        ).quantize(
            Decimal("0.01"),
            rounding=ROUND_HALF_UP,
        )

        actual_price_inr.append(
            actual
        )

    # --------------------------------------------------------
    # Compare Decimal values directly.
    # --------------------------------------------------------

    conversion_matches = []

    for expected, actual in zip(
        expected_price_inr,
        actual_price_inr,
    ):

        conversion_matches.append(
            expected == actual
        )

    if not all(conversion_matches):

        mismatches = []

        for index, matches in enumerate(
            conversion_matches
        ):

            if not matches:

                mismatches.append(
                    {
                        "price_gbp": dataframe.iloc[
                            index
                        ]["price_gbp"],

                        "expected_price_inr": float(
                            expected_price_inr[index]
                        ),

                        "actual_price_inr": float(
                            actual_price_inr[index]
                        ),
                    }
                )

        mismatch_df = pd.DataFrame(
            mismatches
        )

        print(
            "\nPrice conversion mismatches:"
        )

        print(
            mismatch_df.head(10).to_string(
                index=False
            )
        )

        raise RuntimeError(
            "price_inr does not match "
            "the required 105.50 INR/GBP rate."
        )

    # --------------------------------------------------------
    # Validation summary
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("DATASET VALIDATION")
    print("=" * 80)

    print(
        f"Valid books:       {len(dataframe)}"
    )

    print(
        f"Unique categories: {category_count}"
    )

    print(
        f"GBP → INR rate:    {GBP_TO_INR}"
    )

    print(
        "Required columns:  PASS"
    )

    print(
        "Rating 1-5:        PASS"
    )

    print(
        "Price conversion:  PASS"
    )

    print(
        "Missing values:    PASS"
    )

    print(
        "Boolean values:    PASS"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("Zepto Data & AI Platform")
    print("Module 1 - Data Pipeline")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Scrape
    # --------------------------------------------------------

    raw_books = scrape_books()

    print(
        f"\nRaw books scraped: "
        f"{len(raw_books)}"
    )

    if len(raw_books) < MINIMUM_BOOKS:
        raise RuntimeError(
            f"Scraping produced only "
            f"{len(raw_books)} books."
        )

    # --------------------------------------------------------
    # 2. Clean
    # --------------------------------------------------------

    cleaned_df = clean_books(
        raw_books
    )

    print(
        "\nCleaned dataframe preview:"
    )

    print(
        cleaned_df.head().to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 3. Validate
    # --------------------------------------------------------

    validate_dataset(
        cleaned_df
    )

    # --------------------------------------------------------
    # 4. Create SQLite database
    # --------------------------------------------------------

    connection = sqlite3.connect(
        DB_PATH
    )

    try:

        # Explicitly enable foreign-key enforcement.
        connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        create_database(
            connection
        )

        insert_data(
            connection,
            cleaned_df,
        )

        # ----------------------------------------------------
        # 5. Database validation
        # ----------------------------------------------------

        book_count = pd.read_sql(
            """
            SELECT COUNT(*) AS count
            FROM books
            """,
            connection,
        ).iloc[0]["count"]

        category_count = pd.read_sql(
            """
            SELECT COUNT(*) AS count
            FROM categories
            """,
            connection,
        ).iloc[0]["count"]

        # Verify the database contains the expected data.
        if int(book_count) < MINIMUM_BOOKS:
            raise RuntimeError(
                f"Database contains only "
                f"{int(book_count)} books."
            )

        if int(category_count) < MINIMUM_CATEGORIES:
            raise RuntimeError(
                f"Database contains only "
                f"{int(category_count)} categories."
            )

        print("\n" + "=" * 80)
        print("DATABASE CREATED")
        print("=" * 80)

        print(
            f"Database:    {DB_PATH}"
        )

        print(
            f"Books:       {int(book_count)}"
        )

        print(
            f"Categories:  {int(category_count)}"
        )

        # ----------------------------------------------------
        # 6. Execute SQL queries
        # ----------------------------------------------------

        execute_sql_queries(
            connection
        )

        # ----------------------------------------------------
        # 7. Pandas verification
        # ----------------------------------------------------

        verify_with_pandas(
            connection
        )

    finally:

        connection.close()

    # --------------------------------------------------------
    # 8. Completion message
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("DATA PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)

    print(
        f"SQLite database: {DB_PATH}"
    )

    print(
        f"SQL output file: {SQL_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
